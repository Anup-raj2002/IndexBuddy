"""
candidates/generator.py
───────────────────────
Generates single-column and two-column B-tree candidate indexes for a parsed SQL
query, capped at ``max_candidates`` per query (default from config.py).

Zero hardcoded table or column names. Primary key and TEXT/BLOB columns are
dynamically excluded via table introspection when available.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Set, Tuple

import config
from parsing.query_parser import ParsedQuery, parse_query


@dataclass(frozen=True)
class CandidateIndex:
    """Represents a single-column or two-column candidate B-tree index."""

    table: str
    columns: Tuple[str, ...]
    index_name: str
    ddl: str
    source_clauses: Tuple[str, ...]

    @property
    def num_columns(self) -> int:
        return len(self.columns)

    @property
    def is_composite(self) -> bool:
        return len(self.columns) > 1

    @property
    def columns_display(self) -> str:
        return ", ".join(self.columns)


def build_index_name(table: str, columns: Sequence[str]) -> str:
    """Return deterministic MySQL index name ``idx_ib_<table>_<col1>[_<col2>]``."""
    clean_tbl = "".join(ch if ch.isalnum() else "_" for ch in table)
    clean_cols = "_".join("".join(ch if ch.isalnum() else "_" for ch in c) for c in columns)
    raw = f"idx_ib_{clean_tbl}_{clean_cols}"
    return raw[:64]


def build_create_index_ddl(
    table: str,
    columns: Sequence[str],
    index_name: Optional[str] = None,
    invisible: bool = False,
) -> str:
    """Return standard MySQL ``CREATE INDEX`` statement."""
    idx_name = index_name or build_index_name(table, columns)
    cols_sql = ", ".join(f"`{c}`" for c in columns)
    vis = " INVISIBLE" if invisible else ""
    return f"CREATE INDEX `{idx_name}` ON `{table}` ({cols_sql}){vis};"


def _column_clause_tags(parsed: ParsedQuery, col: str) -> Tuple[str, ...]:
    tags: List[str] = []
    if col in parsed.where_columns:
        p_type = parsed.predicate_types.get(col, "other")
        tags.append(f"where:{p_type}")
    if col in parsed.join_columns:
        tags.append("join")
    if col in parsed.group_by_columns:
        tags.append("group_by")
    if col in parsed.order_by_columns:
        tags.append("order_by")
    return tuple(tags) if tags else ("referenced",)


def generate_candidates(
    query_or_parsed: str | ParsedQuery,
    max_candidates: int = config.MAX_CANDIDATES_PER_QUERY,
    excluded_columns: Optional[Iterable[str]] = None,
    single_pk_column: Optional[str] = None,
) -> List[CandidateIndex]:
    """Generate single-column and two-column candidate indexes for a SQL query.

    Parameters
    ----------
    query_or_parsed : str | ParsedQuery
        Either a raw SQL string or an already-parsed ``ParsedQuery`` object.
    max_candidates : int
        Maximum number of candidate indexes returned per query.
    excluded_columns : Optional[Iterable[str]]
        Columns that cannot be indexed directly without prefix lengths (e.g. TEXT/BLOB).
    single_pk_column : Optional[str]
        If the table has a single-column PRIMARY KEY, skip generating a redundant
        1-column secondary index on that exact column alone.
    """
    parsed = (
        parse_query(query_or_parsed)
        if isinstance(query_or_parsed, str)
        else query_or_parsed
    )
    excluded: Set[str] = (
        {c.lower() for c in excluded_columns}
        if excluded_columns is not None
        else set()
    )
    pk_low = single_pk_column.lower() if single_pk_column else None
    table = parsed.primary_table

    # Map lowercase column name -> canonical column name from parsed query
    ordered_cols: List[str] = []
    seen_low: Set[str] = set()
    for col in (
        parsed.where_columns
        + parsed.join_columns
        + parsed.group_by_columns
        + parsed.order_by_columns
    ):
        col_low = col.lower()
        if col_low in excluded:
            continue
        if (
            parsed.predicate_types.get(col) == "like_wildcard"
            and col not in parsed.join_columns
            and col not in parsed.group_by_columns
            and col not in parsed.order_by_columns
        ):
            continue
        if col_low not in seen_low:
            seen_low.add(col_low)
            ordered_cols.append(col)

    candidates: List[CandidateIndex] = []
    seen_tuples: Set[Tuple[str, ...]] = set()

    def _add_candidate(cols: Tuple[str, ...]) -> None:
        key_low = tuple(c.lower() for c in cols)
        if len(cols) == 1 and pk_low and key_low[0] == pk_low:
            return
        if key_low in seen_tuples or len(candidates) >= max_candidates:
            return
        seen_tuples.add(key_low)
        idx_name = build_index_name(table, cols)
        ddl = build_create_index_ddl(table, cols, idx_name)
        tags: List[str] = []
        for c in cols:
            for t in _column_clause_tags(parsed, c):
                if t not in tags:
                    tags.append(t)
        candidates.append(
            CandidateIndex(
                table=table,
                columns=cols,
                index_name=idx_name,
                ddl=ddl,
                source_clauses=tuple(tags),
            )
        )

    # 1. Single-column candidates
    for col in ordered_cols:
        _add_candidate((col,))

    # 2. Two-column composite candidates
    leading_preferred = [
        c
        for c in ordered_cols
        if parsed.predicate_types.get(c) in ("equality", "in")
        or c in parsed.join_columns
    ]
    other_cols = [c for c in ordered_cols if c not in leading_preferred]

    for col1 in leading_preferred + other_cols:
        for col2 in ordered_cols:
            if col1.lower() != col2.lower():
                _add_candidate((col1, col2))

    return candidates[:max_candidates]
