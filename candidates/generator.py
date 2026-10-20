"""
candidates/generator.py
───────────────────────
Generates single-column and two-column B-tree candidate indexes for a parsed SQL
query, capped at ``max_candidates`` per query (default from config.py).

Preserves the clean ``CREATE INDEX idx_<table>_<cols> ON <table> (<cols>)`` DDL
format from the original IndexBuddy recommender while delegating all scoring to
the downstream ML model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Set, Tuple

import config
from parsing.query_parser import ParsedQuery, parse_query


# Default columns excluded from standard B-tree secondary index generation:
# - 'order_id' is already the clustered PRIMARY KEY on orders_benchmark
# - 'notes' is a TEXT column (MySQL InnoDB requires explicit key prefix length on TEXT/BLOB)
DEFAULT_EXCLUDED_COLUMNS: Set[str] = {"order_id", "notes"}


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
    """Return deterministic MySQL index name ``idx_<table>_<col1>[_<col2>]``."""
    raw = f"idx_{table}_{'_'.join(columns)}"
    return raw[:64]  # MySQL identifier limit is 64 chars


def build_create_index_ddl(table: str, columns: Sequence[str], index_name: Optional[str] = None) -> str:
    """Return standard MySQL ``CREATE INDEX`` statement."""
    idx_name = index_name or build_index_name(table, columns)
    cols_sql = ", ".join(f"`{c}`" for c in columns)
    return f"CREATE INDEX `{idx_name}` ON `{table}` ({cols_sql});"


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
) -> List[CandidateIndex]:
    """Generate single-column and two-column candidate indexes for a SQL query.

    Parameters
    ----------
    query_or_parsed : str | ParsedQuery
        Either a raw SQL string or an already-parsed ``ParsedQuery`` object.
    max_candidates : int
        Maximum number of candidate indexes returned per query.
    excluded_columns : Optional[Iterable[str]]
        Columns to skip (defaults to ``DEFAULT_EXCLUDED_COLUMNS``).
    """
    parsed = (
        parse_query(query_or_parsed)
        if isinstance(query_or_parsed, str)
        else query_or_parsed
    )
    excluded = (
        {c.lower() for c in excluded_columns}
        if excluded_columns is not None
        else set(DEFAULT_EXCLUDED_COLUMNS)
    )
    table = parsed.primary_table

    # Collect candidate columns in clause priority order:
    # 1) WHERE + JOIN columns, 2) GROUP BY columns, 3) ORDER BY columns
    ordered_cols: List[str] = []
    for col in (
        parsed.where_columns
        + parsed.join_columns
        + parsed.group_by_columns
        + parsed.order_by_columns
    ):
        col_low = col.lower()
        if col_low in excluded:
            continue
        # Skip leading-wildcard LIKE columns if they have no other role
        if (
            parsed.predicate_types.get(col_low) == "like_wildcard"
            and col_low not in parsed.join_columns
            and col_low not in parsed.group_by_columns
            and col_low not in parsed.order_by_columns
        ):
            continue
        if col_low not in ordered_cols:
            ordered_cols.append(col_low)

    candidates: List[CandidateIndex] = []
    seen_tuples: Set[Tuple[str, ...]] = set()

    def _add_candidate(cols: Tuple[str, ...]) -> None:
        if cols in seen_tuples or len(candidates) >= max_candidates:
            return
        seen_tuples.add(cols)
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

    # 1. All single-column candidates
    for col in ordered_cols:
        _add_candidate((col,))

    # 2. Two-column composite candidates
    # Prioritize equality / IN / JOIN columns in leading position (col1),
    # followed by remaining referenced columns in second position (col2),
    # then include remaining ordered pairs up to max_candidates.
    leading_preferred = [
        c
        for c in ordered_cols
        if parsed.predicate_types.get(c) in ("equality", "in")
        or c in parsed.join_columns
    ]
    other_cols = [c for c in ordered_cols if c not in leading_preferred]

    for col1 in leading_preferred + other_cols:
        for col2 in ordered_cols:
            if col1 != col2:
                _add_candidate((col1, col2))

    return candidates[:max_candidates]


if __name__ == "__main__":
    sample_queries = [
        (
            "SELECT * FROM orders_benchmark "
            "WHERE customer_id = 48219 AND status = 'completed' "
            "ORDER BY order_date DESC LIMIT 25"
        ),
        (
            "SELECT o.order_id, o.amount, c.name "
            "FROM orders_benchmark o "
            "JOIN customers_benchmark c ON o.customer_id = c.customer_id "
            "WHERE o.product_id IN (101, 202, 303) "
            "AND o.order_date BETWEEN '2023-01-01' AND '2023-06-30'"
        ),
        (
            "SELECT region, payment_method, SUM(amount) "
            "FROM orders_benchmark "
            "WHERE payment_method LIKE 'credit%' "
            "GROUP BY region, payment_method"
        ),
    ]
    for idx, q in enumerate(sample_queries, 1):
        p = parse_query(q)
        cands = generate_candidates(p)
        print(f"\n=== Sample Query {idx} ===")
        print(f"SQL          : {p.raw_sql}")
        print(f"Template     : {p.template_sql}")
        print(f"Primary table: {p.primary_table}")
        print(f"WHERE cols   : {p.where_columns}")
        print(f"JOIN cols    : {p.join_columns}")
        print(f"GROUP BY cols: {p.group_by_columns}")
        print(f"ORDER BY cols: {p.order_by_columns}")
        print(f"Predicates   : {p.predicate_types}")
        print(f"Candidates ({len(cands)}):")
        for c in cands:
            print(f"  - ({c.columns_display:<26}) | clauses={list(c.source_clauses)} | {c.ddl}")
