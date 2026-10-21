"""
features/extract.py
───────────────────
Extracts schema-agnostic catalog statistics, query structure, predicate
selectivity, candidate index column-position features, baseline EXPLAIN plan
features, and estimated B-tree index size for (query, candidate_index) pairs.

The ML model never sees table or column names. All statistics come from the live
MySQL 8.x database (information_schema, COUNT(DISTINCT), value distributions,
and EXPLAIN), with automatic subquery sampling on very large tables.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

import config
from candidates.generator import CandidateIndex
from db.mysql_adapter import MySQLAdapter
from parsing.query_parser import ParsedQuery, PredicateInfo, parse_query


DATA_TYPE_CODE: Dict[str, int] = {
    "int": 1,
    "integer": 1,
    "bigint": 1,
    "mediumint": 1,
    "smallint": 1,
    "tinyint": 1,
    "year": 1,
    "decimal": 2,
    "numeric": 2,
    "float": 2,
    "double": 2,
    "real": 2,
    "date": 3,
    "datetime": 3,
    "timestamp": 3,
    "time": 3,
    "varchar": 4,
    "char": 4,
    "enum": 4,
    "set": 4,
    "text": 5,
    "mediumtext": 5,
    "longtext": 5,
    "tinytext": 5,
    "blob": 5,
    "mediumblob": 5,
    "longblob": 5,
}

DEFAULT_TYPE_WIDTH: Dict[str, int] = {
    "int": 4,
    "integer": 4,
    "bigint": 8,
    "mediumint": 3,
    "smallint": 2,
    "tinyint": 1,
    "year": 1,
    "decimal": 6,
    "numeric": 6,
    "float": 4,
    "double": 8,
    "date": 3,
    "datetime": 8,
    "timestamp": 4,
    "time": 3,
    "varchar": 24,
    "char": 16,
    "enum": 2,
    "set": 4,
    "text": 64,
    "mediumtext": 64,
    "longtext": 64,
    "blob": 64,
}

UNINDEXABLE_TYPES = {
    "text",
    "mediumtext",
    "longtext",
    "tinytext",
    "blob",
    "mediumblob",
    "longblob",
    "tinyblob",
    "json",
    "geometry",
}

EXPLAIN_ACCESS_TYPE_CODE: Dict[str, int] = {
    "all": 0,
    "index": 1,
    "range": 2,
    "index_merge": 2,
    "ref_or_null": 3,
    "fulltext": 3,
    "ref": 4,
    "eq_ref": 5,
    "const": 6,
    "system": 6,
}


@dataclass
class ColumnStats:
    """Live database statistics for a single column."""

    column_name: str
    data_type: str
    data_type_code: int
    byte_width: int
    is_nullable: bool
    is_primary_key: bool
    is_indexable: bool
    cardinality: int
    distinct_ratio: float
    null_fraction: float
    min_value: Any = None
    max_value: Any = None
    value_frequencies: Dict[str, float] = field(default_factory=dict)
    sample_values: List[Any] = field(default_factory=list)


@dataclass
class TableStats:
    """Live database statistics for a table and all its columns."""

    database_name: str
    table_name: str
    table_rows: int
    avg_row_length: int
    data_length_mb: float
    primary_key_columns: List[str] = field(default_factory=list)
    columns: Dict[str, ColumnStats] = field(default_factory=dict)
    canonical_columns: Dict[str, str] = field(default_factory=dict)  # lower -> actual

    def get_col(self, col_name: str) -> Optional[ColumnStats]:
        if col_name in self.columns:
            return self.columns[col_name]
        canon = self.canonical_columns.get(col_name.lower())
        if canon and canon in self.columns:
            return self.columns[canon]
        return None

    def excluded_columns(self) -> List[str]:
        """Return columns that cannot be B-tree indexed without a prefix length."""
        return [
            cs.column_name
            for cs in self.columns.values()
            if not cs.is_indexable
        ]

    def single_pk_column(self) -> Optional[str]:
        if len(self.primary_key_columns) == 1:
            return self.primary_key_columns[0]
        return None


@dataclass
class BaselineExplainInfo:
    """Features extracted from MySQL EXPLAIN on the unindexed baseline query."""

    access_type: str
    access_type_code: int
    estimated_rows: int
    filtered_pct: float
    uses_filesort: int
    uses_temporary: int
    raw_rows: List[Dict[str, Any]] = field(default_factory=list)


class StatsCache:
    """Caches table and column statistics from live MySQL to avoid redundant scans."""

    def __init__(self, db: MySQLAdapter) -> None:
        self.db = db
        self._tables: Dict[str, TableStats] = {}
        self._explain_cache: Dict[str, BaselineExplainInfo] = {}

    def clear(self) -> None:
        self._tables.clear()
        self._explain_cache.clear()

    def get_table_stats(self, table: str) -> TableStats:
        cache_key = f"{self.db.database}.{table.lower()}"
        if cache_key in self._tables:
            return self._tables[cache_key]

        # Resolve exact case-sensitive table name in MySQL
        tbl_meta = self.db.execute(
            "SELECT TABLE_NAME, TABLE_ROWS, AVG_ROW_LENGTH, DATA_LENGTH "
            "FROM information_schema.TABLES "
            "WHERE TABLE_SCHEMA = %s AND LOWER(TABLE_NAME) = LOWER(%s)",
            (self.db.database, table),
        )
        actual_table = str(tbl_meta[0]["TABLE_NAME"]) if tbl_meta else table
        avg_row_len = int(tbl_meta[0]["AVG_ROW_LENGTH"] or 80) if tbl_meta else 80
        data_len_mb = (
            float(tbl_meta[0]["DATA_LENGTH"] or 0) / (1024.0 * 1024.0)
            if tbl_meta
            else 0.0
        )

        total_rows = self.db.row_count(actual_table)
        pk_cols = self.db.get_primary_key_columns(actual_table)
        pk_set_low = {c.lower() for c in pk_cols}

        col_rows = self.db.execute(
            "SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, "
            "CHARACTER_MAXIMUM_LENGTH, NUMERIC_PRECISION "
            "FROM information_schema.COLUMNS "
            "WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s "
            "ORDER BY ORDINAL_POSITION",
            (self.db.database, actual_table),
        )

        # Use bounded subquery sampling on very large tables (>200k rows)
        use_sample = total_rows > config.LARGE_TABLE_SAMPLE_THRESHOLD
        sample_n = config.SAMPLE_SIZE_ROWS
        if use_sample:
            source_sql = f"(SELECT * FROM `{actual_table}` LIMIT {sample_n}) AS _ib_sample"
            denom_rows = min(total_rows, sample_n)
        else:
            source_sql = f"`{actual_table}`"
            denom_rows = max(1, total_rows)

        select_exprs: List[str] = []
        col_meta_list: List[Dict[str, Any]] = []
        for idx, r in enumerate(col_rows):
            c_name = str(r["COLUMN_NAME"])
            d_type = str(r["DATA_TYPE"]).lower()
            col_meta_list.append(r)
            if d_type not in UNINDEXABLE_TYPES:
                select_exprs.append(f"COUNT(DISTINCT `{c_name}`) AS `card_{idx}`")
                select_exprs.append(f"MIN(`{c_name}`) AS `min_{idx}`")
                select_exprs.append(f"MAX(`{c_name}`) AS `max_{idx}`")
            select_exprs.append(
                f"SUM(CASE WHEN `{c_name}` IS NULL THEN 1 ELSE 0 END) AS `null_{idx}`"
            )

        agg_result: Dict[str, Any] = {}
        if select_exprs and total_rows > 0:
            sql_agg = f"SELECT {', '.join(select_exprs)} FROM {source_sql}"
            try:
                rows_agg = self.db.execute(sql_agg)
                if rows_agg:
                    agg_result = rows_agg[0]
            except Exception:
                agg_result = {}

        columns_dict: Dict[str, ColumnStats] = {}
        canon_map: Dict[str, str] = {}

        for idx, r in enumerate(col_meta_list):
            c_name = str(r["COLUMN_NAME"])
            c_low = c_name.lower()
            canon_map[c_low] = c_name
            d_type = str(r["DATA_TYPE"]).lower()
            is_null = str(r["IS_NULLABLE"]).upper() == "YES"
            is_pk = c_low in pk_set_low
            char_len = r.get("CHARACTER_MAXIMUM_LENGTH")

            if d_type in ("varchar", "char"):
                if char_len and int(char_len) > 0:
                    width = min(int(char_len), 64)
                else:
                    width = 24
                # Very wide VARCHAR (>768 chars) in utf8mb4 exceeds InnoDB 3072-byte key limit
                is_indexable = (int(char_len or 0) <= 500)
            elif d_type in UNINDEXABLE_TYPES:
                width = 64
                is_indexable = False
            else:
                width = DEFAULT_TYPE_WIDTH.get(d_type, 8)
                is_indexable = True

            if not is_indexable:
                card = 0
            else:
                raw_card = int(agg_result.get(f"card_{idx}") or 1)
                if use_sample and denom_rows > 0:
                    sample_ratio = raw_card / float(denom_rows)
                    if sample_ratio > 0.85:
                        # High-uniqueness column: scale cardinality to full table
                        card = min(total_rows, int(round(sample_ratio * total_rows)))
                    elif sample_ratio > 0.15:
                        card = min(total_rows, int(round(raw_card * math.sqrt(total_rows / denom_rows))))
                    else:
                        card = raw_card
                else:
                    card = raw_card

            null_cnt = int(agg_result.get(f"null_{idx}") or 0)
            dist_ratio = min(1.0, card / max(1, total_rows))
            null_frac = min(1.0, null_cnt / max(1, denom_rows))
            min_val = agg_result.get(f"min_{idx}")
            max_val = agg_result.get(f"max_{idx}")

            # Fetch top frequent values & real sample values for workload generation and skew
            val_freqs: Dict[str, float] = {}
            sample_vals: List[Any] = []
            if is_indexable and total_rows > 0 and null_frac < 0.99:
                try:
                    freq_rows = self.db.execute(
                        f"SELECT `{c_name}` AS val, COUNT(*) AS cnt "
                        f"FROM {source_sql} "
                        f"WHERE `{c_name}` IS NOT NULL "
                        f"GROUP BY `{c_name}` "
                        f"ORDER BY cnt DESC LIMIT 30"
                    )
                    for fr in freq_rows:
                        v = fr.get("val")
                        if v is not None:
                            sample_vals.append(v)
                            val_freqs[str(v).lower()] = min(
                                1.0, float(fr["cnt"]) / max(1, denom_rows)
                            )
                except Exception:
                    pass

            columns_dict[c_name] = ColumnStats(
                column_name=c_name,
                data_type=d_type,
                data_type_code=DATA_TYPE_CODE.get(d_type, 0),
                byte_width=width,
                is_nullable=is_null,
                is_primary_key=is_pk,
                is_indexable=is_indexable,
                cardinality=card,
                distinct_ratio=dist_ratio,
                null_fraction=null_frac,
                min_value=min_val,
                max_value=max_val,
                value_frequencies=val_freqs,
                sample_values=sample_vals,
            )

        t_stats = TableStats(
            database_name=self.db.database,
            table_name=actual_table,
            table_rows=total_rows,
            avg_row_length=avg_row_len,
            data_length_mb=data_len_mb,
            primary_key_columns=pk_cols,
            columns=columns_dict,
            canonical_columns=canon_map,
        )
        self._tables[cache_key] = t_stats
        return t_stats

    def get_baseline_explain(
        self, sql: str, primary_table: str
    ) -> BaselineExplainInfo:
        key = f"{self.db.database}:{sql.strip()}"
        if key in self._explain_cache:
            return self._explain_cache[key]

        plan_rows = self.db.explain(sql)
        target_row: Optional[Dict[str, Any]] = None
        for r in plan_rows:
            tbl = str(r.get("table") or "").lower()
            if tbl == primary_table.lower():
                target_row = r
                break
        if target_row is None and plan_rows:
            target_row = plan_rows[0]

        if target_row is None:
            info = BaselineExplainInfo(
                access_type="ALL",
                access_type_code=0,
                estimated_rows=1,
                filtered_pct=100.0,
                uses_filesort=0,
                uses_temporary=0,
                raw_rows=[],
            )
        else:
            acc_type = str(target_row.get("type") or "ALL")
            acc_code = EXPLAIN_ACCESS_TYPE_CODE.get(acc_type.lower(), 0)
            est_rows = int(target_row.get("rows") or 1)
            filtered = float(target_row.get("filtered") or 100.0)
            extra = str(target_row.get("Extra") or "").lower()
            filesort = 1 if "using filesort" in extra else 0
            temporary = 1 if "using temporary" in extra else 0
            info = BaselineExplainInfo(
                access_type=acc_type,
                access_type_code=acc_code,
                estimated_rows=est_rows,
                filtered_pct=filtered,
                uses_filesort=filesort,
                uses_temporary=temporary,
                raw_rows=plan_rows,
            )

        self._explain_cache[key] = info
        return info


def _to_days(val: Any) -> Optional[float]:
    """Convert a date/datetime/timedelta or ISO string into numeric days/seconds."""
    if isinstance(val, datetime):
        return float(val.date().toordinal())
    if isinstance(val, date):
        return float(val.toordinal())
    if isinstance(val, timedelta):
        return float(val.total_seconds())
    if isinstance(val, str):
        try:
            return float(date.fromisoformat(val[:10]).toordinal())
        except ValueError:
            return None
    return None


def estimate_column_selectivity(
    col_stats: ColumnStats,
    predicates: Sequence[PredicateInfo],
    is_join_col: bool = False,
    has_table_where_preds: bool = False,
) -> float:
    """Estimate fraction of rows (0.0 to 1.0) matching predicates on *col_stats*."""
    col_preds = [
        p for p in predicates if p.column.lower() == col_stats.column_name.lower()
    ]
    if not col_preds:
        if is_join_col and not has_table_where_preds and col_stats.cardinality > 0:
            return 1.0 / col_stats.cardinality
        return 1.0

    selectivity = 1.0
    card = max(1, col_stats.cardinality)

    for pred in col_preds:
        p_type = pred.predicate_type
        op = pred.operator
        lits = pred.literals

        if p_type == "equality":
            if lits and col_stats.value_frequencies:
                lit_key = str(lits[0]).lower()
                p_sel = col_stats.value_frequencies.get(lit_key, 1.0 / card)
            else:
                p_sel = 1.0 / card

        elif p_type == "in":
            if lits and col_stats.value_frequencies:
                p_sel = sum(
                    col_stats.value_frequencies.get(str(x).lower(), 1.0 / card)
                    for x in lits
                )
            else:
                n_vals = max(1, len(lits))
                p_sel = min(1.0, n_vals / card)

        elif p_type == "like_prefix":
            if lits and col_stats.value_frequencies:
                prefix = str(lits[0]).lower().rstrip("%_")
                matched = sum(
                    freq
                    for k, freq in col_stats.value_frequencies.items()
                    if k.startswith(prefix)
                )
                p_sel = matched if matched > 0 else min(1.0, 2.0 / card)
            else:
                p_sel = min(1.0, 5.0 / card)

        elif p_type == "range":
            p_sel = _estimate_range_selectivity(col_stats, op, lits)

        else:
            p_sel = 0.5

        selectivity *= max(1e-7, min(1.0, p_sel))

    return max(1e-7, min(1.0, selectivity))


def _estimate_range_selectivity(
    col_stats: ColumnStats, op: str, lits: Sequence[Any]
) -> float:
    """Interpolate range selectivity using column min/max bounds."""
    min_v = col_stats.min_value
    max_v = col_stats.max_value

    if col_stats.data_type_code in (1, 2) and min_v is not None and max_v is not None:
        try:
            lo_bound = float(min_v)
            hi_bound = float(max_v)
            span = max(1e-6, hi_bound - lo_bound)
            if op == "BETWEEN" and len(lits) >= 2:
                v1, v2 = float(lits[0]), float(lits[1])
                return max(1e-5, min(1.0, abs(v2 - v1) / span))
            if lits:
                v = float(lits[0])
                if op in (">", ">="):
                    return max(1e-5, min(1.0, (hi_bound - v) / span))
                if op in ("<", "<="):
                    return max(1e-5, min(1.0, (v - lo_bound) / span))
        except (TypeError, ValueError):
            pass

    if col_stats.data_type_code == 3 and min_v is not None and max_v is not None:
        d_min = _to_days(min_v)
        d_max = _to_days(max_v)
        if d_min is not None and d_max is not None:
            span = max(1.0, d_max - d_min)
            if op == "BETWEEN" and len(lits) >= 2:
                d1, d2 = _to_days(lits[0]), _to_days(lits[1])
                if d1 is not None and d2 is not None:
                    return max(1e-4, min(1.0, abs(d2 - d1) / span))
            if lits:
                d_val = _to_days(lits[0])
                if d_val is not None:
                    if op in (">", ">="):
                        return max(1e-4, min(1.0, (d_max - d_val) / span))
                    if op in ("<", "<="):
                        return max(1e-4, min(1.0, (d_val - d_min) / span))

    return 0.15


def estimate_index_size_mb(table_rows: int, col_widths: Sequence[int]) -> float:
    """Estimate InnoDB secondary B-tree index size in MB."""
    key_bytes = sum(col_widths) + 14
    raw_bytes = table_rows * key_bytes * 1.25
    return round(raw_bytes / (1024.0 * 1024.0), 3)


FEATURE_NAMES: List[str] = [
    "table_rows",
    "log_table_rows",
    "num_index_cols",
    "is_composite",
    "estimated_index_size_mb",
    # Position 1 (leading column)
    "col1_cardinality",
    "col1_log_cardinality",
    "col1_distinct_ratio",
    "col1_null_fraction",
    "col1_byte_width",
    "col1_data_type_code",
    "col1_in_where",
    "col1_in_join",
    "col1_in_group_by",
    "col1_in_order_by",
    "col1_pred_equality",
    "col1_pred_range",
    "col1_pred_in",
    "col1_pred_like_prefix",
    "col1_selectivity",
    "col1_log_selectivity",
    # Position 2 (second column in composite index, 0 if single-column)
    "col2_cardinality",
    "col2_log_cardinality",
    "col2_distinct_ratio",
    "col2_null_fraction",
    "col2_byte_width",
    "col2_data_type_code",
    "col2_in_where",
    "col2_in_join",
    "col2_in_group_by",
    "col2_in_order_by",
    "col2_pred_equality",
    "col2_pred_range",
    "col2_pred_in",
    "col2_pred_like_prefix",
    "col2_selectivity",
    "col2_log_selectivity",
    # Query & B-tree interaction features
    "has_join",
    "has_group_by",
    "has_order_by",
    "has_limit",
    "limit_ratio",
    "num_where_cols",
    "where_coverage_ratio",
    "effective_index_selectivity",
    "log_effective_selectivity",
    "estimated_index_rows_scanned",
    "leading_col_is_equality",
    "leading_eq_second_range_or_order",
    "order_by_covered_with_limit",
    "group_by_covered",
    # Baseline EXPLAIN plan features
    "baseline_access_type_code",
    "baseline_explain_rows",
    "baseline_explain_rows_ratio",
    "baseline_explain_filtered",
    "baseline_uses_filesort",
    "baseline_uses_temporary",
]


def _col_in_list(col_name: str, col_list: Sequence[str]) -> bool:
    c_low = col_name.lower()
    return any(x.lower() == c_low for x in col_list)


def _get_pred_type(col_name: str, pred_types: Dict[str, str]) -> str:
    if col_name in pred_types:
        return pred_types[col_name]
    c_low = col_name.lower()
    for k, v in pred_types.items():
        if k.lower() == c_low:
            return v
    return ""


def _extract_pos_features(
    col_name: Optional[str],
    parsed: ParsedQuery,
    table_stats: TableStats,
) -> Dict[str, float]:
    """Extract schema-agnostic numeric features for a single index position."""
    cs = table_stats.get_col(col_name) if col_name else None
    if cs is None:
        return {
            "cardinality": 0.0,
            "log_cardinality": 0.0,
            "distinct_ratio": 0.0,
            "null_fraction": 0.0,
            "byte_width": 0.0,
            "data_type_code": 0.0,
            "in_where": 0.0,
            "in_join": 0.0,
            "in_group_by": 0.0,
            "in_order_by": 0.0,
            "pred_equality": 0.0,
            "pred_range": 0.0,
            "pred_in": 0.0,
            "pred_like_prefix": 0.0,
            "selectivity": 1.0,
            "log_selectivity": 0.0,
        }

    in_where = 1.0 if _col_in_list(cs.column_name, parsed.where_columns) else 0.0
    in_join = 1.0 if _col_in_list(cs.column_name, parsed.join_columns) else 0.0
    in_group = 1.0 if _col_in_list(cs.column_name, parsed.group_by_columns) else 0.0
    in_order = 1.0 if _col_in_list(cs.column_name, parsed.order_by_columns) else 0.0
    p_type = _get_pred_type(cs.column_name, parsed.predicate_types)

    sel = estimate_column_selectivity(
        cs,
        parsed.predicates,
        is_join_col=(in_join > 0),
        has_table_where_preds=(len(parsed.where_columns) > 0),
    )
    return {
        "cardinality": float(cs.cardinality),
        "log_cardinality": math.log1p(cs.cardinality),
        "distinct_ratio": float(cs.distinct_ratio),
        "null_fraction": float(cs.null_fraction),
        "byte_width": float(cs.byte_width),
        "data_type_code": float(cs.data_type_code),
        "in_where": in_where,
        "in_join": in_join,
        "in_group_by": in_group,
        "in_order_by": in_order,
        "pred_equality": 1.0 if p_type == "equality" else 0.0,
        "pred_range": 1.0 if p_type == "range" else 0.0,
        "pred_in": 1.0 if p_type == "in" else 0.0,
        "pred_like_prefix": 1.0 if p_type == "like_prefix" else 0.0,
        "selectivity": float(sel),
        "log_selectivity": math.log10(max(1e-7, sel)),
    }


def extract_features(
    query_or_parsed: str | ParsedQuery,
    candidate: CandidateIndex,
    stats_cache: StatsCache,
    baseline_explain: Optional[BaselineExplainInfo] = None,
) -> Dict[str, float]:
    """Compute the full schema-agnostic feature dictionary for a (query, candidate) pair."""
    t_stats = stats_cache.get_table_stats(candidate.table)
    parsed = (
        parse_query(
            query_or_parsed,
            target_table=t_stats.table_name,
            schema_columns=t_stats.canonical_columns,
        )
        if isinstance(query_or_parsed, str)
        else query_or_parsed
    )
    exp_info = baseline_explain or stats_cache.get_baseline_explain(
        parsed.raw_sql, candidate.table
    )

    col1 = candidate.columns[0] if len(candidate.columns) >= 1 else None
    col2 = candidate.columns[1] if len(candidate.columns) >= 2 else None

    f1 = _extract_pos_features(col1, parsed, t_stats)
    f2 = _extract_pos_features(col2, parsed, t_stats)

    col_widths = []
    for c in candidate.columns:
        cs = t_stats.get_col(c)
        if cs is not None:
            col_widths.append(cs.byte_width)
    idx_size_mb = estimate_index_size_mb(t_stats.table_rows, col_widths)

    col1_filters_table = f1["in_where"] > 0 or (
        f1["in_join"] > 0 and len(parsed.where_columns) == 0
    )
    if col1_filters_table:
        eff_sel = f1["selectivity"]
        if (
            col2 is not None
            and (f1["pred_equality"] > 0 or f1["pred_in"] > 0 or f1["in_join"] > 0)
            and f2["in_where"] > 0
        ):
            eff_sel *= f2["selectivity"]
    elif (
        f1["in_order_by"] > 0
        and parsed.has_limit
        and parsed.limit_value
        and len(parsed.where_columns) == 0
    ):
        eff_sel = min(1.0, float(parsed.limit_value) / max(1, t_stats.table_rows))
    else:
        eff_sel = 1.0

    eff_sel = max(1e-7, min(1.0, eff_sel))
    est_index_rows = eff_sel * t_stats.table_rows

    num_where = len(parsed.where_columns)
    covered_where = sum(
        1 for c in candidate.columns if _col_in_list(c, parsed.where_columns)
    )
    where_cov = covered_where / max(1, num_where) if num_where > 0 else 0.0

    leading_eq = 1.0 if (f1["pred_equality"] > 0 or f1["in_join"] > 0) else 0.0
    leading_eq_second_ro = (
        1.0
        if (
            leading_eq > 0
            and col2 is not None
            and (f2["pred_range"] > 0 or f2["in_order_by"] > 0 or f2["in_group_by"] > 0)
        )
        else 0.0
    )

    order_covered_limit = 0.0
    if parsed.has_order_by and parsed.has_limit and parsed.order_by_columns:
        first_order_col = parsed.order_by_columns[0].lower()
        if col1 and col1.lower() == first_order_col and num_where == 0:
            order_covered_limit = 1.0
        elif leading_eq > 0 and col2 and col2.lower() == first_order_col:
            order_covered_limit = 1.0

    group_covered = 0.0
    if parsed.has_group_by and parsed.group_by_columns:
        first_group_col = parsed.group_by_columns[0].lower()
        if (col1 and col1.lower() == first_group_col) or (
            leading_eq > 0 and col2 and col2.lower() == first_group_col
        ):
            group_covered = 1.0

    limit_ratio = (
        min(1.0, float(parsed.limit_value) / max(1, t_stats.table_rows))
        if (parsed.has_limit and parsed.limit_value is not None)
        else 1.0
    )

    return {
        "table_rows": float(t_stats.table_rows),
        "log_table_rows": math.log1p(t_stats.table_rows),
        "num_index_cols": float(len(candidate.columns)),
        "is_composite": 1.0 if candidate.is_composite else 0.0,
        "estimated_index_size_mb": float(idx_size_mb),
        # col1
        "col1_cardinality": f1["cardinality"],
        "col1_log_cardinality": f1["log_cardinality"],
        "col1_distinct_ratio": f1["distinct_ratio"],
        "col1_null_fraction": f1["null_fraction"],
        "col1_byte_width": f1["byte_width"],
        "col1_data_type_code": f1["data_type_code"],
        "col1_in_where": f1["in_where"],
        "col1_in_join": f1["in_join"],
        "col1_in_group_by": f1["in_group_by"],
        "col1_in_order_by": f1["in_order_by"],
        "col1_pred_equality": f1["pred_equality"],
        "col1_pred_range": f1["pred_range"],
        "col1_pred_in": f1["pred_in"],
        "col1_pred_like_prefix": f1["pred_like_prefix"],
        "col1_selectivity": f1["selectivity"],
        "col1_log_selectivity": f1["log_selectivity"],
        # col2
        "col2_cardinality": f2["cardinality"],
        "col2_log_cardinality": f2["log_cardinality"],
        "col2_distinct_ratio": f2["distinct_ratio"],
        "col2_null_fraction": f2["null_fraction"],
        "col2_byte_width": f2["byte_width"],
        "col2_data_type_code": f2["data_type_code"],
        "col2_in_where": f2["in_where"],
        "col2_in_join": f2["in_join"],
        "col2_in_group_by": f2["in_group_by"],
        "col2_in_order_by": f2["in_order_by"],
        "col2_pred_equality": f2["pred_equality"],
        "col2_pred_range": f2["pred_range"],
        "col2_pred_in": f2["pred_in"],
        "col2_pred_like_prefix": f2["pred_like_prefix"],
        "col2_selectivity": f2["selectivity"],
        "col2_log_selectivity": f2["log_selectivity"],
        # Interaction & query structure
        "has_join": 1.0 if parsed.has_join else 0.0,
        "has_group_by": 1.0 if parsed.has_group_by else 0.0,
        "has_order_by": 1.0 if parsed.has_order_by else 0.0,
        "has_limit": 1.0 if parsed.has_limit else 0.0,
        "limit_ratio": float(limit_ratio),
        "num_where_cols": float(num_where),
        "where_coverage_ratio": float(where_cov),
        "effective_index_selectivity": float(eff_sel),
        "log_effective_selectivity": math.log10(eff_sel),
        "estimated_index_rows_scanned": float(est_index_rows),
        "leading_col_is_equality": leading_eq,
        "leading_eq_second_range_or_order": leading_eq_second_ro,
        "order_by_covered_with_limit": order_covered_limit,
        "group_by_covered": group_covered,
        # Baseline EXPLAIN
        "baseline_access_type_code": float(exp_info.access_type_code),
        "baseline_explain_rows": float(exp_info.estimated_rows),
        "baseline_explain_rows_ratio": float(exp_info.estimated_rows)
        / max(1.0, float(t_stats.table_rows)),
        "baseline_explain_filtered": float(exp_info.filtered_pct),
        "baseline_uses_filesort": float(exp_info.uses_filesort),
        "baseline_uses_temporary": float(exp_info.uses_temporary),
    }
