"""
selection/recommend.py
──────────────────────
ML-driven index selection engine and dynamic sample query generator.
Zero hardcoded table or column names.

All candidate scoring and ranking is performed exclusively by the trained
XGBoost speedup regression model (``models/predict.py``). A candidate is
recommended when its ML-predicted speedup is at least ``threshold`` (default
1.5x from ``config.py``); otherwise no index is recommended.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import config
from candidates.generator import CandidateIndex, generate_candidates
from features.extract import ColumnStats, StatsCache, TableStats, extract_features
from models.predict import SpeedupPredictor
from parsing.query_parser import ParsedQuery, parse_query


@dataclass
class ScoredCandidate:
    """A candidate index together with its ML-predicted speedup and features."""

    candidate: CandidateIndex
    predicted_speedup: float
    predicted_log_speedup: float
    estimated_selectivity: float
    estimated_index_size_mb: float
    features: Dict[str, float] = field(default_factory=dict)
    shap_top_factors: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class RecommendationResult:
    """Result of evaluating all candidate indexes for a SQL query."""

    parsed_query: ParsedQuery
    ranked_candidates: List[ScoredCandidate]
    recommended: Optional[ScoredCandidate]
    threshold: float
    reason: str


def _sql_literal(val: Any) -> str:
    """Format a Python value as a safe MySQL SQL literal."""
    if val is None:
        return "NULL"
    if isinstance(val, (int, float)):
        return str(val)
    text = str(val).replace("'", "''")
    return f"'{text}'"


def generate_sample_queries_for_table(
    t_stats: TableStats,
    foreign_keys: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    """Dynamically build sample SELECT queries from any discovered table's real columns & values.

    Includes:
      1. High-selectivity equality lookup (+ range/ORDER BY if available)
      2. Multi-column equality + range / ORDER BY ... LIMIT
      3. Low-selectivity skewed filter (where the ML model should recommend NO index)
      4. Aggregation (GROUP BY) query
      5. Foreign-key JOIN query (if foreign keys exist on the database)
    """
    tbl = t_stats.table_name
    pk_set = {c.lower() for c in t_stats.primary_key_columns}
    non_pk_cols = [
        cs
        for cs in t_stats.columns.values()
        if cs.is_indexable and cs.column_name.lower() not in pk_set and cs.cardinality > 0
    ]
    if not non_pk_cols:
        non_pk_cols = [cs for cs in t_stats.columns.values() if cs.is_indexable]

    # Sort non-PK columns by cardinality descending
    by_card_desc = sorted(non_pk_cols, key=lambda c: c.cardinality, reverse=True)
    by_card_asc = [c for c in sorted(non_pk_cols, key=lambda c: c.cardinality) if c.cardinality >= 2]

    high_col: Optional[ColumnStats] = by_card_desc[0] if by_card_desc else None
    second_col: Optional[ColumnStats] = by_card_desc[1] if len(by_card_desc) > 1 else high_col
    low_col: Optional[ColumnStats] = by_card_asc[0] if by_card_asc else high_col

    # Pick a range/orderable column (numeric or date with >10 distinct values)
    range_cols = [
        c
        for c in by_card_desc
        if c.data_type_code in (1, 2, 3)
        and c.min_value is not None
        and c.max_value is not None
        and c.cardinality >= 5
    ]
    range_col: Optional[ColumnStats] = (
        next((c for c in range_cols if high_col and c.column_name != high_col.column_name), None)
        or (range_cols[0] if range_cols else second_col)
    )

    select_cols = [c.column_name for c in list(t_stats.columns.values())[:5]]
    proj = ", ".join(f"`{c}`" for c in select_cols) if select_cols else "*"

    samples: List[Dict[str, str]] = []

    # 1. High-selectivity point lookup + range/sort (Strong index candidate)
    if high_col and high_col.sample_values:
        # Pick a less common sample value (tail of top-30) for high selectivity
        val_hi = high_col.sample_values[-1]
        if range_col and range_col.column_name != high_col.column_name and range_col.sample_values:
            val_rng = range_col.sample_values[len(range_col.sample_values) // 2]
            q1 = (
                f"SELECT {proj} FROM `{tbl}` "
                f"WHERE `{high_col.column_name}` = {_sql_literal(val_hi)} "
                f"AND `{range_col.column_name}` >= {_sql_literal(val_rng)} "
                f"ORDER BY `{range_col.column_name}` DESC LIMIT 25"
            )
            samples.append(
                {
                    "title": f"1. High-Selectivity Composite Filter + Sort (`{high_col.column_name}`, `{range_col.column_name}`)",
                    "category": "high_selectivity_composite",
                    "description": (
                        f"Equality filter on high-cardinality `{high_col.column_name}` "
                        f"({high_col.cardinality:,} distinct values) combined with range & ORDER BY on `{range_col.column_name}`."
                    ),
                    "sql": q1,
                }
            )
        else:
            q1 = (
                f"SELECT {proj} FROM `{tbl}` "
                f"WHERE `{high_col.column_name}` = {_sql_literal(val_hi)} LIMIT 25"
            )
            samples.append(
                {
                    "title": f"1. High-Selectivity Point Lookup (`{high_col.column_name}`)",
                    "category": "high_selectivity_point",
                    "description": f"Selective equality lookup on `{high_col.column_name}` ({high_col.cardinality:,} distinct).",
                    "sql": q1,
                }
            )

    # 2. Single-column selective lookup or IN list
    if high_col and len(high_col.sample_values) >= 3:
        in_vals = ", ".join(_sql_literal(v) for v in high_col.sample_values[-3:])
        q2 = (
            f"SELECT {proj} FROM `{tbl}` "
            f"WHERE `{high_col.column_name}` IN ({in_vals})"
        )
        samples.append(
            {
                "title": f"2. Selective IN-List Lookup (`{high_col.column_name}`)",
                "category": "in_list",
                "description": f"IN-list predicate on `{high_col.column_name}` matching 3 specific values.",
                "sql": q2,
            }
        )

    # 3. Low-Selectivity / High-Frequency Query (where model should recommend NO INDEX)
    # Find the column+value with highest frequency (skew) or broad range
    best_skew_col: Optional[ColumnStats] = None
    best_skew_val: Any = None
    best_skew_freq: float = 0.0
    for cs in non_pk_cols:
        if cs.sample_values and cs.value_frequencies:
            top_v = cs.sample_values[0]
            freq = cs.value_frequencies.get(str(top_v).lower(), 0.0)
            if freq > best_skew_freq:
                best_skew_freq = freq
                best_skew_col = cs
                best_skew_val = top_v

    if best_skew_col is not None and best_skew_freq >= 0.15:
        q3 = (
            f"SELECT `{best_skew_col.column_name}`, COUNT(*) AS row_cnt "
            f"FROM `{tbl}` "
            f"WHERE `{best_skew_col.column_name}` = {_sql_literal(best_skew_val)} "
            f"GROUP BY `{best_skew_col.column_name}`"
        )
        samples.append(
            {
                "title": (
                    f"3. Low-Selectivity Skewed Filter (`{best_skew_col.column_name}` — "
                    f"Matches ~{best_skew_freq*100:.1f}% of Table → Should Recommend NO INDEX)"
                ),
                "category": "low_selectivity_no_index",
                "description": (
                    f"Filters on the most frequent value of `{best_skew_col.column_name}` "
                    f"(~{best_skew_freq*100:.1f}% of rows). A secondary index is unselective here."
                ),
                "sql": q3,
            }
        )
    elif range_col and range_col.min_value is not None:
        q3 = (
            f"SELECT COUNT(*) AS total_matched "
            f"FROM `{tbl}` "
            f"WHERE `{range_col.column_name}` >= {_sql_literal(range_col.min_value)}"
        )
        samples.append(
            {
                "title": (
                    f"3. Unselective Broad Range Scan (`{range_col.column_name}` — "
                    f"Matches ~100% of Table → Should Recommend NO INDEX)"
                ),
                "category": "low_selectivity_no_index",
                "description": (
                    f"Broad range predicate on `{range_col.column_name}` starting at the minimum value, "
                    "scanning nearly the entire table."
                ),
                "sql": q3,
            }
        )

    # 4. Multi-column equality + GROUP BY
    if high_col and second_col and high_col.column_name != second_col.column_name and high_col.sample_values:
        v1 = high_col.sample_values[-1]
        q4 = (
            f"SELECT `{second_col.column_name}`, COUNT(*) AS cnt "
            f"FROM `{tbl}` "
            f"WHERE `{high_col.column_name}` = {_sql_literal(v1)} "
            f"GROUP BY `{second_col.column_name}`"
        )
        samples.append(
            {
                "title": f"4. Equality Filter + GROUP BY (`{high_col.column_name}`, `{second_col.column_name}`)",
                "category": "equality_group_by",
                "description": f"Filters on `{high_col.column_name}` and groups by `{second_col.column_name}`.",
                "sql": q4,
            }
        )

    # 5. Foreign-key JOIN query if foreign keys involve this table
    if foreign_keys:
        for fk in foreign_keys:
            if fk["table_name"].lower() == tbl.lower() and high_col and high_col.sample_values:
                ref_tbl = fk["ref_table_name"]
                fk_col = fk["column_name"]
                ref_col = fk["ref_column_name"]
                v_hi = high_col.sample_values[-1]
                q5 = (
                    f"SELECT t1.`{fk_col}`, t1.`{high_col.column_name}` "
                    f"FROM `{tbl}` t1 "
                    f"JOIN `{ref_tbl}` t2 ON t1.`{fk_col}` = t2.`{ref_col}` "
                    f"WHERE t1.`{high_col.column_name}` = {_sql_literal(v_hi)} LIMIT 25"
                )
                samples.append(
                    {
                        "title": f"5. Foreign-Key JOIN (`{tbl}` ⋈ `{ref_tbl}`)",
                        "category": "join_query",
                        "description": f"Joins `{tbl}` with `{ref_tbl}` on `{fk_col} = {ref_col}` with selective filter.",
                        "sql": q5,
                    }
                )
                break

    return samples


def recommend_index(
    sql: str,
    stats_cache: StatsCache,
    target_table: Optional[str] = None,
    predictor: Optional[SpeedupPredictor] = None,
    threshold: float = config.SPEEDUP_THRESHOLD,
    max_candidates: int = config.MAX_CANDIDATES_PER_QUERY,
) -> RecommendationResult:
    """Parse *sql*, generate candidates, score them with XGBoost, and select the best."""
    pred = predictor or SpeedupPredictor()

    # First pass parse to identify primary table if not explicitly provided
    initial_parsed = parse_query(sql, target_table=target_table)
    t_stats = stats_cache.get_table_stats(initial_parsed.primary_table)

    # Re-parse with exact schema column mapping for accurate multi-table disambiguation
    parsed = parse_query(
        sql,
        target_table=t_stats.table_name,
        schema_columns=t_stats.canonical_columns,
    )

    candidates = generate_candidates(
        parsed,
        max_candidates=max_candidates,
        excluded_columns=t_stats.excluded_columns(),
        single_pk_column=t_stats.single_pk_column(),
    )

    if not candidates:
        return RecommendationResult(
            parsed_query=parsed,
            ranked_candidates=[],
            recommended=None,
            threshold=threshold,
            reason=(
                "No indexable non-PK B-tree candidate columns found in WHERE, JOIN, "
                "GROUP BY, or ORDER BY clauses."
            ),
        )

    baseline_exp = stats_cache.get_baseline_explain(parsed.raw_sql, parsed.primary_table)
    feature_list = [
        extract_features(parsed, cand, stats_cache, baseline_explain=baseline_exp)
        for cand in candidates
    ]
    preds = pred.predict_batch(feature_list)

    scored: List[ScoredCandidate] = []
    for cand, feats, (spd, log_spd) in zip(candidates, feature_list, preds):
        shap_factors = pred.explain_one(feats, top_k=5)
        scored.append(
            ScoredCandidate(
                candidate=cand,
                predicted_speedup=round(spd, 2),
                predicted_log_speedup=round(log_spd, 4),
                estimated_selectivity=float(feats.get("effective_index_selectivity", 1.0)),
                estimated_index_size_mb=float(feats.get("estimated_index_size_mb", 0.0)),
                features=feats,
                shap_top_factors=shap_factors,
            )
        )

    scored.sort(key=lambda s: s.predicted_log_speedup, reverse=True)
    best = scored[0]

    if best.predicted_speedup >= threshold:
        recommended = best
        reason = (
            f"Top candidate `{best.candidate.index_name}` on ({best.candidate.columns_display}) "
            f"has ML-predicted speedup {best.predicted_speedup:.2f}x >= threshold {threshold:.2f}x "
            f"(estimated effective selectivity {best.estimated_selectivity*100:.4f}%, "
            f"estimated index size ~{best.estimated_index_size_mb:.2f} MB)."
        )
    else:
        recommended = None
        reason = (
            f"No index recommended: highest ML-predicted speedup is {best.predicted_speedup:.2f}x "
            f"on ({best.candidate.columns_display}), which is below the {threshold:.2f}x threshold "
            f"(estimated effective selectivity {best.estimated_selectivity*100:.2f}% scans too many "
            f"rows to justify secondary B-tree maintenance and random bookmark lookups)."
        )

    return RecommendationResult(
        parsed_query=parsed,
        ranked_candidates=scored,
        recommended=recommended,
        threshold=threshold,
        reason=reason,
    )


def explain_prediction_vs_actual(
    predicted_speedup: float,
    actual_speedup: float,
    rows_before: int,
    rows_after: int,
    time_before_s: float,
    time_after_s: float,
    access_type_after: str,
    key_used_after: str,
    estimated_selectivity: float,
) -> str:
    """Generate a plain-English diagnostic analysis comparing predicted vs actual speedup."""
    err_pct = (
        abs(predicted_speedup - actual_speedup) / max(actual_speedup, 1e-6) * 100.0
    )
    parts: List[str] = []

    parts.append(
        f"The XGBoost model predicted a **{predicted_speedup:.2f}x** speedup, and live "
        f"MySQL 8 execution achieved a **{actual_speedup:.2f}x** speedup "
        f"({time_before_s*1000:.2f} ms → {time_after_s*1000:.2f} ms, relative difference {err_pct:.1f}%)."
    )

    if key_used_after and key_used_after.upper() != "NONE":
        row_red_pct = (1.0 - rows_after / max(1, rows_before)) * 100.0
        parts.append(
            f"MySQL's optimizer switched the access path from a full scan (`{rows_before:,}` "
            f"estimated rows) to `{access_type_after}` using key `{key_used_after}`, "
            f"narrowing estimated rows examined to **{rows_after:,}** "
            f"({row_red_pct:.2f}% reduction in rows examined)."
        )
    else:
        parts.append(
            f"MySQL's cost-based optimizer chose access type `{access_type_after}` "
            f"without using a secondary index (`key=NULL`), examining `{rows_after:,}` "
            f"estimated rows because random B-tree bookmark lookups would cost more "
            f"than a sequential scan."
        )

    reasons: List[str] = []
    if rows_before > rows_after and actual_speedup < (rows_before / max(1, rows_after)):
        reasons.append(
            f"**InnoDB Buffer Pool Caching & Fixed Round-Trip Overhead**: Rows examined fell "
            f"from `{rows_before:,}` to `{rows_after:,}`, but wall-clock runtime fell by "
            f"`{actual_speedup:.2f}x` because hot pages were already cached in the InnoDB "
            f"buffer pool and Python-to-MySQL query parsing/network overhead (~0.3–1.0 ms) "
            f"puts a floor on minimum execution time."
        )
    elif time_after_s < 0.002:
        reasons.append(
            "**Sub-millisecond Fixed Overhead**: Once index lookups drop execution below 2 ms, "
            "fixed SQL parsing and TCP round-trip latency dominate total runtime."
        )

    actual_row_ratio = rows_after / max(1.0, float(rows_before))
    if abs(actual_row_ratio - estimated_selectivity) > 0.005:
        reasons.append(
            f"**Selectivity Estimation vs Actual Data Skew**: Catalog statistics estimated "
            f"an effective selectivity of `{estimated_selectivity*100:.4f}%`, whereas MySQL's "
            f"post-index `EXPLAIN` examined `{actual_row_ratio*100:.4f}%` (`{rows_after:,}` rows) "
            f"due to real multi-column correlation and skew."
        )
    else:
        reasons.append(
            f"**B-Tree Leaf Traversal & Clustered PK Lookups**: Catalog selectivity "
            f"(`{estimated_selectivity*100:.4f}%`) closely matched the B-tree scan (`{rows_after:,}` rows); "
            f"secondary-to-primary index bookmark lookups account for the remaining runtime."
        )

    parts.append("Why prediction and actual runtime differ:\n- " + "\n- ".join(reasons))
    return "\n\n".join(parts)
