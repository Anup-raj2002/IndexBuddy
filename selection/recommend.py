"""
selection/recommend.py
──────────────────────
ML-driven index selection engine.

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
from features.extract import StatsCache, extract_features
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


def recommend_index(
    sql: str,
    stats_cache: StatsCache,
    predictor: Optional[SpeedupPredictor] = None,
    threshold: float = config.SPEEDUP_THRESHOLD,
    max_candidates: int = config.MAX_CANDIDATES_PER_QUERY,
) -> RecommendationResult:
    """Parse *sql*, generate candidates, score them with XGBoost, and select the best."""
    pred = predictor or SpeedupPredictor()
    parsed = parse_query(sql)
    candidates = generate_candidates(parsed, max_candidates=max_candidates)

    if not candidates:
        return RecommendationResult(
            parsed_query=parsed,
            ranked_candidates=[],
            recommended=None,
            threshold=threshold,
            reason=(
                "No indexable B-tree candidate columns found in WHERE, JOIN, "
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

    # Sort strictly by ML-predicted speedup descending
    scored.sort(key=lambda s: s.predicted_log_speedup, reverse=True)
    best = scored[0]

    if best.predicted_speedup >= threshold:
        recommended = best
        reason = (
            f"Top candidate `{best.candidate.index_name}` on ({best.candidate.columns_display}) "
            f"has ML-predicted speedup {best.predicted_speedup:.2f}x >= threshold {threshold:.2f}x "
            f"(estimated effective selectivity {best.estimated_selectivity*100:.4f}%, "
            f"estimated index size ~{best.estimated_index_size_mb:.1f} MB)."
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
        f"({time_before_s*1000:.2f} ms -> {time_after_s*1000:.2f} ms, relative difference {err_pct:.1f}%)."
    )

    if key_used_after:
        parts.append(
            f"MySQL's optimizer switched the access path from a full table scan (`ALL`, "
            f"{rows_before:,} estimated rows) to `{access_type_after}` using key "
            f"`{key_used_after}`, narrowing estimated rows examined to **{rows_after:,}** "
            f"({(1.0 - rows_after / max(1, rows_before))*100:.2f}% reduction in scanned rows)."
        )
    else:
        parts.append(
            f"Even after creating the index, MySQL's cost-based optimizer chose access "
            f"type `{access_type_after}` without using a secondary key (`key=NULL`), "
            f"scanning {rows_after:,} estimated rows because random B-tree bookmark lookups "
            f"would cost more than a sequential clustered scan."
        )

    # Explain why predicted and actual numbers differ
    reasons: List[str] = []
    if time_after_s < 0.002:
        reasons.append(
            "**Sub-millisecond Fixed Overhead & Buffer Pool Caching**: Once index lookups drop "
            "execution into the < 2 ms regime, fixed Python-to-MySQL TCP round-trip, SQL parser, "
            "and InnoDB buffer pool latch overhead (~0.25–0.50 ms) dominate total latency, creating "
            "natural variance in large speedup ratios."
        )
    actual_row_ratio = rows_after / max(1.0, float(rows_before))
    if abs(actual_row_ratio - estimated_selectivity) > 0.005:
        reasons.append(
            f"**Selectivity & Histogram Estimation Gap**: Catalog feature extraction estimated "
            f"an effective B-tree selectivity of {estimated_selectivity*100:.4f}%, whereas MySQL's "
            f"post-index `EXPLAIN` estimated {actual_row_ratio*100:.4f}% ({rows_after:,} rows) "
            f"based on live B-tree dive statistics."
        )
    else:
        reasons.append(
            f"**Selectivity & B-Tree Page Dives**: Catalog selectivity ({estimated_selectivity*100:.4f}%) "
            f"aligned closely with MySQL's B-tree leaf page estimate ({rows_after:,} rows), while "
            f"clustered PK bookmark lookups (`InnoDB` secondary-to-primary index hops) and OS scheduling "
            f"account for the remaining runtime difference."
        )

    parts.append("Why prediction and reality differ:\n- " + "\n- ".join(reasons))
    return "\n\n".join(parts)


if __name__ == "__main__":
    from db.mysql_adapter import MySQLAdapter

    db = MySQLAdapter()
    db.connect()
    sc = StatsCache(db)

    demo_queries = [
        (
            "High-Selectivity Composite Demo (Held-Out T11)",
            "SELECT order_id, customer_id, product_id, order_date, amount "
            "FROM orders_benchmark "
            "WHERE customer_id = 42310 AND order_date >= '2023-06-01' "
            "ORDER BY order_date DESC LIMIT 25",
        ),
        (
            "Low-Selectivity Skewed Demo (Held-Out T03 — should recommend NO index)",
            "SELECT status, COUNT(*) AS order_cnt, AVG(amount) AS avg_amt "
            "FROM orders_benchmark "
            "WHERE status = 'completed' GROUP BY status",
        ),
        (
            "Multi-Column Equality Demo (Held-Out T07)",
            "SELECT order_id, customer_id, product_id, region, amount "
            "FROM orders_benchmark "
            "WHERE product_id = 1250 AND region = 'North'",
        ),
    ]

    for title, q in demo_queries:
        res = recommend_index(q, sc)
        print(f"\n=== {title} ===")
        print(f"SQL   : {q}")
        print(f"Reason: {res.reason}")
        for rank, c in enumerate(res.ranked_candidates, 1):
            print(
                f"  {rank}. ({c.candidate.columns_display:<24}) -> "
                f"Predicted Speedup: {c.predicted_speedup:>8.2f}x "
                f"(log={c.predicted_log_speedup:>6.3f}, "
                f"eff_sel={c.estimated_selectivity*100:.4f}%, "
                f"size={c.estimated_index_size_mb:.1f}MB)"
            )
        if res.recommended:
            print(f"Chosen DDL: {res.recommended.candidate.ddl}")
        else:
            print("Chosen DDL: NONE (No Index Recommended)")

    db.close()
