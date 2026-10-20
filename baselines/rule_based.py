"""
baselines/rule_based.py
───────────────────────
Preserves the legacy rule-based scorer and regex SQL parser from the original
IndexBuddy repository (formerly in ``src/sql_parser.py`` and ``src/model_trainer.py``),
alongside a ``NoIndexBaseline``, solely for academic baseline comparison against
the trained XGBoost speedup regressor on held-out query templates.

IMPORTANT: This module is NEVER used to label training data or make live
dashboard recommendations. All live recommendations come exclusively from the
XGBoost model in ``models/predict.py`` and ``selection/recommend.py``.
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, Optional

import config


def legacy_regex_parse_sql(sql_string: str) -> Dict[str, Any]:
    """Original regex-based SQL parser from legacy ``src/sql_parser.py``."""
    if not sql_string or not sql_string.strip():
        return {
            "table_name": "unknown",
            "candidate_column": "unknown",
            "has_join": 0,
            "is_range_scan": 0,
        }

    sql_upper = sql_string.upper()
    has_join = 1 if "JOIN" in sql_upper else 0
    range_operators = [">", "<", ">=", "<=", "BETWEEN"]
    is_range_scan = 1 if any(op in sql_upper for op in range_operators) else 0

    table_match = re.search(r"(?:FROM|JOIN)\s+([a-zA-Z0-9_]+)", sql_upper)
    table_name = table_match.group(1).lower() if table_match else "unknown"

    col_match = re.search(
        r"(?:WHERE|AND|OR|ON)\s+(?:[a-zA-Z0-9_]+\.)?([a-zA-Z0-9_]+)\s*"
        r"(?:=|>|<|>=|<=|!=|LIKE|IN|BETWEEN)",
        sql_upper,
    )
    candidate_column = col_match.group(1).lower() if col_match else "unknown"

    return {
        "table_name": table_name,
        "candidate_column": candidate_column,
        "has_join": has_join,
        "is_range_scan": is_range_scan,
    }


class NoIndexBaseline:
    """Baseline that never recommends creating any secondary index (speedup = 1.0x)."""

    name: str = "No-Index Baseline"

    @staticmethod
    def predict_speedup(_feature_row: Dict[str, Any]) -> float:
        return 1.0

    @staticmethod
    def predict_log_speedup(_feature_row: Dict[str, Any]) -> float:
        return 0.0

    @staticmethod
    def recommend_for_query(_candidates_df: Any) -> Optional[str]:
        return None


class LegacyRuleBasedScorer:
    """Original hand-written heuristic scorer from legacy IndexBuddy.

    In the legacy system (``src/model_trainer.py`` lines 81 & 97):
      - An index was flagged beneficial iff ``selectivity < 0.05`` (or ``< 0.10``)
        OR ``has_join == 1``.
      - Single-column indexes were favored over composite indexes (+0.2 heuristic
        bonus), and column ordering within a composite index was ignored.
    """

    name: str = "Legacy Rule-Based Scorer"

    @staticmethod
    def score_candidate_row(row: Dict[str, Any]) -> float:
        """Return a heuristic pseudo-speedup score for a candidate row."""
        col1_sel = float(row.get("col1_selectivity", 1.0))
        has_join = int(float(row.get("has_join", 0))) == 1
        is_composite = int(float(row.get("is_composite", 0))) == 1
        in_where = int(float(row.get("col1_in_where", 0))) == 1
        in_group = int(float(row.get("col1_in_group_by", 0))) == 1
        in_order = int(float(row.get("col1_in_order_by", 0))) == 1

        # Legacy rule: beneficial if selectivity < 0.10 or has_join
        is_beneficial = (col1_sel < 0.10 and in_where) or has_join
        if not is_beneficial:
            # Legacy scorer gave a flat base confidence < threshold
            return 1.0

        # Heuristic confidence formula from legacy utils/recommender:
        # starts at base score, favors single-column indexes (+0.2),
        # adds small bonuses for join/group/order clauses, ignores col2 position
        score = 2.0
        if not is_composite:
            score += 1.5  # Legacy heuristic preferred single-column indexes
        if has_join:
            score += 1.0
        if in_group:
            score += 0.5
        if in_order:
            score += 0.5
        if col1_sel < 0.01:
            score += 2.0
        return score

    @classmethod
    def predict_speedup(cls, row: Dict[str, Any]) -> float:
        return cls.score_candidate_row(row)

    @classmethod
    def predict_log_speedup(cls, row: Dict[str, Any]) -> float:
        return math.log(max(cls.score_candidate_row(row), 1e-3))

    @classmethod
    def recommend_for_query(
        cls,
        candidates_df: Any,
        threshold: float = config.SPEEDUP_THRESHOLD,
    ) -> Optional[str]:
        """Select the highest-scoring candidate index name, or None if < threshold."""
        best_idx: Optional[str] = None
        best_score: float = -1.0
        for _, row in candidates_df.iterrows():
            s = cls.score_candidate_row(row.to_dict())
            if s > best_score:
                best_score = s
                best_idx = str(row["index_name"])
        if best_score >= threshold:
            return best_idx
        return None
