"""selection package — ML-powered candidate ranking and index recommendation."""
from selection.recommend import (
    RecommendationResult,
    ScoredCandidate,
    explain_prediction_vs_actual,
    recommend_index,
)

__all__ = [
    "RecommendationResult",
    "ScoredCandidate",
    "explain_prediction_vs_actual",
    "recommend_index",
]
