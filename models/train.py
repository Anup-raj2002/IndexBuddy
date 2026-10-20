"""
models/train.py
───────────────
Trains an XGBoost regressor to predict ``log_speedup = log(time_before / time_after)``
from live MySQL catalog, query AST, candidate index position, and baseline EXPLAIN
features.

Key guarantees:
  - Splits train and test sets STRICTLY BY QUERY TEMPLATE (never by random row).
  - Holds out both test templates (T16, T19) and live demo templates (T03, T07, T11)
    from training, verifying zero template overlap.
  - Reports MAE and RMSE on log(speedup), Spearman rank correlation, and Top-1
    recommendation accuracy on held-out templates.
  - Produces a SHAP feature-importance summary plot at ``evaluation/shap_summary.png``.
  - Saves the trained model bundle to ``models/xgb_speedup.joblib`` via joblib.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

import config
from features.extract import FEATURE_NAMES


def compute_query_level_accuracy(
    df_eval: pd.DataFrame,
    pred_log_col: str,
    threshold: float = config.SPEEDUP_THRESHOLD,
) -> Dict[str, float]:
    """Compute query-level recommendation metrics across held-out queries.

    For each query:
      - Oracle action: if max actual speedup < threshold -> 'NO_INDEX',
        else the candidate with highest actual speedup.
      - Model action: if max predicted speedup < threshold -> 'NO_INDEX',
        else the candidate with highest predicted speedup.
    """
    log_thresh = math.log(threshold)
    exact_matches = 0
    effective_matches = 0
    decision_matches = 0
    achieved_speedups: List[float] = []
    oracle_speedups: List[float] = []

    for _, q_df in df_eval.groupby("query_id"):
        # Oracle best
        best_actual_row = q_df.loc[q_df["log_speedup"].idxmax()]
        oracle_log = float(best_actual_row["log_speedup"])
        oracle_speedup = float(best_actual_row["speedup"])
        oracle_recommends = oracle_log >= log_thresh
        oracle_action = (
            str(best_actual_row["index_name"]) if oracle_recommends else "NO_INDEX"
        )
        oracle_eff_speedup = oracle_speedup if oracle_recommends else 1.0
        oracle_speedups.append(oracle_eff_speedup)

        # Predictor choice
        best_pred_row = q_df.loc[q_df[pred_log_col].idxmax()]
        pred_log = float(best_pred_row[pred_log_col])
        pred_recommends = pred_log >= log_thresh
        pred_action = (
            str(best_pred_row["index_name"]) if pred_recommends else "NO_INDEX"
        )

        # Actual speedup achieved by following predictor's choice
        if pred_recommends:
            chosen_actual_speedup = float(best_pred_row["speedup"])
            chosen_actual_log = float(best_pred_row["log_speedup"])
        else:
            chosen_actual_speedup = 1.0
            chosen_actual_log = 0.0
        achieved_speedups.append(chosen_actual_speedup)

        if pred_recommends == oracle_recommends:
            decision_matches += 1

        if pred_action == oracle_action:
            exact_matches += 1
            effective_matches += 1
        elif (
            pred_recommends
            and oracle_recommends
            and chosen_actual_log >= log_thresh
            and chosen_actual_log >= 0.85 * oracle_log
        ):
            # Composite vs single-column tie on selective queries (both >1000x speedup)
            effective_matches += 1

    n_queries = max(1, df_eval["query_id"].nunique())
    geo_mean_achieved = math.exp(
        sum(math.log(max(s, 1e-3)) for s in achieved_speedups) / n_queries
    )
    geo_mean_oracle = math.exp(
        sum(math.log(max(s, 1e-3)) for s in oracle_speedups) / n_queries
    )

    return {
        "num_queries": float(n_queries),
        "top1_exact_accuracy": exact_matches / n_queries,
        "top1_effective_accuracy": effective_matches / n_queries,
        "index_decision_accuracy": decision_matches / n_queries,
        "geometric_mean_speedup": geo_mean_achieved,
        "oracle_geometric_mean_speedup": geo_mean_oracle,
        "mean_achieved_speedup": float(np.mean(achieved_speedups)),
    }


def generate_shap_plot(
    model: XGBRegressor,
    X_eval: pd.DataFrame,
    output_path: str = config.SHAP_PLOT_PATH,
) -> List[Tuple[str, float]]:
    """Compute SHAP values on held-out templates and save a beeswarm summary plot."""
    import shap

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_eval)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    ranked_indices = np.argsort(mean_abs_shap)[::-1]
    top_features: List[Tuple[str, float]] = [
        (FEATURE_NAMES[i], float(mean_abs_shap[i])) for i in ranked_indices[:15]
    ]

    plt.figure(figsize=(10, 6.5), dpi=160)
    shap.summary_plot(
        shap_values,
        X_eval,
        feature_names=FEATURE_NAMES,
        max_display=15,
        show=False,
    )
    plt.title(
        "SHAP Feature Importance Summary (Predicting log(Speedup) on Held-Out Templates)",
        fontsize=12,
        pad=14,
    )
    plt.tight_layout()
    plt.savefig(out_file, dpi=160, bbox_inches="tight")
    plt.close()
    return top_features


def train_model(
    csv_path: str = config.LABEL_CSV,
    model_out: str = config.MODEL_PATH,
    shap_out: str = config.SHAP_PLOT_PATH,
) -> Dict[str, Any]:
    """Train XGBoost speedup regressor with template-level splitting and save artifacts."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} labeled (query, candidate) pairs from {csv_path}")

    # Strict template-level split: train vs held-out (test + demo_holdout)
    train_df = df[df["split_group"] == "train"].copy()
    test_df = df[df["split_group"].isin(["test", "demo_holdout"])].copy()

    train_templates = set(train_df["template_id"].unique())
    test_templates = set(test_df["template_id"].unique())
    overlap_ids = train_templates.intersection(test_templates)
    overlap_sqls = set(train_df["template_sql"].unique()).intersection(
        set(test_df["template_sql"].unique())
    )
    assert not overlap_ids, f"Template ID leakage detected: {overlap_ids}"
    assert not overlap_sqls, f"Template SQL leakage detected: {overlap_sqls}"

    print(
        f"Template Split Verified (0 overlap):\n"
        f"  Train templates ({len(train_templates)}): {sorted(train_templates)} "
        f"-> {train_df['query_id'].nunique()} queries, {len(train_df)} pairs\n"
        f"  Held-out templates ({len(test_templates)}): {sorted(test_templates)} "
        f"-> {test_df['query_id'].nunique()} queries, {len(test_df)} pairs"
    )

    X_train = train_df[FEATURE_NAMES].astype(float)
    y_train = train_df["log_speedup"].astype(float)

    X_test = test_df[FEATURE_NAMES].astype(float)
    y_test = test_df["log_speedup"].astype(float)

    model = XGBRegressor(
        n_estimators=250,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=2,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=4,
    )
    model.fit(X_train, y_train)

    # Predictions
    train_pred_log = model.predict(X_train)
    test_pred_log = model.predict(X_test)
    train_df["pred_log_speedup"] = train_pred_log
    test_df["pred_log_speedup"] = test_pred_log
    test_df["pred_speedup"] = np.exp(test_pred_log)

    # Regression & ranking metrics
    train_mae = float(mean_absolute_error(y_train, train_pred_log))
    train_rmse = float(math.sqrt(mean_squared_error(y_train, train_pred_log)))
    test_mae = float(mean_absolute_error(y_test, test_pred_log))
    test_rmse = float(math.sqrt(mean_squared_error(y_test, test_pred_log)))

    spearman_res = spearmanr(y_test, test_pred_log)
    test_spearman = float(spearman_res.statistic)

    q_metrics = compute_query_level_accuracy(test_df, "pred_log_speedup")

    print("\n=== XGBoost Speedup Model Evaluation (Held-Out Query Templates) ===")
    print(f"  Train MAE (log speedup)           : {train_mae:.4f}")
    print(f"  Train RMSE (log speedup)          : {train_rmse:.4f}")
    print(f"  Held-Out MAE (log speedup)        : {test_mae:.4f}")
    print(f"  Held-Out RMSE (log speedup)       : {test_rmse:.4f}")
    print(f"  Held-Out Spearman Correlation (rho): {test_spearman:.4f}")
    print(
        f"  Top-1 Exact Recommendation Acc    : "
        f"{q_metrics['top1_exact_accuracy']*100:.1f}% "
        f"({int(round(q_metrics['top1_exact_accuracy']*q_metrics['num_queries']))}/{int(q_metrics['num_queries'])} queries)"
    )
    print(
        f"  Top-1 Effective Recommendation Acc: "
        f"{q_metrics['top1_effective_accuracy']*100:.1f}% "
        f"(within >=85% of oracle log-speedup)"
    )
    print(
        f"  Index vs No-Index Decision Acc    : "
        f"{q_metrics['index_decision_accuracy']*100:.1f}%"
    )
    print(
        f"  Geometric Mean Speedup Achieved   : "
        f"{q_metrics['geometric_mean_speedup']:.2f}x "
        f"(Oracle: {q_metrics['oracle_geometric_mean_speedup']:.2f}x)"
    )

    # Generate SHAP summary plot
    print(f"\nGenerating SHAP summary plot -> {shap_out} ...")
    top_shap = generate_shap_plot(model, X_test, shap_out)
    print("Top 10 SHAP Features (mean |SHAP value| on log_speedup):")
    for rank, (fname, sval) in enumerate(top_shap[:10], 1):
        print(f"  {rank:>2}. {fname:<34}: {sval:.4f}")

    # Save model bundle
    model_path = Path(model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    bundle: Dict[str, Any] = {
        "model": model,
        "feature_names": FEATURE_NAMES,
        "speedup_threshold": config.SPEEDUP_THRESHOLD,
        "train_templates": sorted(train_templates),
        "heldout_templates": sorted(test_templates),
        "metrics": {
            "train_mae_log": round(train_mae, 4),
            "train_rmse_log": round(train_rmse, 4),
            "heldout_mae_log": round(test_mae, 4),
            "heldout_rmse_log": round(test_rmse, 4),
            "heldout_spearman": round(test_spearman, 4),
            "top1_exact_accuracy": round(q_metrics["top1_exact_accuracy"], 4),
            "top1_effective_accuracy": round(q_metrics["top1_effective_accuracy"], 4),
            "index_decision_accuracy": round(q_metrics["index_decision_accuracy"], 4),
            "geometric_mean_speedup": round(q_metrics["geometric_mean_speedup"], 2),
            "oracle_geometric_mean_speedup": round(
                q_metrics["oracle_geometric_mean_speedup"], 2
            ),
        },
        "top_shap_features": top_shap,
    }
    joblib.dump(bundle, model_path)
    print(f"\nSaved trained model bundle to {model_path}")
    return bundle


if __name__ == "__main__":
    train_model()
