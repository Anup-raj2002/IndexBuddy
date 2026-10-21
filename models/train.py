"""
models/train.py
───────────────
Trains an XGBoost regressor to predict ``log_speedup = log(time_before / time_after)``
from schema-agnostic MySQL catalog, query AST, candidate index position, and
baseline EXPLAIN features.

Key guarantees:
  - Splits train and test sets STRICTLY BY LEAVE-DATABASES-OUT
    (trains on ``config.TRAIN_DATABASES``, evaluates on ``config.HELDOUT_TEST_DATABASES``
    that the model never saw during training or hyperparameter selection).
  - Never feeds table names or column names to the model (uses only ``FEATURE_NAMES``).
  - Reports MAE, RMSE on log(speedup), Spearman rank correlation, and Top-1
    recommendation accuracy overall and per held-out database.
  - Produces a SHAP feature-importance summary plot at ``evaluation/shap_summary.png``.
  - Saves the trained model bundle to ``models/xgb_speedup.joblib``.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
    """Compute query-level recommendation metrics across queries in *df_eval*."""
    log_thresh = math.log(threshold)
    exact_matches = 0
    effective_matches = 0
    decision_matches = 0
    achieved_speedups: List[float] = []
    oracle_speedups: List[float] = []

    for _, q_df in df_eval.groupby("query_id"):
        best_actual_row = q_df.loc[q_df["log_speedup"].idxmax()]
        oracle_log = float(best_actual_row["log_speedup"])
        oracle_speedup = float(best_actual_row["speedup"])
        oracle_recommends = oracle_log >= log_thresh
        oracle_action = (
            str(best_actual_row["index_name"]) if oracle_recommends else "NO_INDEX"
        )
        oracle_eff_speedup = oracle_speedup if oracle_recommends else 1.0
        oracle_speedups.append(oracle_eff_speedup)

        best_pred_row = q_df.loc[q_df[pred_log_col].idxmax()]
        pred_log = float(best_pred_row[pred_log_col])
        pred_recommends = pred_log >= log_thresh
        pred_action = (
            str(best_pred_row["index_name"]) if pred_recommends else "NO_INDEX"
        )

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
            and chosen_actual_log >= 0.80 * oracle_log
        ):
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
    """Compute SHAP values on held-out databases and save a summary plot."""
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
        "SHAP Feature Importance Summary (Leave-Databases-Out Evaluation)",
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
    """Train XGBoost speedup regressor with Leave-Databases-Out splitting."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} labeled (query, candidate) pairs from {csv_path}")

    train_df = df[df["database_name"].isin(config.TRAIN_DATABASES)].copy()
    test_df = df[df["database_name"].isin(config.HELDOUT_TEST_DATABASES)].copy()

    train_dbs = sorted(train_df["database_name"].unique().tolist())
    test_dbs = sorted(test_df["database_name"].unique().tolist())
    overlap_dbs = set(train_dbs).intersection(set(test_dbs))
    assert not overlap_dbs, f"Database leakage detected: {overlap_dbs}"

    print(
        f"Leave-Databases-Out Split Verified (0 database overlap):\n"
        f"  Train databases ({len(train_dbs)}): {train_dbs} "
        f"-> {train_df['query_id'].nunique()} queries, {len(train_df)} pairs\n"
        f"  Held-out test databases ({len(test_dbs)}): {test_dbs} "
        f"-> {test_df['query_id'].nunique()} queries, {len(test_df)} pairs"
    )

    X_train = train_df[FEATURE_NAMES].astype(float)
    y_train = train_df["log_speedup"].astype(float)

    X_test = test_df[FEATURE_NAMES].astype(float)
    y_test = test_df["log_speedup"].astype(float)

    model = XGBRegressor(
        n_estimators=220,
        max_depth=4,
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

    train_pred_log = model.predict(X_train)
    test_pred_log = model.predict(X_test)
    train_df["pred_log_speedup"] = train_pred_log
    test_df["pred_log_speedup"] = test_pred_log
    test_df["pred_speedup"] = np.exp(test_pred_log)

    train_mae = float(mean_absolute_error(y_train, train_pred_log))
    train_rmse = float(math.sqrt(mean_squared_error(y_train, train_pred_log)))
    test_mae = float(mean_absolute_error(y_test, test_pred_log))
    test_rmse = float(math.sqrt(mean_squared_error(y_test, test_pred_log)))

    spearman_res = spearmanr(y_test, test_pred_log)
    test_spearman = float(spearman_res.statistic) if not math.isnan(float(spearman_res.statistic)) else 0.0

    q_metrics = compute_query_level_accuracy(test_df, "pred_log_speedup")

    # Per held-out database breakdown
    per_db_metrics: Dict[str, Dict[str, float]] = {}
    for db_name, db_grp in test_df.groupby("database_name"):
        db_y = db_grp["log_speedup"].astype(float)
        db_p = db_grp["pred_log_speedup"].astype(float)
        db_mae = float(mean_absolute_error(db_y, db_p))
        db_rmse = float(math.sqrt(mean_squared_error(db_y, db_p)))
        sp = spearmanr(db_y, db_p)
        db_rho = float(sp.statistic) if not math.isnan(float(sp.statistic)) else 0.0
        db_q = compute_query_level_accuracy(db_grp, "pred_log_speedup")
        per_db_metrics[str(db_name)] = {
            "num_queries": db_q["num_queries"],
            "num_pairs": float(len(db_grp)),
            "mae_log": round(db_mae, 4),
            "rmse_log": round(db_rmse, 4),
            "spearman": round(db_rho, 4),
            "top1_exact_accuracy": round(db_q["top1_exact_accuracy"], 4),
            "top1_effective_accuracy": round(db_q["top1_effective_accuracy"], 4),
            "index_decision_accuracy": round(db_q["index_decision_accuracy"], 4),
            "geometric_mean_speedup": round(db_q["geometric_mean_speedup"], 2),
            "oracle_geometric_mean_speedup": round(db_q["oracle_geometric_mean_speedup"], 2),
        }

    print("\n=== XGBoost Speedup Model Evaluation (Leave-Databases-Out) ===")
    print(f"  Train MAE (log speedup)           : {train_mae:.4f}")
    print(f"  Train RMSE (log speedup)          : {train_rmse:.4f}")
    print(f"  Held-Out MAE (log speedup)        : {test_mae:.4f}")
    print(f"  Held-Out RMSE (log speedup)       : {test_rmse:.4f}")
    print(f"  Held-Out Spearman Correlation (rho): {test_spearman:.4f}")
    print(
        f"  Top-1 Exact Recommendation Acc    : "
        f"{q_metrics['top1_exact_accuracy']*100:.1f}%"
    )
    print(
        f"  Top-1 Effective Recommendation Acc: "
        f"{q_metrics['top1_effective_accuracy']*100:.1f}%"
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

    print("\n--- Per Held-Out Database Breakdown ---")
    for db_name, m in per_db_metrics.items():
        print(
            f"  [{db_name:<13}] queries={int(m['num_queries']):>2d} | pairs={int(m['num_pairs']):>3d} | "
            f"MAE={m['mae_log']:.4f} | RMSE={m['rmse_log']:.4f} | Spearman={m['spearman']:.4f} | "
            f"Top1_Exact={m['top1_exact_accuracy']*100:.1f}% | Top1_Eff={m['top1_effective_accuracy']*100:.1f}% | "
            f"GeoMean={m['geometric_mean_speedup']:.2f}x"
        )

    print(f"\nGenerating SHAP summary plot -> {shap_out} ...")
    top_shap = generate_shap_plot(model, X_test, shap_out)
    print("Top 10 SHAP Features (mean |SHAP value| on log_speedup):")
    for rank, (fname, sval) in enumerate(top_shap[:10], 1):
        print(f"  {rank:>2}. {fname:<34}: {sval:.4f}")

    model_path = Path(model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    bundle: Dict[str, Any] = {
        "model": model,
        "feature_names": FEATURE_NAMES,
        "speedup_threshold": config.SPEEDUP_THRESHOLD,
        "train_databases": train_dbs,
        "heldout_databases": test_dbs,
        "per_database_metrics": per_db_metrics,
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
