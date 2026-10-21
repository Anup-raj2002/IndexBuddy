"""
evaluation/evaluate_baselines.py
────────────────────────────────
Compares three strategies on the HELD-OUT real databases (never seen in training):
  1. No-Index baseline
  2. Legacy rule-based scorer
  3. XGBoost speedup model

Writes evaluation/results.md. Every number and every sentence is computed from the
measured data; nothing is hardcoded about any specific database or table.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

import config
from baselines.rule_based import LegacyRuleBasedScorer
from features.extract import FEATURE_NAMES
from models.predict import SpeedupPredictor
from models.train import compute_query_level_accuracy


def _metrics(df: pd.DataFrame, col: str) -> Dict[str, float]:
    y, p = df["log_speedup"].astype(float), df[col].astype(float)
    rho = 0.0
    if len(df) > 2 and np.std(p) > 1e-9:
        r = spearmanr(y, p).statistic
        rho = 0.0 if math.isnan(float(r)) else float(r)
    q = compute_query_level_accuracy(df, col)
    return {
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(math.sqrt(mean_squared_error(y, p))),
        "rho": rho,
        "top1_exact": q["top1_exact_accuracy"],
        "top1_eff": q["top1_effective_accuracy"],
        "decision": q["index_decision_accuracy"],
        "geo": q["geometric_mean_speedup"],
        "oracle": q["oracle_geometric_mean_speedup"],
    }


def _row(label: str, m: Dict[str, float]) -> str:
    return (
        f"| {label} | {m['mae']:.4f} | {m['rmse']:.4f} | {m['rho']:.4f} | "
        f"{m['top1_exact']*100:.1f}% | {m['top1_eff']*100:.1f}% | "
        f"{m['decision']*100:.1f}% | {m['geo']:.2f}x (oracle {m['oracle']:.2f}x) |"
    )


HEADER = (
    "| Method | MAE (log speedup) | RMSE (log speedup) | Spearman | Top-1 exact | "
    "Top-1 effective | Index/no-index decision | Geo-mean speedup |\n"
    "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
)


def evaluate_all(
    csv_path: str = config.LABEL_CSV,
    results_md_path: str = "evaluation/results.md",
) -> Dict[str, Any]:
    df = pd.read_csv(csv_path)
    train_df = df[df["database_name"].isin(config.TRAIN_DATABASES)].copy()
    test_df = df[df["database_name"].isin(config.HELDOUT_TEST_DATABASES)].copy()
    if test_df.empty:
        raise SystemExit("No labeled rows for the held-out databases. Run labeling first.")

    predictor = SpeedupPredictor()
    bundle = predictor.load()

    test_df["noindex"] = 0.0
    test_df["rule"] = [
        LegacyRuleBasedScorer.predict_log_speedup(r.to_dict()) for _, r in test_df.iterrows()
    ]
    test_df["xgb"] = predictor.model.predict(test_df[FEATURE_NAMES].astype(float))

    methods = [("No-index baseline", "noindex"), ("Legacy rule-based scorer", "rule"),
               ("XGBoost speedup model (ours)", "xgb")]
    overall = {label: _metrics(test_df, col) for label, col in methods}

    per_db: List[Dict[str, Any]] = []
    for db_name, g in test_df.groupby("database_name"):
        per_db.append({
            "db": db_name, "queries": int(g["query_id"].nunique()), "pairs": len(g),
            "xgb": _metrics(g, "xgb"), "rule": _metrics(g, "rule"),
        })

    wins_mae = sum(1 for r in per_db if r["xgb"]["mae"] < r["rule"]["mae"])
    wins_rho = sum(1 for r in per_db if r["xgb"]["rho"] > r["rule"]["rho"])
    wins_geo = sum(1 for r in per_db if r["xgb"]["geo"] > r["rule"]["geo"])

    L: List[str] = []
    L.append("# IndexBuddy: Empirical Evaluation (leave-databases-out)\n")
    L.append("## 1. Setup\n")
    L.append(f"- Data: real databases from the CTU Prague Relational Learning Repository, "
             f"restored on local MySQL; all timings measured locally.")
    L.append(f"- Training databases ({train_df['database_name'].nunique()}): "
             f"{', '.join(sorted(train_df['database_name'].unique()))} "
             f"({train_df['query_id'].nunique()} queries, {len(train_df)} labeled pairs)")
    L.append(f"- Held-out databases ({test_df['database_name'].nunique()}): "
             f"{', '.join(sorted(test_df['database_name'].unique()))} "
             f"({test_df['query_id'].nunique()} queries, {len(test_df)} labeled pairs)")
    L.append(f"- Recommendation threshold: predicted speedup >= {config.SPEEDUP_THRESHOLD:.2f}x, "
             f"otherwise no index.")
    L.append("- Queries are generated from the real schemas and real value distributions "
             "(these datasets contain no real query logs). Labels are measured speedups "
             "(median of timed runs, warm-up discarded) with and without each candidate index.\n")

    L.append("## 2. Overall comparison on held-out databases\n")
    L.append(HEADER)
    for label, _ in methods:
        L.append(_row(label, overall[label]))
    L.append("")

    L.append("## 3. Per held-out database\n")
    L.append("| Database | Queries | Pairs | XGB MAE | Rule MAE | XGB Spearman | Rule Spearman | "
             "XGB geo-speedup | Rule geo-speedup | Oracle |")
    L.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for r in per_db:
        L.append(
            f"| {r['db']} | {r['queries']} | {r['pairs']} | {r['xgb']['mae']:.4f} | "
            f"{r['rule']['mae']:.4f} | {r['xgb']['rho']:.4f} | {r['rule']['rho']:.4f} | "
            f"{r['xgb']['geo']:.2f}x | {r['rule']['geo']:.2f}x | {r['xgb']['oracle']:.2f}x |"
        )
    L.append("")

    L.append("## 4. Top SHAP features (held-out data)\n")
    L.append("Summary plot: `evaluation/shap_summary.png`.\n")
    L.append("| Rank | Feature | Mean abs SHAP |\n| :---: | :--- | :---: |")
    for i, (fname, val) in enumerate(bundle.get("top_shap_features", [])[:12], 1):
        L.append(f"| {i} | `{fname}` | {float(val):.4f} |")
    L.append("")

    n = len(per_db)
    L.append("## 5. Findings (computed from the results above)\n")
    L.append(f"- The ML model has lower log-speedup MAE than the rule-based scorer on "
             f"{wins_mae} of {n} held-out databases, higher rank correlation on {wins_rho} of {n}, "
             f"and higher achieved geometric-mean speedup on {wins_geo} of {n}.")
    x, rl = overall["XGBoost speedup model (ours)"], overall["Legacy rule-based scorer"]
    L.append(f"- Overall MAE: {x['mae']:.3f} (ML) vs {rl['mae']:.3f} (rules); "
             f"Spearman: {x['rho']:.3f} vs {rl['rho']:.3f}.")
    L.append(f"- Achieved geometric-mean speedup: {x['geo']:.2f}x (ML) vs {rl['geo']:.2f}x (rules) "
             f"vs {x['oracle']:.2f}x for the oracle that always picks the best measured index.")
    L.append("\n## 6. Limitations\n")
    L.append("- Few databases and a modest number of labeled queries per database; results "
             "can change with more data.")
    L.append("- Workloads are synthetic-from-real-data, not production query logs.")
    L.append("- Timings come from a single local MySQL instance; cached pages and shared "
             "hardware add noise to small queries.")

    out = Path(results_md_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
    return {"overall": overall, "per_db": per_db}


if __name__ == "__main__":
    evaluate_all()