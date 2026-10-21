# IndexBuddy: Empirical Evaluation (leave-databases-out)

## 1. Setup

- Data: real databases from the CTU Prague Relational Learning Repository, restored on local MySQL; all timings measured locally.
- Training databases (4): ErgastF1, financial, geneea, imdb_full (321 queries, 702 labeled pairs)
- Held-out databases (2): Walmart_2014, stats_CEB (198 queries, 444 labeled pairs)
- Recommendation threshold: predicted speedup >= 1.50x, otherwise no index.
- Queries are generated from the real schemas and real value distributions (these datasets contain no real query logs). Labels are measured speedups (median of timed runs, warm-up discarded) with and without each candidate index.

## 2. Overall comparison on held-out databases

| Method | MAE (log speedup) | RMSE (log speedup) | Spearman | Top-1 exact | Top-1 effective | Index/no-index decision | Geo-mean speedup |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| No-index baseline | 1.9066 | 2.5136 | 0.0000 | 13.6% | 13.6% | 13.6% | 1.00x (oracle 8.88x) |
| Legacy rule-based scorer | 1.3438 | 1.7670 | 0.5184 | 58.1% | 76.8% | 84.8% | 6.76x (oracle 8.88x) |
| XGBoost speedup model (ours) | 0.7300 | 1.1630 | 0.7415 | 65.7% | 83.3% | 89.4% | 7.61x (oracle 8.88x) |

## 3. Per held-out database

| Database | Queries | Pairs | XGB MAE | Rule MAE | XGB Spearman | Rule Spearman | XGB geo-speedup | Rule geo-speedup | Oracle |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Walmart_2014 | 98 | 230 | 0.3836 | 0.8011 | 0.7181 | 0.5292 | 3.33x | 3.02x | 3.54x |
| stats_CEB | 100 | 214 | 1.1023 | 1.9272 | 0.7096 | 0.5758 | 17.10x | 14.92x | 21.86x |

## 4. Top SHAP features (held-out data)

Summary plot: `evaluation/shap_summary.png`.

| Rank | Feature | Mean abs SHAP |
| :---: | :--- | :---: |
| 1 | `effective_index_selectivity` | 0.7214 |
| 2 | `baseline_explain_rows` | 0.5545 |
| 3 | `estimated_index_size_mb` | 0.2100 |
| 4 | `col1_cardinality` | 0.1790 |
| 5 | `estimated_index_rows_scanned` | 0.1426 |
| 6 | `table_rows` | 0.1305 |
| 7 | `limit_ratio` | 0.1304 |
| 8 | `col1_selectivity` | 0.1119 |
| 9 | `baseline_explain_filtered` | 0.0917 |
| 10 | `col1_null_fraction` | 0.0687 |
| 11 | `num_where_cols` | 0.0645 |
| 12 | `log_effective_selectivity` | 0.0457 |

## 5. Findings (computed from the results above)

- The ML model has lower log-speedup MAE than the rule-based scorer on 2 of 2 held-out databases, higher rank correlation on 2 of 2, and higher achieved geometric-mean speedup on 2 of 2.
- Overall MAE: 0.730 (ML) vs 1.344 (rules); Spearman: 0.741 vs 0.518.
- Achieved geometric-mean speedup: 7.61x (ML) vs 6.76x (rules) vs 8.88x for the oracle that always picks the best measured index.

## 6. Limitations

- Few databases and a modest number of labeled queries per database; results can change with more data.
- Workloads are synthetic-from-real-data, not production query logs.
- Timings come from a single local MySQL instance; cached pages and shared hardware add noise to small queries.
