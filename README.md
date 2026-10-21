# IndexBuddy — ML-Based Automatic MySQL Index Recommender

IndexBuddy is a schema-agnostic, end-to-end **Machine Learning MySQL 8.x Index Advisor** with a step-by-step **Streamlit** live demonstration dashboard (`app/dashboard.py`). It works on **any table in any MySQL database with zero code changes**.

Given a SQL `SELECT` query against a live MySQL 8.0.18+ database, IndexBuddy:
1. Parses the SQL AST using **`sqlglot` (`dialect="mysql"`)** to extract tables, aliases, `WHERE`/`JOIN`/`GROUP BY`/`ORDER BY` columns, and predicate types (`equality`, `range`, `in`, `like_prefix`).
2. Generates single-column and two-column composite B-tree candidate indexes (`CREATE INDEX ...`), including both column orderings `(col1, col2)` and `(col2, col1)` with a configurable cap per query.
3. Extracts **schema-agnostic live database & optimizer features** per `(query, candidate)` pair (`information_schema` & sampled `COUNT(DISTINCT)` cardinalities, null fraction, data type category, byte width, skewed MCV frequency distributions, B-tree left-prefix effective selectivity, estimated index size in MB, and baseline MySQL `EXPLAIN` plan features). The model never sees table or column names.
4. Uses a trained **XGBoost Regressor (`models/xgb_speedup.joblib`)** to predict $\log(\text{speedup}) = \log(t_{\text{before}} / t_{\text{after}})$ and recommends the top candidate **if and only if** its predicted speedup meets the configurable threshold (default `1.50x`), otherwise recommending **No Index**.
5. Executes `CREATE INDEX` on the live MySQL server **only after explicit user confirmation**, re-runs the query over 10 executions (discarding 1 warm-up), and compares **Predicted vs. Actual Speedup** with `EXPLAIN`, `EXPLAIN ANALYZE`, visual bar charts, and a plain-English diagnostic explanation generated from the live measurements.

---

## Real Data Sources & Citation

All training and evaluation data come from **real-world relational databases** hosted on the **CTU Prague Relational Learning Repository**:

> **Citation:**
> Jan Motl and Oliver Schulte, *"The CTU Prague Relational Learning Repository"*, arXiv:1511.03086, 2015.
> Public read-only server: `relational.fel.cvut.cz:3306` (credentials documented in `.env.example`).

### Selected Real Databases (6 Databases Across Diverse Domains)

| Database | Domain | Tables | Largest Table | Largest Table Rows | Total Rows |
| :--- | :--- | :---: | :--- | :---: | :---: |
| **`financial`** | Banking / Financial (`PKDD'99`) | 8 | `trans` | 1,056,320 | 1,079,680 |
| **`ErgastF1`** | Sports / Formula 1 Racing | 13 | `lapTimes` | 426,633 | 501,583 |
| **`imdb_full`** | Entertainment / Movies | 7 | `roles` | 3,431,966 | 4,578,535 |
| **`geneea`** | Government / Czech Parliament | 19 | `hl_check` | 1,401,568 | 1,729,344 |
| **`stats_CEB`** *(Held-Out Test)* | Q&A Community / Stats StackExchange | 8 | `comments` | 174,305 | 665,606 |
| **`Walmart_2014`** *(Held-Out Test)* | Retail / E-Commerce Forecasting | 4 | `train` | 421,570 | 538,992 |

All secondary indexes that shipped with the dumps were stripped (keeping only primary keys and saving the original definitions in `data/original_indexes.json`) and `ANALYZE TABLE` was run on every table so all candidate indexes are measured from a clean baseline on local MySQL.

---

## Repository Structure

```text
IndexBuddy/
├── config.py                       # Loads MySQL connection, CTU connection & model settings from .env
├── .env.example                    # Template for MySQL & public CTU credentials (never commit .env)
├── requirements.txt                # Clean Python dependencies
├── db/
│   └── mysql_adapter.py            # Abstract DBAdapter base + concrete MySQLAdapter (MySQL 8.x)
├── data/
│   ├── select_ctu_datasets.py      # Inspects CTU repository & prints qualifying real datasets
│   ├── download_and_restore.py     # Dumps databases to data/raw/, restores locally & strips secondary indexes
│   └── original_indexes.json       # Saved original non-PK indexes from the raw dumps
├── parsing/
│   └── query_parser.py             # sqlglot MySQL AST parser & SELECT-only validator
├── candidates/
│   └── generator.py                # Single-column & 2-column composite candidate index generator
├── features/
│   └── extract.py                  # Schema-agnostic catalog, selectivity, index size & EXPLAIN feature extractor
├── labeling/
│   ├── generate_workload.py        # Generates >=300 deduplicated queries per DB from real schemas & values
│   ├── build_training_set.py       # Restartable empirical benchmark runner on local MySQL
│   ├── generated_workload.csv      # 2,167 deduplicated real-schema queries across the 6 CTU databases
│   └── training_data.csv           # 1,146 empirically measured (query, candidate) rows on local MySQL
├── models/
│   ├── train.py                    # Leave-databases-out XGBoost trainer + SHAP plot generator
│   ├── predict.py                  # SpeedupPredictor inference & per-candidate SHAP explainer
│   └── xgb_speedup.joblib          # Trained XGBoost model bundle
├── selection/
│   └── recommend.py                # ML-driven candidate ranking, threshold gate & discrepancy analyzer
├── baselines/
│   └── rule_based.py               # No-Index baseline & legacy Rule-Based Scorer for comparison
├── evaluation/
│   ├── evaluate_baselines.py       # Evaluates No-Index, Rule-Based & XGBoost on held-out databases
│   ├── results.md                  # Full empirical leave-databases-out evaluation report
│   └── shap_summary.png            # SHAP beeswarm feature importance summary plot
├── app/
│   └── dashboard.py                # 6-step Streamlit interactive live demo dashboard
└── tests/
    └── test_parser_and_candidates.py # Unit tests for parser, candidates & features across multiple schemas
```

---

## Setup & Exact Run Order

### 1. Configure `.env`
Copy `.env.example` to `.env` and fill in your local MySQL 8.0.18+ credentials:
```bash
cp .env.example .env
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Pipeline in Order
```bash
# Stage 1 & 2: Inspect qualifying real CTU datasets
python -m data.select_ctu_datasets

# Stage 3: Dump from CTU (once, cached in data/raw/), restore to local MySQL,
# strip non-PK secondary indexes (saved in data/original_indexes.json), and run ANALYZE TABLE
python -m data.download_and_restore

# Stage 4: Run unit tests for parser, candidate generator, and feature extractor across schemas
pytest -v

# Stage 5: Generate >=300 deduplicated queries per database from real schemas/data and label on local MySQL
python -m labeling.generate_workload
python -m labeling.build_training_set

# Stage 6: Train XGBoost model (leave-databases-out split) & evaluate against baselines
python -m models.train
python -m evaluation.evaluate_baselines

# Stage 7: Launch the 6-step Streamlit Dashboard
streamlit run app/dashboard.py
```

---

## Leave-Databases-Out Evaluation Summary (`evaluation/results.md`)

- **Training Databases (4)**: `ErgastF1`, `financial`, `geneea`, `imdb_full` (`321` queries, `702` labeled pairs)
- **Held-Out Test Databases (2)**: `Walmart_2014`, `stats_CEB` (`198` queries, `444` labeled pairs — never seen during training)

| Method | MAE ($\log$ speedup) $\downarrow$ | RMSE ($\log$ speedup) $\uparrow$ | Spearman $\rho$ $\uparrow$ | Top-1 Exact $\uparrow$ | Top-1 Effective $\uparrow$ | Decision Acc $\uparrow$ | Geo-Mean Speedup $\uparrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **No-Index Baseline** | `1.9066` | `2.5136` | `0.0000` | `13.6%` | `13.6%` | `13.6%` | `1.00x` *(oracle 8.88x)* |
| **Legacy Rule-Based Scorer** | `1.3438` | `1.7670` | `0.5184` | `58.1%` | `76.8%` | `84.8%` | `6.76x` *(oracle 8.88x)* |
| **XGBoost Speedup Model (Ours)** | **`0.7300`** | **`1.1630`** | **`0.7415`** | **`65.7%`** | **`83.3%`** | **`89.4%`** | **`7.61x`** *(oracle 8.88x)* |

### Per Held-Out Database Breakdown

| Held-Out Database | Queries | Pairs | XGB MAE | Rule MAE | XGB Spearman | Rule Spearman | XGB Geo-Speedup | Rule Geo-Speedup | Oracle |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Walmart_2014`** | 98 | 230 | `0.3836` | `0.8011` | `0.7181` | `0.5292` | **`3.33x`** | `3.02x` | `3.54x` |
| **`stats_CEB`** | 100 | 214 | `1.1023` | `1.9272` | `0.7096` | `0.5758` | **`17.10x`** | `14.92x` | `21.86x` |

---

## Live Demo Script (6-Step Streamlit Dashboard)

Launch the dashboard with `streamlit run app/dashboard.py`:

1. **Connect & Select Any Database/Table (Sidebar + Top Status Bar)**:
   - Enter local MySQL host/port/user/password and click **Connect**.
   - Pick any held-out database and table (e.g., `stats_CEB.posts` or `Walmart_2014.train`). The top status bar always displays the live connection state, selected database, selected table, and total row count.
2. **Step 1 — Table Schema & Sample Records**:
   - Inspect auto-discovered column types, primary keys, sampled distinct counts, null fractions, existing indexes, and 20 live sample rows. Click **Next ➡️**.
3. **Step 2 — Choose or Edit Query & Run Baseline (No Index)**:
   - Choose from dynamically generated sample queries for the selected table, paste a custom `SELECT` query (e.g., as a purely illustrative example on a hypothetical e-commerce schema: `SELECT * FROM orders_benchmark WHERE customer_id = 42 AND payment_method = 'card'`), or upload a `.sql`/`.csv` file.
   - Click **▶ Run without index** to execute 10 timed runs (discarding 1 warm-up) and view the live median runtime, rows examined, `EXPLAIN`, and `EXPLAIN ANALYZE`. Click **Next ➡️**.
4. **Step 3 — ML Recommendation**:
   - Click **🤖 Get recommendation from IndexBuddy** to view all candidate indexes ranked by predicted speedup, selectivity, estimated index size (MB), and top SHAP factors.
   - Select the generated low-selectivity sample query (#3) to verify that the model predicts `< 1.50x` speedup and recommends **No Index**.
5. **Step 4 — Generated DDL & Explicit Confirmation**:
   - Inspect the `CREATE INDEX` DDL, estimated index size, and large-table warning. Click **✅ Confirm and create index** to build the index on local MySQL.
6. **Step 5 & Step 6 — Post-Index Benchmark, Comparison & Reset**:
   - Click **▶ Run same query with index** to measure the new median runtime over 10 runs, new `EXPLAIN`, and `EXPLAIN ANALYZE`.
   - Advance to **Step 6** to view the Predicted vs. Actual speedup table, error %, before/after rows examined, interactive bar chart, plain-English discrepancy analysis, and saved JSON report (`results/run_<timestamp>.json`).
   - Click **🔄 Reset demo (drop IndexBuddy index)** in the sidebar to drop only the `idx_ib_*` index created by IndexBuddy.

---

## Limitations of the Approach

1. **Workload Generated from Real Schemas & Data (No Real Query Logs)**: The public CTU Prague Relational Learning Repository provides real database schemas and real row distributions, but does not include production query logs. Queries in `labeling/generated_workload.csv` are systematically generated from the real schemas, real foreign keys, and real column value distributions (preserving actual data skew).
2. **Single-Query Scope vs. Workload-Level Trade-Offs**: IndexBuddy evaluates candidates for one `SELECT` query at a time and does not jointly optimize write amplification (`INSERT`/`UPDATE`/`DELETE` overhead) or shared prefix consolidation across a multi-query workload.
3. **Buffer Pool Warm-Cache Effects**: Because measurements run against a warm InnoDB buffer pool on local MySQL (discarding 1 warm-up run), measured wall-clock speedups reflect in-memory page scanning and B-tree traversal rather than cold-disk I/O latency.
4. **Up to 2-Column Composite B-Tree Indexes**: Candidate generation focuses on single-column and 2-column B-tree indexes and does not propose covering `INCLUDE`-style extensions, spatial (`R-tree`), or full-text indexes.
