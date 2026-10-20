# IndexBuddy — ML-Based Automatic MySQL Index Recommender

IndexBuddy is an end-to-end **Machine Learning MySQL 8.x Index Advisor** with a closed-loop **React + Vite** (and **Streamlit**) live demonstration dashboard.

Given a SQL query against a live MySQL 8.x database, IndexBuddy:
1. Parses the SQL AST using **`sqlglot` (`dialect="mysql"`)** to extract tables, aliases, `WHERE`/`JOIN`/`GROUP BY`/`ORDER BY` columns, and predicate types (`equality`, `range`, `in`, `like_prefix`).
2. Generates single-column and two-column composite B-tree candidate indexes (`CREATE INDEX ...`), including both column orderings `(col1, col2)` and `(col2, col1)`.
3. Extracts **53 live database & optimizer features** per `(query, candidate)` pair (`information_schema` & `COUNT(DISTINCT)` cardinalities, null fraction, byte width, skewed MCV frequency distributions, B-tree left-prefix effective selectivity, estimated index size in MB, and baseline MySQL `EXPLAIN` plan features).
4. Uses a trained **XGBoost Regressor (`models/xgb_speedup.joblib`)** to predict $\log(\text{speedup}) = \log(t_{\text{before}} / t_{\text{after}})$ and recommends the top candidate **if and only if** its predicted speedup meets the configurable threshold (default `1.50x`), otherwise recommending **No Index**.
5. Executes `CREATE INDEX` on the live MySQL server **only after explicit user confirmation**, re-runs the query over 10 executions (discarding warm-up), and compares **Predicted vs. Actual Speedup** with `EXPLAIN`, `EXPLAIN ANALYZE`, visual bar charts, and a plain-English diagnostic explanation.

---

## Repository Structure

```text
IndexBuddy/
├── config.py                       # Loads MySQL connection & model settings from .env
├── .env.example                    # Template for MySQL credentials (never commit .env)
├── requirements.txt                # Clean Python dependencies
├── run_demo.py                     # One-command launcher for FastAPI + React/Vite UI
├── db/
│   └── mysql_adapter.py            # Abstract DBAdapter base + concrete MySQLAdapter (MySQL 8.x)
├── data/
│   └── setup_orders_benchmark.py   # Creates 1,000,000-row orders_benchmark & customers_benchmark
├── parsing/
│   └── query_parser.py             # sqlglot MySQL AST parser & canonical template normalizer
├── candidates/
│   └── generator.py                # Single-column & 2-column composite candidate index generator
├── features/
│   └── extract.py                  # Live catalog, selectivity, index size & EXPLAIN feature extractor
├── labeling/
│   ├── build_training_set.py       # Restartable empirical benchmark runner (315 queries, 21 templates)
│   └── training_data.csv           # 855 empirically measured (query, candidate) speedup rows
├── models/
│   ├── train.py                    # Template-split XGBoost trainer + SHAP plot generator
│   ├── predict.py                  # SpeedupPredictor inference & per-candidate SHAP explainer
│   └── xgb_speedup.joblib          # Trained XGBoost model bundle
├── selection/
│   └── recommend.py                # ML-driven candidate ranking, threshold gate & discrepancy analyzer
├── baselines/
│   └── rule_based.py               # No-Index baseline & legacy Rule-Based Scorer for comparison
├── evaluation/
│   ├── evaluate_baselines.py       # Evaluates baselines vs XGBoost on held-out templates
│   ├── results.md                  # Full empirical evaluation report (MAE, RMSE, Spearman, Top-1)
│   └── shap_summary.png            # SHAP beeswarm feature importance summary plot
├── app/
│   ├── api_server.py               # FastAPI backend serving live MySQL & XGBoost endpoints
│   └── dashboard.py                # 6-step Streamlit dashboard
├── frontend/                       # Modern React + Vite 6-step Live Demo & Evaluation Lab UI
└── tests/
    └── test_parser_and_candidates.py # Unit tests for sqlglot parser & candidate generator
```

---

## Setup & Exact Run Order

### 1. Configure `.env`
Copy `.env.example` to `.env` and fill in your local MySQL 8.x credentials:
```bash
cp .env.example .env
```
```ini
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=your_password_here
MYSQL_DB=indexbuddy
```

### 2. Install Dependencies
```bash
# Python dependencies
pip install -r requirements.txt

# React + Vite frontend dependencies
cd frontend
npm install
npm run build
cd ..
```

### 3. Run Pipeline in Order

```bash
# Stage (a): Populate 1,000,000 rows in orders_benchmark (0 secondary indexes) + ANALYZE TABLE
python -m data.setup_orders_benchmark

# Stage (b): Run unit tests & verify sqlglot parser + candidate generator
pytest -v tests/test_parser_and_candidates.py
python -m candidates.generator

# Stage (c): Build empirical training dataset (315 queries across 21 templates, restartable)
python -u -m labeling.build_training_set --workers 4

# Stage (d): Train XGBoost model (split strictly by query template), generate SHAP plot & evaluate baselines
python -m models.train
python -m evaluation.evaluate_baselines

# Stage (e): Launch the Live Demo Dashboard
# Option 1 (Recommended — React + Vite UI + FastAPI Backend):
python run_demo.py
# -> Opens http://localhost:5173 (React + Vite UI) and http://127.0.0.1:8000 (API + Built UI)

# Option 2 (Streamlit UI):
streamlit run app/dashboard.py
```

---

## Held-Out Template Evaluation Summary (`evaluation/results.md`)

Queries are split **strictly by canonical SQL template** (`16` training templates / `600` pairs vs. `5` held-out unseen templates `T03, T07, T11, T16, T19` / `255` pairs):

| Method | MAE ($\log$ speedup) $\downarrow$ | RMSE ($\log$ speedup) $\downarrow$ | Spearman $\rho$ $\uparrow$ | Top-1 Exact Acc $\uparrow$ | Top-1 Effective Acc $\uparrow$ | Decision Acc $\uparrow$ | Geometric Mean Speedup $\uparrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **No-Index Baseline** | `4.5055` | `5.5096` | `0.0000` | `8.0%` | `8.0%` | `8.0%` | **`1.00x`** *(Oracle: 488.35x)* |
| **Legacy Rule-Based Scorer** | `3.7428` | `4.5245` | `0.5337` | `40.0%` | `100.0%` | `100.0%` | **`429.50x`** *(Oracle: 488.35x)* |
| **XGBoost Speedup Model (Ours)** | **`0.9207`** | **`1.3092`** | **`0.8415`** | **`68.0%`** | **`100.0%`** | **`100.0%`** | **`461.43x`** *(Oracle: 488.35x)* |

---

## Professor Live Demo Script (5 Minutes)

1. **Top Status Bar & Step 1 (Schema & Skew)**:
   - Point out the live MySQL 8.0.46 connection indicator, `orders_benchmark` (`1,000,000` rows), and `0` active secondary indexes.
   - Show the live `COUNT(DISTINCT)` cardinalities (`customer_id` ~100k, `product_id` 5,000, `status` 5) and the **`status` skew chart** (`'completed'` = ~70% of all rows).
2. **Walkthrough A — High-Selectivity Composite Query (Held-Out Template `T11`)**:
   - In **Step 2**, select Demo Query #1:
     ```sql
     SELECT order_id, customer_id, product_id, order_date, amount
     FROM orders_benchmark
     WHERE customer_id = 42310 AND order_date >= '2023-06-01'
     ORDER BY order_date DESC LIMIT 25
     ```
   - Click **Run Query With NO Index (10 Runs)**. Observe median time (~750–1,100 ms), `EXPLAIN` (`type = ALL`, `rows = 991,211`, `Using where; Using filesort`), and the live `EXPLAIN ANALYZE` tree.
   - In **Step 3**, click **Get Recommendation**. Show how XGBoost ranks `(customer_id, order_date)` #1 (~2,110x predicted speedup) and `(customer_id)` #2 (~1,735x), while rejecting `(order_date)` (`1.39x`) and the reversed index `(order_date, customer_id)` (`1.03x`) because `order_date` is a range predicate that violates B-tree left-prefix equality narrowing.
   - In **Step 4**, review `CREATE INDEX idx_orders_benchmark_customer_id_order_date ON orders_benchmark (customer_id, order_date);` and click **Confirm & Execute CREATE INDEX**.
   - In **Step 5**, click **Run Query With Index (10 Runs)**. Observe median latency drop to **< 1 ms**, `type = range`/`ref`, and `rows` drop from `991,211` to `~5`.
   - In **Step 6**, inspect the **Predicted vs. Actual Speedup** table, bar charts, plain-English diagnostic explanation, and saved `results/run_<timestamp>.json`.
   - Click **Reset Demo** in the top bar to drop the created index.
3. **Walkthrough B — Low-Selectivity Negative Case (Held-Out Template `T03`)**:
   - In **Step 2**, select Demo Query #2:
     ```sql
     SELECT status, COUNT(*) AS order_cnt, AVG(amount) AS avg_amt
     FROM orders_benchmark
     WHERE status = 'completed'
     GROUP BY status
     ```
   - Run baseline (scans ~1M rows).
   - In **Step 3**, click **Get Recommendation**. Show that XGBoost predicts **`0.33x` speedup** (below the `1.50x` threshold) because `status = 'completed'` matches 69.97% of the table, and therefore **recommends NO INDEX**.
   - Optionally use the **Professor Verification Option** in Step 4 to force-create `idx_orders_benchmark_status` anyway and run Step 5 & 6 to prove empirically that MySQL's cost-based optimizer either ignores the index or gains ~1.0x (no speedup).
4. **SHAP & Evaluation Lab Tab**:
   - Switch to the **SHAP & Evaluation Lab** tab in the top bar to show the held-out template evaluation metrics and the `evaluation/shap_summary.png` beeswarm plot.
