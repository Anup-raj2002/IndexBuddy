"""
app/api_server.py
─────────────────
FastAPI backend powering the React + Vite live MySQL Index Recommender dashboard.

Every number returned by this server (table row counts, column cardinalities,
EXPLAIN plans, EXPLAIN ANALYZE trees, 10-run median latencies, XGBoost predicted
speedups, SHAP attributions, and baseline evaluations) comes directly from the
live MySQL 8.x server or the trained XGBoost model.

Run with:
    uvicorn app.api_server:app --reload --port 8000
"""
from __future__ import annotations

import json
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from mysql.connector import Error as MySQLError
from pydantic import BaseModel, Field

import config
from baselines.rule_based import LegacyRuleBasedScorer
from db.mysql_adapter import MySQLAdapter
from evaluation.evaluate_baselines import evaluate_all
from features.extract import StatsCache
from models.predict import SpeedupPredictor
from parsing.query_parser import parse_query
from selection.recommend import explain_prediction_vs_actual, recommend_index


app = FastAPI(
    title="IndexBuddy ML API",
    description="Live MySQL 8.x & XGBoost Automatic Index Recommender API",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


DEMO_QUERIES: List[Dict[str, str]] = [
    {
        "id": "demo_t11_composite",
        "label": "1. High-Selectivity Equality + Date Range + ORDER BY (Held-Out T11)",
        "template_id": "T11",
        "category": "High Selectivity (Recommends Composite Index)",
        "sql": (
            "SELECT order_id, customer_id, product_id, order_date, amount "
            "FROM orders_benchmark "
            "WHERE customer_id = 42310 AND order_date >= '2023-06-01' "
            "ORDER BY order_date DESC LIMIT 25"
        ),
        "description": (
            "Held-out template T11: equality filter on high-cardinality customer_id "
            "combined with a range predicate and ORDER BY ... LIMIT on order_date."
        ),
    },
    {
        "id": "demo_t03_low_selectivity",
        "label": "2. Low-Selectivity Skewed Filter on status='completed' (Held-Out T03 — Recommends NO Index)",
        "template_id": "T03",
        "category": "Low Selectivity Negative Case (Recommends No Index)",
        "sql": (
            "SELECT status, COUNT(*) AS order_cnt, AVG(amount) AS avg_amt "
            "FROM orders_benchmark "
            "WHERE status = 'completed' "
            "GROUP BY status"
        ),
        "description": (
            "Held-out template T03: status = 'completed' matches ~70% of the 1M-row table. "
            "Demonstrates that the ML model discriminates against unhelpful indexes."
        ),
    },
    {
        "id": "demo_t07_multicol_eq",
        "label": "3. Multi-Column Equality on product_id & region (Held-Out T07)",
        "template_id": "T07",
        "category": "Composite Equality (Recommends 2-Column Index)",
        "sql": (
            "SELECT order_id, customer_id, product_id, region, amount "
            "FROM orders_benchmark "
            "WHERE product_id = 1250 AND region = 'North'"
        ),
        "description": (
            "Held-out template T07: two equality predicates on medium-cardinality product_id "
            "and low-cardinality region."
        ),
    },
    {
        "id": "demo_t16_join",
        "label": "4. Multi-Table JOIN with Product & Customer Tier Filter (Held-Out T16)",
        "template_id": "T16",
        "category": "Multi-Table JOIN (Held-Out Test Template)",
        "sql": (
            "SELECT o.order_id, o.product_id, o.amount, c.name, c.tier "
            "FROM orders_benchmark o "
            "JOIN customers_benchmark c ON o.customer_id = c.customer_id "
            "WHERE o.product_id = 2048 AND c.tier = 'gold'"
        ),
        "description": (
            "Held-out template T16: joins orders_benchmark (1M rows) with customers_benchmark "
            "(100k rows) while filtering on o.product_id and c.tier."
        ),
    },
    {
        "id": "demo_t19_groupby",
        "label": "5. Product Filter with GROUP BY Region Aggregation (Held-Out T19)",
        "template_id": "T19",
        "category": "GROUP BY Aggregation (Held-Out Test Template)",
        "sql": (
            "SELECT region, COUNT(*) AS cnt, AVG(amount) AS avg_amt "
            "FROM orders_benchmark "
            "WHERE product_id = 777 "
            "GROUP BY region"
        ),
        "description": (
            "Held-out template T19: filters by product_id and aggregates order counts "
            "and average amount across regions."
        ),
    },
]


_stats_cache: Optional[StatsCache] = None
_predictor: Optional[SpeedupPredictor] = None


def _get_db() -> MySQLAdapter:
    db = MySQLAdapter()
    db.connect()
    return db


def _get_predictor() -> SpeedupPredictor:
    global _predictor
    if _predictor is None:
        _predictor = SpeedupPredictor()
        _predictor.load()
    return _predictor


def _serialize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in row.items():
        if hasattr(v, "isoformat"):
            out[k] = v.isoformat()
        elif isinstance(v, (int, float, str, bool)) or v is None:
            out[k] = v
        else:
            try:
                out[k] = float(v)
            except (TypeError, ValueError):
                out[k] = str(v)
    return out


def _measure_query_detailed(
    db: MySQLAdapter,
    sql: str,
    runs: int = config.DASHBOARD_RUNS,
    warmup: int = 1,
) -> Dict[str, Any]:
    """Run *sql* over *runs* (discarding *warmup*), plus EXPLAIN and EXPLAIN ANALYZE."""
    all_times_ms: List[float] = []
    measured_times_s: List[float] = []
    sample_rows: List[Dict[str, Any]] = []

    for i in range(runs):
        t0 = time.perf_counter()
        res = db.execute(sql)
        elapsed_s = time.perf_counter() - t0
        all_times_ms.append(round(elapsed_s * 1000.0, 3))
        if i == 0:
            sample_rows = [_serialize_row(r) for r in res[:10]]
        if i >= warmup:
            measured_times_s.append(elapsed_s)

    median_s = statistics.median(measured_times_s) if measured_times_s else 0.0
    mean_s = statistics.mean(measured_times_s) if measured_times_s else 0.0
    min_s = min(measured_times_s) if measured_times_s else 0.0
    max_s = max(measured_times_s) if measured_times_s else 0.0

    explain_rows = [_serialize_row(r) for r in db.explain(sql)]
    explain_analyze_tree = db.explain_analyze(sql)

    # Extract primary table row from EXPLAIN
    primary_exp = None
    for r in explain_rows:
        tbl = str(r.get("table") or "").lower()
        if tbl in (config.BENCHMARK_TABLE.lower(), "o"):
            primary_exp = r
            break
    if primary_exp is None and explain_rows:
        primary_exp = explain_rows[0]

    rows_examined = int((primary_exp or {}).get("rows") or 0)
    access_type = str((primary_exp or {}).get("type") or "ALL")
    key_used = str((primary_exp or {}).get("key") or "")
    extra_info = str((primary_exp or {}).get("Extra") or "")

    return {
        "runs_total": runs,
        "warmup_discarded": warmup,
        "all_run_times_ms": all_times_ms,
        "measured_run_times_ms": [round(t * 1000.0, 3) for t in measured_times_s],
        "median_time_s": round(median_s, 6),
        "median_time_ms": round(median_s * 1000.0, 3),
        "mean_time_ms": round(mean_s * 1000.0, 3),
        "min_time_ms": round(min_s * 1000.0, 3),
        "max_time_ms": round(max_s * 1000.0, 3),
        "rows_examined": rows_examined,
        "access_type": access_type,
        "key_used": key_used,
        "extra": extra_info,
        "explain_rows": explain_rows,
        "explain_analyze": explain_analyze_tree,
        "query_result_preview": sample_rows,
    }


# ── Request models ──────────────────────────────────────────────


class RunQueryRequest(BaseModel):
    sql: str
    runs: int = Field(default=config.DASHBOARD_RUNS, ge=2, le=25)
    warmup: int = Field(default=1, ge=0, le=5)
    drop_secondary_indexes_first: bool = False


class RecommendRequest(BaseModel):
    sql: str
    threshold: float = Field(default=config.SPEEDUP_THRESHOLD, ge=1.0, le=100.0)


class CreateIndexRequest(BaseModel):
    table: str = config.BENCHMARK_TABLE
    index_name: str
    ddl: str


class CompareRequest(BaseModel):
    sql: str
    chosen_index_name: Optional[str] = None
    chosen_columns: Optional[str] = None
    ddl_executed: Optional[str] = None
    predicted_speedup: float
    estimated_selectivity: float = 1.0
    threshold: float = config.SPEEDUP_THRESHOLD
    before_measurement: Dict[str, Any]
    after_measurement: Dict[str, Any]
    ranked_candidates: List[Dict[str, Any]] = Field(default_factory=list)


# ── Endpoints ───────────────────────────────────────────────────


@app.get("/api/status")
def get_db_status() -> Dict[str, Any]:
    """Return live MySQL connection status, table row counts, and active indexes."""
    try:
        db = _get_db()
        try:
            version = str(db.execute_scalar("SELECT VERSION()") or "8.x")
            exists = db.table_exists(config.BENCHMARK_TABLE)
            total_rows = db.row_count(config.BENCHMARK_TABLE) if exists else 0
            cust_rows = (
                db.row_count("customers_benchmark")
                if db.table_exists("customers_benchmark")
                else 0
            )
            sec_indexes = (
                db.get_non_primary_indexes(config.BENCHMARK_TABLE) if exists else []
            )
            return {
                "connected": True,
                "host": db.host,
                "port": db.port,
                "database": db.database,
                "mysql_version": version,
                "table_name": config.BENCHMARK_TABLE,
                "total_rows": total_rows,
                "helper_table_rows": cust_rows,
                "secondary_indexes": sec_indexes,
                "default_threshold": config.SPEEDUP_THRESHOLD,
            }
        finally:
            db.close()
    except Exception as exc:
        return {
            "connected": False,
            "host": config.MYSQL_HOST,
            "port": config.MYSQL_PORT,
            "database": config.MYSQL_DB,
            "table_name": config.BENCHMARK_TABLE,
            "total_rows": 0,
            "secondary_indexes": [],
            "error": str(exc),
        }


@app.get("/api/step1/schema")
def get_step1_schema() -> Dict[str, Any]:
    """Return live schema metadata, column cardinalities, status skew, and sample rows."""
    global _stats_cache
    try:
        db = _get_db()
        try:
            sc = StatsCache(db)
            t_stats = sc.get_table_stats(config.BENCHMARK_TABLE)
            _stats_cache = sc

            col_details = db.execute(
                "SELECT COLUMN_NAME, COLUMN_TYPE, IS_NULLABLE, COLUMN_KEY, EXTRA "
                "FROM information_schema.COLUMNS "
                "WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s "
                "ORDER BY ORDINAL_POSITION",
                (db.database, config.BENCHMARK_TABLE),
            )

            schema_list: List[Dict[str, Any]] = []
            for cd in col_details:
                c_name = str(cd["COLUMN_NAME"]).lower()
                cs = t_stats.columns.get(c_name)
                schema_list.append(
                    {
                        "column_name": c_name,
                        "column_type": str(cd["COLUMN_TYPE"]),
                        "is_nullable": str(cd["IS_NULLABLE"]),
                        "column_key": str(cd["COLUMN_KEY"] or ""),
                        "extra": str(cd["EXTRA"] or ""),
                        "cardinality": cs.cardinality if cs else 0,
                        "distinct_ratio_pct": round((cs.distinct_ratio if cs else 0.0) * 100.0, 4),
                        "null_fraction_pct": round((cs.null_fraction if cs else 0.0) * 100.0, 2),
                        "byte_width": cs.byte_width if cs else 0,
                    }
                )

            sample_records = [
                _serialize_row(r)
                for r in db.execute(
                    f"SELECT * FROM `{config.BENCHMARK_TABLE}` LIMIT 12"
                )
            ]

            status_skew = [
                _serialize_row(r)
                for r in db.execute(
                    f"SELECT status, COUNT(*) AS row_count, "
                    f"ROUND(COUNT(*) * 100.0 / {max(1, t_stats.table_rows)}, 2) AS pct "
                    f"FROM `{config.BENCHMARK_TABLE}` GROUP BY status ORDER BY row_count DESC"
                )
            ]

            return {
                "table_name": config.BENCHMARK_TABLE,
                "total_rows": t_stats.table_rows,
                "data_length_mb": round(t_stats.data_length_mb, 2),
                "avg_row_length": t_stats.avg_row_length,
                "schema": schema_list,
                "sample_records": sample_records,
                "status_distribution": status_skew,
                "demo_queries": DEMO_QUERIES,
            }
        finally:
            db.close()
    except MySQLError as exc:
        raise HTTPException(status_code=500, detail=f"MySQL Error: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/step2/run-baseline")
def run_step2_baseline(req: RunQueryRequest) -> Dict[str, Any]:
    """Parse SQL query and run it with NO secondary indexes over 10 runs (1 warm-up discarded)."""
    try:
        parsed = parse_query(req.sql)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"SQL Parse Error: {exc}") from exc

    try:
        db = _get_db()
        try:
            if req.drop_secondary_indexes_first:
                db.drop_all_secondary_indexes(config.BENCHMARK_TABLE)
                db.analyze_table(config.BENCHMARK_TABLE)

            sec_indexes = db.get_non_primary_indexes(config.BENCHMARK_TABLE)
            measurement = _measure_query_detailed(
                db, req.sql, runs=req.runs, warmup=req.warmup
            )

            return {
                "parsed": {
                    "raw_sql": parsed.raw_sql,
                    "template_sql": parsed.template_sql,
                    "primary_table": parsed.primary_table,
                    "tables": parsed.tables,
                    "where_columns": parsed.where_columns,
                    "join_columns": parsed.join_columns,
                    "group_by_columns": parsed.group_by_columns,
                    "order_by_columns": parsed.order_by_columns,
                    "predicate_types": parsed.predicate_types,
                    "has_join": parsed.has_join,
                    "has_group_by": parsed.has_group_by,
                    "has_order_by": parsed.has_order_by,
                    "has_limit": parsed.has_limit,
                    "limit_value": parsed.limit_value,
                },
                "active_secondary_indexes": sec_indexes,
                "measurement": measurement,
            }
        finally:
            db.close()
    except MySQLError as exc:
        raise HTTPException(
            status_code=400, detail=f"MySQL Execution Error: {exc}"
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/step3/recommend")
def run_step3_recommend(req: RecommendRequest) -> Dict[str, Any]:
    """Score and rank candidate indexes using the trained XGBoost speedup regressor."""
    global _stats_cache
    try:
        db = _get_db()
        try:
            if _stats_cache is None:
                _stats_cache = StatsCache(db)
            else:
                _stats_cache.db = db

            predictor = _get_predictor()
            rec = recommend_index(
                req.sql,
                stats_cache=_stats_cache,
                predictor=predictor,
                threshold=req.threshold,
            )

            candidates_out: List[Dict[str, Any]] = []
            for rank, sc_cand in enumerate(rec.ranked_candidates, 1):
                rule_score = LegacyRuleBasedScorer.score_candidate_row(sc_cand.features)
                candidates_out.append(
                    {
                        "rank": rank,
                        "index_name": sc_cand.candidate.index_name,
                        "table": sc_cand.candidate.table,
                        "columns": list(sc_cand.candidate.columns),
                        "columns_display": sc_cand.candidate.columns_display,
                        "is_composite": sc_cand.candidate.is_composite,
                        "ddl": sc_cand.candidate.ddl,
                        "source_clauses": list(sc_cand.candidate.source_clauses),
                        "predicted_speedup": sc_cand.predicted_speedup,
                        "predicted_log_speedup": sc_cand.predicted_log_speedup,
                        "estimated_selectivity": sc_cand.estimated_selectivity,
                        "estimated_selectivity_pct": round(
                            sc_cand.estimated_selectivity * 100.0, 5
                        ),
                        "estimated_index_size_mb": sc_cand.estimated_index_size_mb,
                        "meets_threshold": sc_cand.predicted_speedup >= req.threshold,
                        "legacy_rule_score": round(rule_score, 2),
                        "shap_top_factors": sc_cand.shap_top_factors,
                    }
                )

            chosen_dict = None
            if rec.recommended is not None:
                chosen_dict = next(
                    (
                        c
                        for c in candidates_out
                        if c["index_name"] == rec.recommended.candidate.index_name
                    ),
                    candidates_out[0] if candidates_out else None,
                )

            return {
                "threshold": req.threshold,
                "recommend_create_index": rec.recommended is not None,
                "reason": rec.reason,
                "recommended_candidate": chosen_dict,
                "ranked_candidates": candidates_out,
                "top_fallback_candidate": candidates_out[0] if candidates_out else None,
            }
        finally:
            db.close()
    except MySQLError as exc:
        raise HTTPException(status_code=400, detail=f"MySQL Error: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/step4/create-index")
def run_step4_create_index(req: CreateIndexRequest) -> Dict[str, Any]:
    """Create the confirmed candidate index on live MySQL and run ANALYZE TABLE."""
    try:
        db = _get_db()
        try:
            # Drop if already present so the demo is idempotent
            db.drop_index_if_exists(req.table, req.index_name)
            t0 = time.perf_counter()
            db.execute_ddl(req.ddl)
            db.analyze_table(req.table)
            ddl_elapsed_s = time.perf_counter() - t0
            sec_indexes = db.get_non_primary_indexes(req.table)
            return {
                "created": True,
                "table": req.table,
                "index_name": req.index_name,
                "ddl": req.ddl,
                "ddl_time_s": round(ddl_elapsed_s, 3),
                "active_secondary_indexes": sec_indexes,
            }
        finally:
            db.close()
    except MySQLError as exc:
        raise HTTPException(
            status_code=400, detail=f"MySQL DDL Error creating index: {exc}"
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/step5/run-after")
def run_step5_after(req: RunQueryRequest) -> Dict[str, Any]:
    """Run the same query WITH the newly created index over 10 runs (1 warm-up discarded)."""
    try:
        db = _get_db()
        try:
            sec_indexes = db.get_non_primary_indexes(config.BENCHMARK_TABLE)
            measurement = _measure_query_detailed(
                db, req.sql, runs=req.runs, warmup=req.warmup
            )
            return {
                "active_secondary_indexes": sec_indexes,
                "measurement": measurement,
            }
        finally:
            db.close()
    except MySQLError as exc:
        raise HTTPException(
            status_code=400, detail=f"MySQL Execution Error: {exc}"
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/step6/compare-and-save")
def run_step6_compare(req: CompareRequest) -> Dict[str, Any]:
    """Compute Predicted vs Actual comparison, generate plain-English analysis, and save JSON."""
    t_bef = float(req.before_measurement.get("median_time_s") or 0.0)
    t_aft = float(req.after_measurement.get("median_time_s") or 0.0)
    r_bef = int(req.before_measurement.get("rows_examined") or 0)
    r_aft = int(req.after_measurement.get("rows_examined") or 0)
    acc_bef = str(req.before_measurement.get("access_type") or "ALL")
    acc_aft = str(req.after_measurement.get("access_type") or "ALL")
    key_aft = str(req.after_measurement.get("key_used") or "")

    actual_speedup = round(t_bef / max(t_aft, 1e-6), 2)
    pred_speedup = round(req.predicted_speedup, 2)
    error_pct = round(
        abs(pred_speedup - actual_speedup) / max(actual_speedup, 1e-6) * 100.0, 2
    )
    row_reduction_pct = round(
        (1.0 - (r_aft / max(1, r_bef))) * 100.0, 2
    )

    explanation = explain_prediction_vs_actual(
        predicted_speedup=pred_speedup,
        actual_speedup=actual_speedup,
        rows_before=r_bef,
        rows_after=r_aft,
        time_before_s=t_bef,
        time_after_s=t_aft,
        access_type_after=acc_aft,
        key_used_after=key_aft,
        estimated_selectivity=req.estimated_selectivity,
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    run_file = results_dir / f"run_{ts}.json"

    payload = {
        "timestamp": datetime.now().isoformat(),
        "sql": req.sql,
        "chosen_index_name": req.chosen_index_name,
        "chosen_columns": req.chosen_columns,
        "ddl_executed": req.ddl_executed,
        "threshold": req.threshold,
        "predicted_speedup": pred_speedup,
        "actual_speedup": actual_speedup,
        "error_pct": error_pct,
        "time_before_ms": round(t_bef * 1000.0, 3),
        "time_after_ms": round(t_aft * 1000.0, 3),
        "rows_before": r_bef,
        "rows_after": r_aft,
        "row_reduction_pct": row_reduction_pct,
        "access_type_before": acc_bef,
        "access_type_after": acc_aft,
        "key_used_after": key_aft,
        "explanation": explanation,
        "before_measurement": req.before_measurement,
        "after_measurement": req.after_measurement,
        "ranked_candidates": req.ranked_candidates,
        "saved_file": str(run_file.relative_to(Path(__file__).resolve().parent.parent)),
    }

    run_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


@app.post("/api/reset-demo")
def reset_demo() -> Dict[str, Any]:
    """Drop all secondary indexes on orders_benchmark and run ANALYZE TABLE."""
    try:
        db = _get_db()
        try:
            dropped = db.get_non_primary_indexes(config.BENCHMARK_TABLE)
            db.drop_all_secondary_indexes(config.BENCHMARK_TABLE)
            db.analyze_table(config.BENCHMARK_TABLE)
            remaining = db.get_non_primary_indexes(config.BENCHMARK_TABLE)
            return {
                "reset": True,
                "dropped_indexes": dropped,
                "active_secondary_indexes": remaining,
            }
        finally:
            db.close()
    except MySQLError as exc:
        raise HTTPException(status_code=500, detail=f"MySQL Error: {exc}") from exc


@app.get("/api/evaluation")
def get_evaluation_summary() -> Dict[str, Any]:
    """Return held-out template evaluation metrics, SHAP importances, and past demo runs."""
    pred = _get_predictor()
    meta = pred.metadata
    eval_data = evaluate_all()

    results_dir = Path(__file__).resolve().parent.parent / "results"
    past_runs: List[Dict[str, Any]] = []
    if results_dir.exists():
        for f in sorted(results_dir.glob("run_*.json"), reverse=True)[:15]:
            try:
                past_runs.append(json.loads(f.read_text(encoding="utf-8")))
            except Exception:
                pass

    return {
        "model_metadata": meta,
        "baseline_comparison": eval_data["summary"],
        "template_breakdown": eval_data["templates"],
        "past_runs": past_runs,
    }


@app.get("/api/shap-plot")
def get_shap_plot_image() -> FileResponse:
    """Serve the generated SHAP summary beeswarm PNG."""
    p = Path(config.SHAP_PLOT_PATH)
    if not p.exists():
        raise HTTPException(status_code=404, detail="SHAP summary plot not found.")
    return FileResponse(p, media_type="image/png")


# Serve built React + Vite frontend at / if frontend/dist exists
_DIST_DIR = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if _DIST_DIR.exists():
    app.mount("/", StaticFiles(directory=str(_DIST_DIR), html=True), name="frontend")
