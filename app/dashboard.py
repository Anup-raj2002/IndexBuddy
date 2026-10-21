"""
app/dashboard.py
────────────────
IndexBuddy live demo dashboard (Streamlit). Fully dynamic: works on ANY database
and table of the connected MySQL server. No table or column names are hardcoded.

Run:  streamlit run app/dashboard.py

Every number shown comes from the live database or the trained model.
"""
from __future__ import annotations

import io
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Make project imports work when launched as `streamlit run app/dashboard.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from db.mysql_adapter import MySQLAdapter  # noqa: E402
from features.extract import StatsCache  # noqa: E402
from models.predict import SpeedupPredictor  # noqa: E402
from parsing.query_parser import validate_select_only  # noqa: E402
from selection.recommend import (  # noqa: E402
    explain_prediction_vs_actual,
    generate_sample_queries_for_table,
    recommend_index,
)

st.set_page_config(page_title="IndexBuddy", page_icon="🗂️", layout="wide")

STEP_TITLES = {
    1: "Table schema and records",
    2: "Run the query WITHOUT an index",
    3: "ML index recommendation",
    4: "Generated DDL and confirmation",
    5: "Run the same query WITH the index",
    6: "Predicted vs actual comparison",
}
LARGE_TABLE_ROWS = 2_000_000
LARGE_INDEX_MB = 500.0


def _stretch(fn):
    """Full-width widgets across Streamlit versions (new `width` API, old `use_container_width`)."""
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, width="stretch", **kwargs)
        except TypeError:
            return fn(*args, use_container_width=True, **kwargs)
    return wrapper


_dataframe = _stretch(st.dataframe)
_plotly_chart = _stretch(st.plotly_chart)


# ─────────────────────────── helpers ───────────────────────────
def _ss(key: str, default: Any = None) -> Any:
    return st.session_state.get(key, default)


def _reset_demo_state(keep_connection: bool = True) -> None:
    for k in ("step", "sql", "baseline", "after", "rec", "created_index", "saved_path"):
        st.session_state.pop(k, None)
    st.session_state["step"] = 1


def _safe_df(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """DataFrame that Streamlit/Arrow can always render (Decimal, bytes, dates)."""
    return pd.DataFrame(rows).astype(str)


def _get_db() -> Optional[MySQLAdapter]:
    db: Optional[MySQLAdapter] = _ss("db")
    if db is None:
        return None
    try:
        if not db.is_connected():
            db.connect()
            if _ss("database"):
                db.use_database(_ss("database"))
    except Exception:
        return None
    return db


def _measure(db: MySQLAdapter, sql: str) -> Dict[str, Any]:
    """Run the query live: median time, rows examined, EXPLAIN, EXPLAIN ANALYZE."""
    median_s, rows_examined, times = db.timed_execute(
        sql, runs=config.DASHBOARD_RUNS, warmup=1
    )
    plan = db.explain(sql)
    try:
        analyze = db.explain_analyze(sql)
    except Exception as exc:  # needs MySQL 8.0.18+
        analyze = f"EXPLAIN ANALYZE unavailable: {exc}"
    return {
        "median_s": median_s,
        "rows_examined": int(rows_examined),
        "runs_s": times,
        "explain": plan,
        "explain_analyze": analyze,
    }


def _plan_row(plan: List[Dict[str, Any]], table: str) -> Dict[str, Any]:
    for r in plan:
        if str(r.get("table") or "").lower() == table.lower():
            return r
    return plan[0] if plan else {}


def _parse_uploaded_queries(name: str, data: bytes) -> List[str]:
    text = data.decode("utf-8", errors="replace")
    if name.lower().endswith(".csv"):
        df = pd.read_csv(io.StringIO(text))
        col = "sql" if "sql" in df.columns else df.columns[0]
        return [str(x).strip() for x in df[col].dropna().tolist() if str(x).strip()]
    return [q.strip() for q in text.split(";") if q.strip()]


# ─────────────────────────── sidebar ───────────────────────────
def sidebar() -> None:
    st.sidebar.title("🗂️ IndexBuddy")
    st.sidebar.caption("ML-based MySQL index recommender")
    st.sidebar.subheader("1. Connect")
    host = st.sidebar.text_input("Host", config.MYSQL_HOST)
    port = st.sidebar.number_input("Port", value=int(config.MYSQL_PORT), step=1)
    user = st.sidebar.text_input("User", config.MYSQL_USER)
    password = st.sidebar.text_input("Password", config.MYSQL_PASSWORD, type="password")

    if st.sidebar.button("Connect", type="primary"):
        try:
            db = MySQLAdapter(host=host, port=int(port), user=user, password=password)
            db.connect()
            st.session_state["db"] = db
            st.session_state["databases"] = db.list_databases()
            _reset_demo_state()
            st.sidebar.success("Connected")
        except Exception as exc:
            st.session_state.pop("db", None)
            st.sidebar.error(f"Connection failed: {exc}")

    db = _get_db()
    if db is None:
        return

    st.sidebar.subheader("2. Choose database and table")
    dbs: List[str] = _ss("databases", [])
    database = st.sidebar.selectbox("Database", dbs, key="sel_database")
    if not database:
        return
    db.use_database(database)
    tables = db.list_tables(database)
    names = [t["table_name"] for t in tables]
    if not names:
        st.sidebar.warning("No base tables in this database.")
        return
    table = st.sidebar.selectbox("Table", names, key="sel_table")

    key = f"{database}.{table}"
    if _ss("sel_key") != key:  # selection changed -> fresh demo + fresh stats cache
        st.session_state["sel_key"] = key
        st.session_state["database"] = database
        st.session_state["table"] = table
        st.session_state["stats_cache"] = StatsCache(db)
        st.session_state["total_rows"] = db.row_count(table)
        _reset_demo_state()

    st.sidebar.divider()
    st.sidebar.number_input(
        "Recommendation threshold (x speedup)",
        min_value=1.0, max_value=50.0, value=float(config.SPEEDUP_THRESHOLD),
        step=0.1, key="threshold",
    )
    if st.sidebar.button("🔄 Reset demo (drop IndexBuddy index)"):
        try:
            dropped = db.drop_indexbuddy_indexes(table)
            st.sidebar.success(f"Dropped: {dropped or 'nothing to drop'}")
        except Exception as exc:
            st.sidebar.error(f"Reset failed: {exc}")
        _reset_demo_state()
        st.rerun()


# ─────────────────────────── status bar ───────────────────────────
def status_bar() -> None:
    db = _get_db()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MySQL", "Connected ✅" if db else "Not connected ❌")
    c2.metric("Database", _ss("database", "–"))
    c3.metric("Table", _ss("table", "–"))
    total = _ss("total_rows")
    c4.metric("Total rows", f"{total:,}" if total is not None else "–")


# ─────────────────────────── steps ───────────────────────────
def next_button(step: int, enabled: bool = True, label: str = "Next ➡️") -> None:
    if _ss("step", 1) == step:
        if st.button(label, key=f"next_{step}", disabled=not enabled):
            st.session_state["step"] = step + 1
            st.rerun()


def step1(db: MySQLAdapter, table: str) -> None:
    cache: StatsCache = _ss("stats_cache")
    t_stats = cache.get_table_stats(table)
    st.caption(
        f"{t_stats.table_rows:,} rows (catalog estimate) · primary key: "
        f"{', '.join(t_stats.primary_key_columns) or 'none'}"
    )
    cols = []
    for cs in t_stats.columns.values():
        d = asdict(cs)
        d.pop("value_frequencies", None)
        d.pop("sample_values", None)
        cols.append(d)
    st.subheader("Columns and live statistics")
    _dataframe(_safe_df(cols))

    idx = db.get_existing_indexes(table)
    st.subheader("Existing indexes")
    _dataframe(_safe_df(idx)) if idx else st.info("No indexes.")

    st.subheader("Sample records")
    rows = db.execute(f"SELECT * FROM `{table}` LIMIT 20")
    _dataframe(_safe_df(rows))
    next_button(1)


def step2(db: MySQLAdapter, table: str) -> None:
    cache: StatsCache = _ss("stats_cache")
    leftovers = db.get_indexbuddy_indexes(table)
    if leftovers and not _ss("created_index"):
        st.warning(f"IndexBuddy indexes already exist on `{table}`: {leftovers}. "
                   "Use 'Reset demo' in the sidebar for a clean baseline.")

    source = st.radio("Query source", ["Generated sample query", "Type or paste SQL",
                                       "Upload .sql / .csv"], horizontal=True)
    sql = ""
    if source == "Generated sample query":
        t_stats = cache.get_table_stats(table)
        samples = generate_sample_queries_for_table(t_stats, db.get_foreign_keys())
        if not samples:
            st.info("No sample queries could be generated for this table.")
        else:
            pick = st.selectbox("Sample query", range(len(samples)),
                                format_func=lambda i: samples[i]["title"])
            st.caption(samples[pick].get("description", ""))
            sql = st.text_area("SQL (editable)", samples[pick]["sql"], height=140,
                               key=f"sample_{pick}_{table}")
    elif source == "Type or paste SQL":
        sql = st.text_area("SQL", "", height=140,
                           placeholder="SELECT ... FROM ... WHERE ...")
    else:
        up = st.file_uploader("Upload .sql (semicolon separated) or .csv (column 'sql')",
                              type=["sql", "csv"])
        if up is not None:
            qs = _parse_uploaded_queries(up.name, up.getvalue())
            if qs:
                i = st.selectbox("Query", range(len(qs)), format_func=lambda i: qs[i][:90])
                sql = st.text_area("SQL (editable)", qs[i], height=140, key=f"up_{i}")

    if st.button("▶ Run without index", type="primary", disabled=not sql.strip()):
        try:
            validate_select_only(sql)
        except Exception as exc:
            st.error(f"Only SELECT queries are allowed: {exc}")
            return
        try:
            with st.spinner(f"Running {config.DASHBOARD_RUNS} times (first run discarded)..."):
                st.session_state["baseline"] = _measure(db, sql)
            st.session_state["sql"] = sql
            for k in ("after", "rec", "created_index"):
                st.session_state.pop(k, None)
        except Exception as exc:
            st.error(f"Query failed: {exc}")
            return

    base = _ss("baseline")
    if base:
        c1, c2 = st.columns(2)
        c1.metric("Median time (no index)", f"{base['median_s']*1000:.2f} ms")
        c2.metric("Rows examined (EXPLAIN)", f"{base['rows_examined']:,}")
        st.markdown("**EXPLAIN**")
        _dataframe(_safe_df(base["explain"]))
        st.markdown("**EXPLAIN ANALYZE**")
        st.code(base["explain_analyze"])
    next_button(2, enabled=base is not None)


def step3(db: MySQLAdapter, table: str) -> None:
    if st.button("🤖 Get recommendation from IndexBuddy", type="primary"):
        try:
            with st.spinner("Scoring candidates with the trained model..."):
                st.session_state["rec"] = recommend_index(
                    _ss("sql"), _ss("stats_cache"), target_table=table,
                    predictor=SpeedupPredictor(), threshold=float(_ss("threshold")),
                )
        except FileNotFoundError as exc:
            st.error(f"{exc}")
            return
        except Exception as exc:
            st.error(f"Recommendation failed: {exc}")
            return

    rec = _ss("rec")
    if rec:
        if rec.ranked_candidates:
            _dataframe(pd.DataFrame([{
                "Candidate columns": s.candidate.columns_display,
                "Predicted speedup (x)": s.predicted_speedup,
                "Est. selectivity": f"{s.estimated_selectivity*100:.4f}%",
                "Est. index size (MB)": round(s.estimated_index_size_mb, 2),
            } for s in rec.ranked_candidates]))
        (st.success if rec.recommended else st.info)(rec.reason)
        if rec.recommended and rec.recommended.shap_top_factors:
            with st.expander("Why? Top model factors (SHAP)"):
                _dataframe(_safe_df(rec.recommended.shap_top_factors))
    next_button(3, enabled=bool(rec and rec.recommended),
                label="Next ➡️ (needs a recommended index)")


def step4(db: MySQLAdapter, table: str) -> None:
    rec = _ss("rec")
    cand = rec.recommended
    st.markdown("**Generated DDL**")
    st.code(cand.candidate.ddl, language="sql")
    st.caption(f"Estimated index size: ~{cand.estimated_index_size_mb:.2f} MB · "
               f"predicted speedup {cand.predicted_speedup:.2f}x")
    total = _ss("total_rows", 0)
    if total > LARGE_TABLE_ROWS or cand.estimated_index_size_mb > LARGE_INDEX_MB:
        st.warning("Large table: building this index may take a while and use disk space.")

    if _ss("created_index"):
        st.success(f"Index `{_ss('created_index')}` created.")
    else:
        if st.button("✅ Confirm and create index", type="primary"):
            try:
                with st.spinner("Creating index..."):
                    db.execute_ddl(cand.candidate.ddl)
                st.session_state["created_index"] = cand.candidate.index_name
                st.rerun()
            except Exception as exc:
                st.error(f"Index creation failed: {exc}")
    next_button(4, enabled=bool(_ss("created_index")))


def step5(db: MySQLAdapter, table: str) -> None:
    if st.button("▶ Run same query with index", type="primary"):
        try:
            with st.spinner(f"Running {config.DASHBOARD_RUNS} times (first run discarded)..."):
                st.session_state["after"] = _measure(db, _ss("sql"))
        except Exception as exc:
            st.error(f"Query failed: {exc}")
            return
    after = _ss("after")
    if after:
        base = _ss("baseline")
        c1, c2 = st.columns(2)
        c1.metric("Median time (with index)", f"{after['median_s']*1000:.2f} ms",
                  delta=f"{(after['median_s']-base['median_s'])*1000:.2f} ms",
                  delta_color="inverse")
        c2.metric("Rows examined (EXPLAIN)", f"{after['rows_examined']:,}",
                  delta=f"{after['rows_examined']-base['rows_examined']:,}",
                  delta_color="inverse")
        st.markdown("**EXPLAIN**")
        _dataframe(_safe_df(after["explain"]))
        st.markdown("**EXPLAIN ANALYZE**")
        st.code(after["explain_analyze"])
    next_button(5, enabled=after is not None)


def step6(db: MySQLAdapter, table: str) -> None:
    base, after, rec = _ss("baseline"), _ss("after"), _ss("rec")
    cand = rec.recommended
    actual = base["median_s"] / max(after["median_s"], 1e-9)
    predicted = cand.predicted_speedup
    err_pct = abs(predicted - actual) / max(actual, 1e-9) * 100.0

    _dataframe(pd.DataFrame([{
        "Metric": m, "Before (no index)": b, "After (index)": a,
    } for m, b, a in [
        ("Median time (ms)", f"{base['median_s']*1000:.2f}", f"{after['median_s']*1000:.2f}"),
        ("Rows examined", f"{base['rows_examined']:,}", f"{after['rows_examined']:,}"),
    ]]), hide_index=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("ML predicted speedup", f"{predicted:.2f}x")
    c2.metric("Actual speedup", f"{actual:.2f}x")
    c3.metric("Prediction error", f"{err_pct:.1f}%")

    fig = go.Figure(go.Bar(x=["Predicted", "Actual"], y=[predicted, actual],
                           text=[f"{predicted:.2f}x", f"{actual:.2f}x"],
                           textposition="auto"))
    fig.update_layout(title="Predicted vs actual speedup", yaxis_title="speedup (x)",
                      height=340)
    _plotly_chart(fig)

    prow = _plan_row(after["explain"], table)
    explanation = explain_prediction_vs_actual(
        predicted_speedup=predicted, actual_speedup=actual,
        rows_before=base["rows_examined"], rows_after=after["rows_examined"],
        time_before_s=base["median_s"], time_after_s=after["median_s"],
        access_type_after=str(prow.get("type") or ""),
        key_used_after=str(prow.get("key") or ""),
        estimated_selectivity=cand.estimated_selectivity,
    )
    st.markdown("### Why the prediction and reality differ")
    st.markdown(explanation)

    if not _ss("saved_path"):
        config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        path = config.RESULTS_DIR / f"run_{datetime.now():%Y%m%d_%H%M%S}.json"
        path.write_text(json.dumps({
            "database": _ss("database"), "table": table, "sql": _ss("sql"),
            "ddl": cand.candidate.ddl, "predicted_speedup": predicted,
            "actual_speedup": actual, "error_pct": err_pct,
            "time_before_ms": base["median_s"] * 1000, "time_after_ms": after["median_s"] * 1000,
            "rows_before": base["rows_examined"], "rows_after": after["rows_examined"],
            "explain_before": base["explain"], "explain_after": after["explain"],
            "explanation": explanation,
        }, default=str, indent=2))
        st.session_state["saved_path"] = str(path)
    st.caption(f"Run saved to {_ss('saved_path')}")


# ─────────────────────────── main ───────────────────────────
def main() -> None:
    sidebar()
    st.title("IndexBuddy: ML-based automatic index recommendation")
    status_bar()
    db = _get_db()
    if db is None or not _ss("table"):
        st.info("👈 Connect to MySQL and choose a database and table in the sidebar.")
        return

    table = _ss("table")
    step = _ss("step", 1)
    runners = {1: step1, 2: step2, 3: step3, 4: step4, 5: step5, 6: step6}
    for i in range(1, min(step, 6) + 1):
        st.divider()
        st.header(f"Step {i}: {STEP_TITLES[i]}")
        try:
            runners[i](db, table)
        except Exception as exc:
            st.error(f"Step {i} failed: {exc}")


main()