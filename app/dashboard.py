"""
app/dashboard.py
────────────────
Step-by-step live MySQL 8.x & XGBoost Index Recommender Dashboard (Streamlit).

Every number displayed (row counts, cardinalities, EXPLAIN plans, EXPLAIN ANALYZE
trees, 10-run median latencies, XGBoost predicted speedups, and SHAP attributions)
is queried live from MySQL 8.x or computed by the trained XGBoost model.

Run with:
    streamlit run app/dashboard.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
from app.api_server import (
    DEMO_QUERIES,
    CompareRequest,
    CreateIndexRequest,
    RecommendRequest,
    RunQueryRequest,
    get_db_status,
    get_evaluation_summary,
    get_step1_schema,
    reset_demo,
    run_step2_baseline,
    run_step3_recommend,
    run_step4_create_index,
    run_step5_after,
    run_step6_compare,
)


def main() -> None:
    st.set_page_config(
        page_title="IndexBuddy | ML MySQL Index Recommender",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    if "step" not in st.session_state:
        st.session_state.step = 1
    if "selected_demo_idx" not in st.session_state:
        st.session_state.selected_demo_idx = 0
    if "sql_query" not in st.session_state:
        st.session_state.sql_query = DEMO_QUERIES[0]["sql"]
    if "threshold" not in st.session_state:
        st.session_state.threshold = float(config.SPEEDUP_THRESHOLD)
    if "step2_data" not in st.session_state:
        st.session_state.step2_data = None
    if "step3_data" not in st.session_state:
        st.session_state.step3_data = None
    if "step4_data" not in st.session_state:
        st.session_state.step4_data = None
    if "step5_data" not in st.session_state:
        st.session_state.step5_data = None
    if "step6_data" not in st.session_state:
        st.session_state.step6_data = None

    # ── Top Persistent Database Status Bar ──────────────────────
    status = get_db_status()
    top_c1, top_c2, top_c3, top_c4, top_c5 = st.columns([2.2, 1.5, 1.5, 1.8, 1.2])
    with top_c1:
        if status.get("connected"):
            st.success(
                f"MySQL {status.get('mysql_version')} Connected "
                f"({status.get('host')}:{status.get('port')}/{status.get('database')})"
            )
        else:
            st.error(f"MySQL Disconnected: {status.get('error', 'Unknown error')}")
    with top_c2:
        st.metric("Benchmark Table", f"`{status.get('table_name')}`")
    with top_c3:
        st.metric("Total Rows", f"{status.get('total_rows', 0):,}")
    with top_c4:
        sec_idx = status.get("secondary_indexes", [])
        st.metric(
            "Active Secondary Indexes",
            f"{len(sec_idx)} ({', '.join(sec_idx) if sec_idx else 'None'})",
        )
    with top_c5:
        if st.button("Reset Demo (Drop Indexes)", type="secondary", use_container_width=True):
            try:
                res = reset_demo()
                st.session_state.step2_data = None
                st.session_state.step3_data = None
                st.session_state.step4_data = None
                st.session_state.step5_data = None
                st.session_state.step6_data = None
                st.session_state.step = 1
                st.toast(f"Dropped secondary indexes: {res.get('dropped_indexes')}")
                st.rerun()
            except Exception as exc:
                st.error(f"Failed to reset demo: {exc}")

    st.divider()

    # ── Step Progress Bar ───────────────────────────────────────
    step_titles = [
        "1. Schema & Sample",
        "2. Baseline (No Index)",
        "3. ML Recommendation",
        "4. Confirm CREATE INDEX",
        "5. Post-Index Run",
        "6. Predicted vs Actual",
    ]
    nav_cols = st.columns(6)
    for idx, title in enumerate(step_titles, 1):
        with nav_cols[idx - 1]:
            btn_type = "primary" if st.session_state.step == idx else "secondary"
            if st.button(title, key=f"nav_{idx}", type=btn_type, use_container_width=True):
                st.session_state.step = idx
                st.rerun()

    st.markdown("---")
    step = st.session_state.step

    # ── STEP 1: Schema & Sample ─────────────────────────────────
    if step == 1:
        st.header("Step 1: `orders_benchmark` Live Schema, Cardinalities & Sample Records")
        try:
            s1 = get_step1_schema()
            c_left, c_right = st.columns([1.4, 1.0])
            with c_left:
                st.subheader("Live Column Metadata & `COUNT(DISTINCT)` Cardinality")
                st.dataframe(pd.DataFrame(s1["schema"]), use_container_width=True)
            with c_right:
                st.subheader("`status` Column Skew Distribution (Live)")
                skew_df = pd.DataFrame(s1["status_distribution"])
                st.dataframe(skew_df, use_container_width=True)

            st.subheader("Sample Records from `orders_benchmark` (Live)")
            st.dataframe(pd.DataFrame(s1["sample_records"]), use_container_width=True)
        except Exception as exc:
            st.error(f"Error loading schema: {exc}")

        if st.button("Next: Step 2 (Choose & Run Demo Query) ->", type="primary"):
            st.session_state.step = 2
            st.rerun()

    # ── STEP 2: Baseline Query Execution (No Index) ─────────────
    elif step == 2:
        st.header("Step 2: Choose Held-Out Demo Query & Run with NO Secondary Index")
        labels = [d["label"] for d in DEMO_QUERIES]
        chosen_idx = st.selectbox(
            "Select Held-Out Demo Query Template (Never Seen in Training):",
            options=list(range(len(labels))),
            format_func=lambda i: labels[i],
            index=st.session_state.selected_demo_idx,
        )
        if chosen_idx != st.session_state.selected_demo_idx:
            st.session_state.selected_demo_idx = chosen_idx
            st.session_state.sql_query = DEMO_QUERIES[chosen_idx]["sql"]

        st.caption(DEMO_QUERIES[chosen_idx]["description"])
        sql_text = st.text_area("Editable SQL Query:", value=st.session_state.sql_query, height=100)
        st.session_state.sql_query = sql_text

        if st.button("Run Baseline Query (10 Runs, 1 Warm-Up Discarded, No Index)", type="primary"):
            with st.spinner("Dropping any secondary indexes and executing 10 live runs on MySQL..."):
                try:
                    res2 = run_step2_baseline(
                        RunQueryRequest(
                            sql=sql_text,
                            runs=config.DASHBOARD_RUNS,
                            warmup=1,
                            drop_secondary_indexes_first=True,
                        )
                    )
                    st.session_state.step2_data = res2
                except Exception as exc:
                    st.error(f"Error running baseline query: {exc}")

        if st.session_state.step2_data:
            m = st.session_state.step2_data["measurement"]
            p = st.session_state.step2_data["parsed"]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Median Time (10 runs, 1 warmup)", f"{m['median_time_ms']:.2f} ms")
            m2.metric("Rows Examined (EXPLAIN)", f"{m['rows_examined']:,}")
            m3.metric("Access Type", f"`{m['access_type']}`")
            m4.metric("Key Used", f"`{m['key_used'] or 'NONE'}`")

            st.subheader("Parsed AST Metadata (`sqlglot` MySQL dialect)")
            st.json(p)

            st.subheader("Live `EXPLAIN` Output (No Index)")
            st.dataframe(pd.DataFrame(m["explain_rows"]), use_container_width=True)

            st.subheader("Live `EXPLAIN ANALYZE` Tree (No Index)")
            st.code(m["explain_analyze"], language="text")

            if st.button("Next: Step 3 (Get ML Index Recommendation) ->", type="primary"):
                st.session_state.step = 3
                st.rerun()

    # ── STEP 3: ML Recommendation ───────────────────────────────
    elif step == 3:
        st.header("Step 3: XGBoost Candidate Scoring & Index Recommendation")
        st.code(st.session_state.sql_query, language="sql")
        thresh = st.slider(
            "Minimum Predicted Speedup Threshold (x):",
            min_value=1.1,
            max_value=5.0,
            value=float(st.session_state.threshold),
            step=0.1,
        )
        st.session_state.threshold = thresh

        if st.button("Get Recommendation (Score Candidates with XGBoost)", type="primary"):
            with st.spinner("Extracting live catalog/EXPLAIN features & scoring with XGBoost..."):
                try:
                    res3 = run_step3_recommend(
                        RecommendRequest(sql=st.session_state.sql_query, threshold=thresh)
                    )
                    st.session_state.step3_data = res3
                except Exception as exc:
                    st.error(f"Recommendation error: {exc}")

        if st.session_state.step3_data:
            s3 = st.session_state.step3_data
            if s3["recommend_create_index"]:
                st.success(s3["reason"])
            else:
                st.warning(s3["reason"])

            cands_df = pd.DataFrame(s3["ranked_candidates"])
            if not cands_df.empty:
                display_cols = [
                    "rank",
                    "columns_display",
                    "predicted_speedup",
                    "predicted_log_speedup",
                    "estimated_selectivity_pct",
                    "estimated_index_size_mb",
                    "meets_threshold",
                    "ddl",
                ]
                st.subheader("Ranked Candidate Indexes (Scored by XGBoost)")
                st.dataframe(cands_df[display_cols], use_container_width=True)

            if st.button("Next: Step 4 (Review & Confirm DDL) ->", type="primary"):
                st.session_state.step = 4
                st.rerun()

    # ── STEP 4: Confirm CREATE INDEX ────────────────────────────
    elif step == 4:
        st.header("Step 4: Confirm `CREATE INDEX` DDL Execution")
        s3 = st.session_state.step3_data
        if not s3:
            st.info("Please run Step 3 first to generate a candidate recommendation.")
        else:
            target_cand = s3.get("recommended_candidate") or s3.get("top_fallback_candidate")
            if s3.get("recommend_create_index") and target_cand:
                st.success(
                    f"Recommended Candidate: `{target_cand['index_name']}` "
                    f"(Predicted Speedup: **{target_cand['predicted_speedup']:.2f}x**)"
                )
                st.code(target_cand["ddl"], language="sql")
                if st.button("Confirm & Create Index on Live MySQL", type="primary"):
                    with st.spinner("Executing CREATE INDEX and ANALYZE TABLE on MySQL..."):
                        try:
                            res4 = run_step4_create_index(
                                CreateIndexRequest(
                                    table=target_cand["table"],
                                    index_name=target_cand["index_name"],
                                    ddl=target_cand["ddl"],
                                )
                            )
                            st.session_state.step4_data = {
                                **res4,
                                "candidate": target_cand,
                            }
                        except Exception as exc:
                            st.error(f"MySQL DDL Error: {exc}")
            else:
                st.warning(s3["reason"])
                st.info(
                    "The ML model recommends creating NO index because predicted speedup "
                    f"is below the {s3['threshold']:.2f}x threshold. For demonstration purposes, "
                    "you can optionally force-create the top candidate below to verify empirically "
                    "on live MySQL that the index does not meaningfully speed up the query."
                )
                if target_cand:
                    st.code(target_cand["ddl"], language="sql")
                    if st.button("Force-Create Index Anyway (Verify Low-Selectivity Case)", type="secondary"):
                        with st.spinner("Executing CREATE INDEX and ANALYZE TABLE on MySQL..."):
                            res4 = run_step4_create_index(
                                CreateIndexRequest(
                                    table=target_cand["table"],
                                    index_name=target_cand["index_name"],
                                    ddl=target_cand["ddl"],
                                )
                            )
                            st.session_state.step4_data = {
                                **res4,
                                "candidate": target_cand,
                            }

            if st.session_state.step4_data:
                s4 = st.session_state.step4_data
                st.success(
                    f"Index `{s4['index_name']}` created on `{s4['table']}` in {s4['ddl_time_s']:.2f}s!"
                )
                if st.button("Next: Step 5 (Re-Run Query With Index) ->", type="primary"):
                    st.session_state.step = 5
                    st.rerun()

    # ── STEP 5: Run Query Again With Index ──────────────────────
    elif step == 5:
        st.header("Step 5: Re-Run Query WITH Created Index (10 Runs, 1 Warm-Up Discarded)")
        st.code(st.session_state.sql_query, language="sql")
        if st.button("Run Query With Index (10 Runs)", type="primary"):
            with st.spinner("Running query 10 times with the new index on MySQL..."):
                try:
                    res5 = run_step5_after(
                        RunQueryRequest(
                            sql=st.session_state.sql_query,
                            runs=config.DASHBOARD_RUNS,
                            warmup=1,
                        )
                    )
                    st.session_state.step5_data = res5
                except Exception as exc:
                    st.error(f"Execution error: {exc}")

        if st.session_state.step5_data:
            m = st.session_state.step5_data["measurement"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Median Time After (10 runs)", f"{m['median_time_ms']:.2f} ms")
            c2.metric("Rows Examined After (EXPLAIN)", f"{m['rows_examined']:,}")
            c3.metric("Access Type After", f"`{m['access_type']}`")
            c4.metric("Key Used After", f"`{m['key_used'] or 'NONE'}`")

            st.subheader("New `EXPLAIN` Output (With Index)")
            st.dataframe(pd.DataFrame(m["explain_rows"]), use_container_width=True)

            st.subheader("New `EXPLAIN ANALYZE` Tree (With Index)")
            st.code(m["explain_analyze"], language="text")

            if st.button("Next: Step 6 (Predicted vs Actual Comparison) ->", type="primary"):
                st.session_state.step = 6
                st.rerun()

    # ── STEP 6: Comparison Table, Bar Chart & Diagnostic ────────
    elif step == 6:
        st.header("Step 6: Predicted vs. Actual Speedup Comparison & Diagnostic Analysis")
        s2 = st.session_state.step2_data
        s3 = st.session_state.step3_data
        s4 = st.session_state.step4_data
        s5 = st.session_state.step5_data

        if not (s2 and s3 and s5):
            st.warning("Please complete Steps 2, 3, and 5 first.")
        else:
            cand = (s4 or {}).get("candidate") or s3.get("recommended_candidate") or s3.get("top_fallback_candidate")
            if st.session_state.step6_data is None:
                comp = run_step6_compare(
                    CompareRequest(
                        sql=st.session_state.sql_query,
                        chosen_index_name=cand["index_name"] if cand else None,
                        chosen_columns=cand["columns_display"] if cand else None,
                        ddl_executed=cand["ddl"] if cand else None,
                        predicted_speedup=float(cand["predicted_speedup"]) if cand else 1.0,
                        estimated_selectivity=float(cand["estimated_selectivity"]) if cand else 1.0,
                        threshold=float(st.session_state.threshold),
                        before_measurement=s2["measurement"],
                        after_measurement=s5["measurement"],
                        ranked_candidates=s3.get("ranked_candidates", []),
                    )
                )
                st.session_state.step6_data = comp

            comp = st.session_state.step6_data
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Predicted Speedup (XGBoost)", f"{comp['predicted_speedup']:.2f}x")
            k2.metric("Actual Measured Speedup", f"{comp['actual_speedup']:.2f}x")
            k3.metric("Relative Difference (%)", f"{comp['error_pct']:.1f}%")
            k4.metric("Rows Examined Reduction", f"{comp['row_reduction_pct']:.2f}%")

            comp_table = pd.DataFrame(
                [
                    {
                        "Metric": "Median Execution Time (ms)",
                        "Before Index": f"{comp['time_before_ms']:.3f} ms",
                        "After Index": f"{comp['time_after_ms']:.3f} ms",
                        "Improvement / Ratio": f"{comp['actual_speedup']:.2f}x speedup",
                    },
                    {
                        "Metric": "EXPLAIN Estimated Rows Examined",
                        "Before Index": f"{comp['rows_before']:,}",
                        "After Index": f"{comp['rows_after']:,}",
                        "Improvement / Ratio": f"-{comp['row_reduction_pct']:.2f}% rows",
                    },
                    {
                        "Metric": "MySQL Access Type (`type`)",
                        "Before Index": comp["access_type_before"],
                        "After Index": f"{comp['access_type_after']} (key: {comp['key_used_after'] or 'NULL'})",
                        "Improvement / Ratio": f"Predicted: {comp['predicted_speedup']:.2f}x (err: {comp['error_pct']:.1f}%)",
                    },
                ]
            )
            st.subheader("Before vs. After Summary Table")
            st.table(comp_table)

            ch1, ch2 = st.columns(2)
            with ch1:
                fig_time = go.Figure(
                    data=[
                        go.Bar(
                            x=["Before Index (ms)", "After Index (ms)"],
                            y=[comp["time_before_ms"], comp["time_after_ms"]],
                            marker_color=["#ef4444", "#10b981"],
                            text=[
                                f"{comp['time_before_ms']:.2f} ms",
                                f"{comp['time_after_ms']:.2f} ms",
                            ],
                            textposition="auto",
                        )
                    ]
                )
                fig_time.update_layout(
                    title="Median Execution Time: Before vs. After (ms)",
                    yaxis_title="Milliseconds (lower is better)",
                )
                st.plotly_chart(fig_time, use_container_width=True)

            with ch2:
                fig_spd = go.Figure(
                    data=[
                        go.Bar(
                            x=["XGBoost Predicted Speedup", "Actual Live Speedup"],
                            y=[comp["predicted_speedup"], comp["actual_speedup"]],
                            marker_color=["#6366f1", "#10b981"],
                            text=[
                                f"{comp['predicted_speedup']:.2f}x",
                                f"{comp['actual_speedup']:.2f}x",
                            ],
                            textposition="auto",
                        )
                    ]
                )
                fig_spd.update_layout(
                    title="Predicted vs. Actual Speedup Ratio (x)",
                    yaxis_title="Speedup Factor (x)",
                )
                st.plotly_chart(fig_spd, use_container_width=True)

            st.subheader("Plain-English Diagnostic Analysis")
            st.info(comp["explanation"])
            st.success(f"Run log automatically saved to `{comp['saved_file']}`")


if __name__ == "__main__":
    main()
