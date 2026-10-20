import os
import sys

# Add project root directory to Python path for Streamlit module imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

import streamlit as st
import pandas as pd
import duckdb

from src.sql_parser import parse_sql_query
from src.recommender import evaluate_index_viability
from src.utils import universal_read_csv


def safe_read_csv(uploaded_file):
    """Safely reads an uploaded file object by resetting the stream pointer."""
    if hasattr(uploaded_file, "seek"):
        uploaded_file.seek(0)
    return universal_read_csv(uploaded_file)


# Safe import with inline fallback to prevent ImportError if src/metadata_extractor.py lacks helper functions
try:
    from src.metadata_extractor import (
        extract_dataset_metadata, 
        resolve_table_row_count, 
        estimate_matching_rows_from_df
    )
except ImportError:
    from src.metadata_extractor import extract_dataset_metadata

    def resolve_table_row_count(target_table, dataset_file=None):
        """Resolves total row count from uploaded dataset or defaults to 50,000."""
        if dataset_file is not None:
            try:
                meta_df = extract_dataset_metadata(dataset_file)
                table_row_map = dict(zip(meta_df['table_name'].astype(str).str.lower(), meta_df['total_rows']))
                if target_table and target_table.lower() in table_row_map:
                    return int(table_row_map[target_table.lower()])
                elif not meta_df.empty and 'total_rows' in meta_df.columns:
                    return int(meta_df['total_rows'].iloc[0])
            except Exception:
                pass
        return 50000

    def estimate_matching_rows_from_df(dataset_file, sql_query):
        """Measures matching row count directly using in-memory DuckDB."""
        if not dataset_file or not sql_query or not sql_query.strip():
            return None
        try:
            raw_df = safe_read_csv(dataset_file)
            if raw_df.empty:
                return None
            
            cols_lower = [str(c).lower() for c in raw_df.columns]
            if 'table_name' in cols_lower and 'total_rows' in cols_lower:
                return None

            raw_name = getattr(dataset_file, 'name', 'data')
            table_name = raw_name.rsplit('.', 1)[0].lower().replace("-", "_").replace(" ", "_")
            
            con = duckdb.connect(database=':memory:')
            con.register(table_name, raw_df)
            if table_name != 'data':
                con.register('data', raw_df)
                
            clean_sql = sql_query.strip().rstrip(';')
            count_query = f"SELECT COUNT(*) FROM ({clean_sql})"
            return int(con.execute(count_query).fetchone()[0])
        except Exception:
            return None


st.set_page_config(page_title="Index Buddy | AI Tuning", layout="wide")
st.title("Index Buddy")
st.markdown("Universal AI-Powered Database Index Advisor using AST Parsing and Extra Trees Classification.")
st.divider()

tab1, tab2 = st.tabs(["Single Query Analysis", "Bulk Workload Analyzer (Dataset + Logs)"])

# ==============================================================================
# TAB 1: SINGLE QUERY ANALYSIS
# ==============================================================================
with tab1:
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.subheader("1. Workload Input")
        sql_input = st.text_area(
            "Paste SQL Statement:", 
            value="", 
            placeholder="e.g., SELECT * FROM X WHERE Age > 30 AND Sex = 'female';", 
            height=120
        )
        
        parsed_data = parse_sql_query(sql_input) if sql_input.strip() else {}
        
        st.markdown("**Extracted AST Features:**")
        if parsed_data:
            st.json(parsed_data)
        else:
            st.info("Enter a SQL query above to view extracted AST features.")
        
        ds_file = st.session_state.get("ds_upload", None)
        target_table = parsed_data.get('table_name', '') if parsed_data else ''
        
        # Determine total table rows dynamically or fallback
        if target_table:
            auto_t_rows = resolve_table_row_count(target_table, dataset_file=ds_file)
        else:
            auto_t_rows = 50000
            
        auto_m_rows = estimate_matching_rows_from_df(ds_file, sql_input) if sql_input.strip() else None
        
        st.markdown("**Database Context:**")
        if ds_file and target_table:
            st.success(f"Validated Context: Target table '{target_table}' detected ({auto_t_rows:,} rows).")
        elif target_table:
            st.info(f"Hypothetical Context: Target table '{target_table}' ({auto_t_rows:,} rows default). Upload CSV in Tab 2 for live statistics.")
        else:
            st.caption("*(Upload dataset in Tab 2 or enter a query targeting a table to auto-detect context)*")
            
        t_rows = st.number_input("Total Table Rows", min_value=10, value=auto_t_rows)
        
        if auto_m_rows is not None:
            st.caption(f"*(Dynamically measured **{auto_m_rows:,}** matching rows from uploaded CSV)*")
            initial_m_val = auto_m_rows
        else:
            st.caption("*(Upload CSV in Tab 2 for dynamic matching row measurement, or enter EXPLAIN estimate manually)*")
            initial_m_val = 1
            
        m_rows = st.number_input("Estimated Matching Rows (From EXPLAIN)", min_value=1, value=initial_m_val)

    with col2:
        st.subheader("2. AI Recommendation")
        if st.button("Analyze Query Viability", type="primary", width="stretch"):
            if not sql_input.strip():
                st.warning("Please paste a valid SQL statement first.")
            else:
                try:
                    result = evaluate_index_viability(parsed_data, t_rows, m_rows)
                    st.metric("Calculated Selectivity", f"{result['selectivity']:.6f}")
                    
                    if result['is_beneficial']:
                        st.success(f"Index Recommended ({result['confidence']*100:.1f}% Confidence)")
                        st.code(result['ddl_statement'], language="sql")
                    else:
                        st.warning(f"Index Not Recommended ({result['confidence']*100:.1f}% Confidence)")
                        st.info("The AI determined that an index here would likely increase write overhead without providing significant read improvements.")
                except Exception as e:
                    st.error(f"System Error: {e}")

# ==============================================================================
# TAB 2: ENTERPRISE WORKLOAD AUTO-TUNING
# ==============================================================================
with tab2:
    st.subheader("Enterprise Workload Auto-Tuning")
    st.markdown("Upload both your **Database Table Statistics Dataset** and your **Query Execution Logs** to run a cross-functional workload analysis.")
    
    col_file1, col_file2 = st.columns(2)
    
    with col_file1:
        st.markdown("#### Step A: Table Metadata Dataset")
        sample_dataset_csv = "table_name,total_rows\n"
        st.download_button("Download CSV Template", data=sample_dataset_csv, file_name="table_stats_template.csv", mime="text/csv")
        dataset_file = st.file_uploader("Upload Table Statistics or Raw Dataset (CSV)", type=["csv", "txt"], key="ds_upload")
        
    with col_file2:
        st.markdown("#### Step B: Query Workload Logs")
        sample_logs_csv = "query_text,frequency,avg_latency_ms\n"
        st.download_button("Download Logs Template", data=sample_logs_csv, file_name="query_logs_template.csv", mime="text/csv")
        logs_file = st.file_uploader("Upload Query Logs (CSV)", type=["csv", "txt"], key="log_upload")
    
    if dataset_file or logs_file:
        st.markdown("---")
        col_view1, col_view2 = st.columns(2)
        
        table_df = None
        log_df = None
        
        with col_view1:
            st.markdown("**Loaded Table Statistics:**")
            if dataset_file:
                table_df = extract_dataset_metadata(dataset_file)
                st.dataframe(table_df, width="stretch")
            else:
                st.info("Waiting for Table Statistics dataset (Step A)...")

        with col_view2:
            st.markdown("**Loaded Query Logs:**")
            if logs_file:
                raw_log_df = safe_read_csv(logs_file)
                
                if raw_log_df.shape[1] == 1 or 'query_text' not in [str(c).lower() for c in raw_log_df.columns]:
                    first_col = raw_log_df.columns[0]
                    log_df = raw_log_df.value_counts(first_col).reset_index()
                    log_df.columns = ['query_text', 'frequency']
                    log_df['avg_latency_ms'] = 250.0
                else:
                    log_df = raw_log_df.copy()
                    q_cols = [c for c in log_df.columns if 'query' in str(c).lower()]
                    f_cols = [c for c in log_df.columns if 'freq' in str(c).lower() or 'count' in str(c).lower()]
                    l_cols = [c for c in log_df.columns if 'lat' in str(c).lower() or 'time' in str(c).lower()]
                    
                    log_df.rename(columns={
                        q_cols[0] if q_cols else log_df.columns[0]: 'query_text',
                        f_cols[0] if f_cols else 'frequency': 'frequency',
                        l_cols[0] if l_cols else 'avg_latency_ms': 'avg_latency_ms'
                    }, inplace=True)
                    
                    if 'frequency' not in log_df.columns:
                        log_df['frequency'] = 1
                    if 'avg_latency_ms' not in log_df.columns:
                        log_df['avg_latency_ms'] = 100.0

                st.dataframe(log_df.head(), width="stretch")
            else:
                st.info("Waiting for Query Logs upload (Step B)...")
                
        # --- Validation & Analysis ---
        if logs_file and not dataset_file:
            st.error("Dataset Required: You have uploaded Query Logs, but the Table Metadata Dataset is missing. You must upload a dataset file in Step A to analyze your logs.")
            
        elif dataset_file and logs_file:
            if st.button("Run Comprehensive Workload Analysis", type="primary", width="stretch"):
                table_row_map = dict(zip(table_df['table_name'].astype(str).str.lower(), table_df['total_rows']))
                default_total_rows = table_df['total_rows'].iloc[0] if not table_df.empty and 'total_rows' in table_df.columns else 50000
                
                recommendations = []
                
                for _, row in log_df.iterrows():
                    sql_text = str(row['query_text'])
                    parsed = parse_sql_query(sql_text)
                    target_tbl = parsed.get('table_name', 'data')
                    cand_col = parsed.get('candidate_column', 'unknown')
                    
                    total_rows = table_row_map.get(target_tbl.lower(), default_total_rows)
                    latency_val = float(row.get('avg_latency_ms', 100.0))
                    
                    estimated_matching_rows = int(total_rows * 0.02) if latency_val > 200 else int(total_rows * 0.15)
                    
                    res = evaluate_index_viability(parsed, total_rows, estimated_matching_rows)
                    
                    if res['is_beneficial']:
                        recommendations.append({
                            "Table": target_tbl,
                            "Column": cand_col,
                            "Model Confidence": f"{res['confidence']*100:.1f}%",
                            "Workload Frequency": row.get('frequency', 1),
                            "Avg Latency (ms)": latency_val,
                            "Recommended DDL": res['ddl_statement']
                        })
                
                if recommendations:
                    rec_df = pd.DataFrame(recommendations).drop_duplicates(subset=["Table", "Column"])
                    st.success(f"Analysis Complete. Evaluated {len(log_df)} queries against dataset statistics and found {len(rec_df)} optimal indexes.")
                    st.dataframe(rec_df, width="stretch")
                    
                    st.markdown("#### Consolidated Execution Script")
                    for sql in rec_df["Recommended DDL"]:
                        st.code(sql, language="sql")
                else:
                    st.warning("Analysis Complete. Current workload is fully optimized; no new indexes recommended.")