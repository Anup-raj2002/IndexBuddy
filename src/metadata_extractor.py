import os
import pandas as pd
import duckdb

try:
    from src.utils import universal_read_csv
except ImportError:
    from utils import universal_read_csv


def extract_dataset_metadata(uploaded_file_or_path):
    """
    Extracts table metadata (table_name and total_rows) from uploaded 
    raw dataset CSVs or pre-formatted summary files safely.
    """
    if uploaded_file_or_path is None:
        return pd.DataFrame([{'table_name': 'data', 'total_rows': 50000}])

    df = universal_read_csv(uploaded_file_or_path)
    
    cols_lower = [str(c).lower() for c in df.columns]
    
    # Case 1: Pre-computed schema summary CSV
    if 'table_name' in cols_lower and 'total_rows' in cols_lower:
        return df

    # Case 2: Raw dataset CSV uploaded by user
    if hasattr(uploaded_file_or_path, 'name'):
        raw_name = uploaded_file_or_path.name
    elif isinstance(uploaded_file_or_path, str):
        raw_name = os.path.basename(uploaded_file_or_path)
    else:
        raw_name = 'data'

    # Clean table name format for SQL engine compatibility
    table_name = raw_name.rsplit('.', 1)[0].lower().replace("-", "_").replace(" ", "_")
    total_rows = len(df)
    
    table_summary = pd.DataFrame([{
        'table_name': table_name,
        'total_rows': total_rows,
        'column_count': len(df.columns)
    }])
    
    return table_summary


def resolve_table_row_count(target_table, dataset_file=None):
    """
    Resolves total row count for a given target table name using uploaded dataset file or metadata.
    """
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
    """
    Executes the SQL query against the uploaded dataset using DuckDB to estimate matching rows count.
    """
    if not dataset_file or not sql_query or not sql_query.strip():
        return None
    try:
        raw_df = universal_read_csv(dataset_file)
        if raw_df.empty:
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