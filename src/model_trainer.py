import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'dataset')

sys.path.append(BASE_DIR)
from src.sql_parser import parse_sql_query

def generate_default_datasets_if_missing():
    """Generates default enterprise sample files if none exist, ensuring zero-crash setup."""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    schema_path = os.path.join(DATA_DIR, "database_schema.csv")
    if not os.path.exists(schema_path):
        default_schema = pd.DataFrame({
            "table_name": ["retail_sales", "retail_sales", "retail_sales"],
            "column_name": ["transaction_id", "customer_id", "category"],
            "data_type": ["INT", "INT", "VARCHAR"],
            "total_rows": [100000, 100000, 100000],
            "cardinality": [100000, 25000, 15]
        })
        default_schema.to_csv(schema_path, index=False)
        
    query_logs_path = os.path.join(DATA_DIR, "query_workload_logs.csv")
    if not os.path.exists(query_logs_path):
        default_logs = pd.DataFrame({
            "query_text": [
                "SELECT * FROM retail_sales WHERE transaction_id = 101;",
                "SELECT * FROM retail_sales WHERE category = 'Electronics';",
                "SELECT * FROM retail_sales WHERE price > 500;"
            ],
            "table_name": ["retail_sales", "retail_sales", "retail_sales"],
            "frequency": [4500, 1200, 800],
            "avg_latency_ms": [150.5, 450.2, 520.0]
        })
        default_logs.to_csv(query_logs_path, index=False)

def train_index_advisor_model():
    print("[INFO] Checking and preparing training datasets...")
    generate_default_datasets_if_missing()
    
    schema_csv = os.path.join(DATA_DIR, "database_schema.csv")
    workload_csv = os.path.join(DATA_DIR, "query_workload_logs.csv")
    
    schema_df = pd.read_csv(schema_csv)
    log_df = pd.read_csv(workload_csv)
    
    # Map each table name to its true max total_rows from schema metadata
    table_row_map = schema_df.groupby(schema_df['table_name'].str.lower())['total_rows'].max().to_dict()
    
    training_rows = []
    
    print("[INFO] Processing and correlating workload logs with schema statistics...")
    for _, row in log_df.iterrows():
        sql_text = str(row['query_text']) if 'query_text' in row else ""
        latency = float(row['avg_latency_ms']) if 'avg_latency_ms' in row else 100.0
        frequency = int(row['frequency']) if 'frequency' in row else 1
        
        parsed = parse_sql_query(sql_text)
        target_table = parsed.get('table_name', 'retail_sales').lower()
        
        total_rows = table_row_map.get(target_table, 100000)
        
        # Estimate matching rows based on workload latency and query structure
        if latency > 300:
            matching_rows = int(total_rows * np.random.uniform(0.15, 0.40))
        else:
            matching_rows = int(total_rows * np.random.uniform(0.001, 0.04))
            
        selectivity = round(matching_rows / max(1, total_rows), 6)
        
        # Ground truth: Indexes are beneficial if latency is high, frequency is high, or join exists
        is_beneficial = 1 if (latency > 200 and selectivity < 0.10) or parsed.get('has_join', 0) == 1 or frequency > 5000 else 0
        
        training_rows.append([
            total_rows,
            selectivity,
            parsed.get('has_join', 0),
            parsed.get('is_range_scan', 0),
            is_beneficial
        ])
        
    # Data augmentation to ensure sufficient sample distribution for Extra Trees
    while len(training_rows) < 500:
        t_rows = np.random.choice([10000, 50000, 150000, 500000])
        sel = round(np.random.uniform(0.0001, 0.35), 6)
        join_flag = np.random.choice([0, 1])
        range_flag = np.random.choice([0, 1])
        beneficial = 1 if sel < 0.05 or join_flag == 1 else 0
        training_rows.append([t_rows, sel, join_flag, range_flag, beneficial])
        
    ml_df = pd.DataFrame(training_rows, columns=['table_rows', 'selectivity', 'has_join', 'is_range_scan', 'is_beneficial'])
    
    X = ml_df[['table_rows', 'selectivity', 'has_join', 'is_range_scan']]
    y = ml_df['is_beneficial']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = ExtraTreesClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds) * 100
    print(f"[INFO] Cross-Correlated Model Training Accuracy: {acc:.1f}%")
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "index_buddy_et.joblib")
    joblib.dump(model, model_path)
    print(f"[SUCCESS] Upgraded model compiled and saved to {model_path}")

if __name__ == "__main__":
    train_index_advisor_model()