import os
import sys
import pandas as pd
import numpy as np
from collections import Counter

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_dir)

from src.sql_parser import parse_sql_query

dataset_dir = os.path.join(base_dir, "dataset")
raw_txt_path = os.path.join(dataset_dir, "raw_queries.txt")

if not os.path.exists(raw_txt_path):
    raise FileNotFoundError(
        f"[ERROR] raw_queries.txt not found at {raw_txt_path}.\n"
        "Please run 'python generate_raw_queries.py' or provide raw queries in that file first."
    )

print(f"[INFO] Ingesting raw query traffic from {raw_txt_path}...")

with open(raw_txt_path, "r", encoding="utf-8", errors="ignore") as f:
    raw_queries = [line.strip() for line in f if line.strip() and not line.startswith("--")]

# 1. Count duplicate occurrences to determine frequency
frequency_map = Counter(raw_queries)

processed_rows = []
np.random.seed(42)

print("[INFO] Parsing queries, extracting AST metadata, and calculating metrics...")

for query_text, freq in frequency_map.items():
    # Extract AST details using your parser
    parsed = parse_sql_query(query_text)
    table_name = parsed.get("table_name", "retail_sales")
    
    # Calculate synthetic realistic latency based on query complexity and frequency
    base_latency = 25.0 if parsed.get("has_join", 0) == 0 else 250.0
    latency = float(np.round(base_latency + np.random.uniform(10, 500), 2))
    
    processed_rows.append({
        "query_text": query_text,
        "table_name": table_name,
        "frequency": freq,
        "avg_latency_ms": latency
    })

# 2. Export to query_workload_logs.csv matching the requested schema
output_csv_path = os.path.join(dataset_dir, "query_workload_logs.csv")
output_df = pd.DataFrame(processed_rows)

# Ensure explicit column order matching your required layout
output_df = output_df[["query_text", "table_name", "frequency", "avg_latency_ms"]]
output_df.to_csv(output_csv_path, index=False)

print(f"[SUCCESS] Processed {len(output_df)} unique queries from {len(raw_queries)} raw log lines.")
print(f"[SUCCESS] Output written to {output_csv_path}")