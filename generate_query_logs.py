import os
import glob
import pandas as pd
import numpy as np

base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(base_dir, "dataset")
os.makedirs(dataset_dir, exist_ok=True)

# 1. Locate schema or retail dataset CSV
schema_path = os.path.join(dataset_dir, "database_schema.csv")
retail_csvs = [f for f in glob.glob(os.path.join(dataset_dir, "*.csv")) 
               if os.path.basename(f) not in ["query_workload_logs.csv", "database_schema.csv"]]

if os.path.exists(schema_path):
    schema_df = pd.read_csv(schema_path)
    table_name = schema_df['table_name'].iloc[0]
    columns = schema_df['column_name'].tolist()
elif retail_csvs:
    df = pd.read_csv(retail_csvs[0])
    table_name = os.path.splitext(os.path.basename(retail_csvs[0]))[0].lower().replace("-", "_").replace(" ", "_")
    columns = df.columns.tolist()
else:
    table_name = "retail_sales"
    columns = ["transaction_id", "customer_id", "category", "price", "quantity"]

# 2. Build template pool of SQL queries
query_pool = []
for col in columns:
    query_pool.append(f"SELECT * FROM {table_name} WHERE {col} = 'sample_val';")
    query_pool.append(f"SELECT * FROM {table_name} WHERE {col} > 100;")

# 3. Generate raw traffic log with repeated queries (simulating frequency)
np.random.seed(42)
raw_lines = []
total_log_lines = 1000  # Total raw query executions

for _ in range(total_log_lines):
    # Randomly pick queries from the pool (duplicates simulate execution frequency)
    selected_query = np.random.choice(query_pool)
    raw_lines.append(selected_query)

# 4. Save raw queries line-by-line to raw_queries.txt
raw_txt_path = os.path.join(dataset_dir, "raw_queries.txt")
with open(raw_txt_path, "w", encoding="utf-8") as f:
    f.write("\n".join(raw_lines))

print(f"[SUCCESS] Generated {len(raw_lines)} raw query executions into: {raw_txt_path}")