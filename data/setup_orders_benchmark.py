"""
data/setup_orders_benchmark.py
──────────────────────────────
Creates the ``orders_benchmark`` table with ~1,000,000 rows
(configurable via BENCHMARK_ROWS in .env / config.py or --rows CLI flag).

Columns are designed with varying cardinalities and skew:
  order_id       – INT AUTO_INCREMENT PRIMARY KEY
  customer_id    – INT NOT NULL, high cardinality (~100,000 distinct)
  product_id     – INT NOT NULL, medium cardinality (~5,000 distinct)
  status         – VARCHAR(20) NOT NULL, very low cardinality & skewed
                   (~70% 'completed', ~15% 'shipped', ~8% 'processing',
                    ~5% 'cancelled', ~2% 'returned')
  region         – VARCHAR(20) NOT NULL, low cardinality (8 regions)
  order_date     – DATE NOT NULL, ~3 years of dates (2022-01-01 to 2024-12-31)
  amount         – DECIMAL(10,2) NOT NULL
  payment_method – VARCHAR(20) NOT NULL, low cardinality (4 methods)
  notes          – TEXT NULL (~75% NULL)

Deliberately creates NO secondary indexes on ``orders_benchmark``.
Also creates helper table ``customers_benchmark`` for JOIN queries.
Runs ``ANALYZE TABLE`` at the end.

Usage:
    python -m data.setup_orders_benchmark
    python -m data.setup_orders_benchmark --rows 1000000
"""
from __future__ import annotations

import argparse
import math
import random
import time
from datetime import date, timedelta
from typing import Optional

import mysql.connector

import config
from db.mysql_adapter import MySQLAdapter


STATUS_VALUES = ["completed", "shipped", "processing", "cancelled", "returned"]
STATUS_WEIGHTS = [0.70, 0.15, 0.08, 0.05, 0.02]

REGIONS = [
    "North", "South", "East", "West",
    "NorthEast", "NorthWest", "SouthEast", "SouthWest",
]
PAYMENTS = ["credit_card", "debit_card", "paypal", "bank_transfer"]
TIERS = ["bronze", "silver", "gold", "platinum"]
TIER_WEIGHTS = [0.50, 0.30, 0.15, 0.05]

NOTE_CHOICES = [
    "'Urgent delivery requested'",
    "'Gift wrap needed'",
    "'Leave at front door'",
    "'Contact before delivery'",
    "'Fragile items inside'",
] + ["NULL"] * 15  # 75% NULL

START_DATE = date(2022, 1, 1)
DATE_RANGE_DAYS = 365 * 3 - 1  # 2022-01-01 through 2024-12-31


def _ensure_database_exists() -> None:
    """Create the configured MySQL database if it does not already exist."""
    conn = mysql.connector.connect(
        host=config.MYSQL_HOST,
        port=config.MYSQL_PORT,
        user=config.MYSQL_USER,
        password=config.MYSQL_PASSWORD,
        autocommit=True,
    )
    cur = conn.cursor()
    try:
        cur.execute(f"CREATE DATABASE IF NOT EXISTS `{config.MYSQL_DB}`")
    finally:
        cur.close()
        conn.close()


def setup(num_rows: Optional[int] = None, seed: int = 42) -> None:
    """Populate orders_benchmark and customers_benchmark with no secondary indexes."""
    random.seed(seed)
    rows = num_rows or config.BENCHMARK_ROWS

    _ensure_database_exists()
    db = MySQLAdapter()
    db.connect()
    print(f"Connected to MySQL {db.host}:{db.port}/{db.database}")

    # Pre-compute date strings for fast sampling
    date_pool = [
        f"'{(START_DATE + timedelta(days=d)).isoformat()}'"
        for d in range(DATE_RANGE_DAYS + 1)
    ]

    # Pre-compute weighted status pool of size 100 for O(1) sampling
    status_pool: list[str] = []
    for val, weight in zip(STATUS_VALUES, STATUS_WEIGHTS):
        status_pool.extend([f"'{val}'"] * int(round(weight * 100)))

    region_pool = [f"'{r}'" for r in REGIONS]
    payment_pool = [f"'{p}'" for p in PAYMENTS]

    # ── Create tables (NO secondary indexes) ────────────────────
    print(f"Dropping and recreating `{config.BENCHMARK_TABLE}` and `customers_benchmark`...")
    db.execute_ddl(f"DROP TABLE IF EXISTS `{config.BENCHMARK_TABLE}`")
    db.execute_ddl("DROP TABLE IF EXISTS `customers_benchmark`")

    db.execute_ddl(f"""
        CREATE TABLE `{config.BENCHMARK_TABLE}` (
            order_id       INT AUTO_INCREMENT PRIMARY KEY,
            customer_id    INT            NOT NULL,
            product_id     INT            NOT NULL,
            status         VARCHAR(20)    NOT NULL,
            region         VARCHAR(20)    NOT NULL,
            order_date     DATE           NOT NULL,
            amount         DECIMAL(10,2)  NOT NULL,
            payment_method VARCHAR(20)    NOT NULL,
            notes          TEXT           NULL
        ) ENGINE=InnoDB
    """)

    db.execute_ddl("""
        CREATE TABLE `customers_benchmark` (
            customer_id  INT PRIMARY KEY,
            name         VARCHAR(100) NOT NULL,
            tier         VARCHAR(20)  NOT NULL,
            region       VARCHAR(20)  NOT NULL,
            signup_date  DATE         NOT NULL
        ) ENGINE=InnoDB
    """)

    # ── Insert orders_benchmark in batches ──────────────────────
    batch_size = 10_000
    total_batches = math.ceil(rows / batch_size)
    num_customers = max(rows // 10, 1_000)   # 100,000 for 1M rows
    num_products = max(rows // 200, 100)     # 5,000 for 1M rows

    print(
        f"Inserting {rows:,} rows into `{config.BENCHMARK_TABLE}` "
        f"({total_batches} batches of {batch_size:,})..."
    )
    t0 = time.perf_counter()

    randint = random.randint
    choice = random.choice
    uniform = random.uniform

    for batch_idx in range(total_batches):
        batch_rows = min(batch_size, rows - batch_idx * batch_size)
        values_parts = [
            f"({randint(1, num_customers)},"
            f"{randint(1, num_products)},"
            f"{choice(status_pool)},"
            f"{choice(region_pool)},"
            f"{choice(date_pool)},"
            f"{uniform(5.0, 5000.0):.2f},"
            f"{choice(payment_pool)},"
            f"{choice(NOTE_CHOICES)})"
            for _ in range(batch_rows)
        ]
        sql = (
            f"INSERT INTO `{config.BENCHMARK_TABLE}` "
            "(customer_id,product_id,status,region,order_date,amount,payment_method,notes) "
            f"VALUES {','.join(values_parts)}"
        )
        db.execute_ddl(sql)

        if (batch_idx + 1) % 10 == 0 or batch_idx == total_batches - 1:
            elapsed = time.perf_counter() - t0
            inserted = min((batch_idx + 1) * batch_size, rows)
            pct = inserted / rows * 100
            print(f"  [{pct:5.1f}%] {inserted:>10,} / {rows:,} rows ({elapsed:.1f}s)")

    # ── Insert customers_benchmark ──────────────────────────────
    print(f"Inserting {num_customers:,} rows into `customers_benchmark`...")
    tier_pool: list[str] = []
    for t_val, t_w in zip(TIERS, TIER_WEIGHTS):
        tier_pool.extend([f"'{t_val}'"] * int(round(t_w * 100)))

    cust_batch = 10_000
    for start in range(1, num_customers + 1, cust_batch):
        end = min(start + cust_batch, num_customers + 1)
        parts = [
            f"({cid},'Customer_{cid}',{choice(tier_pool)},{choice(region_pool)},{choice(date_pool)})"
            for cid in range(start, end)
        ]
        db.execute_ddl(
            "INSERT INTO `customers_benchmark` (customer_id,name,tier,region,signup_date) "
            f"VALUES {','.join(parts)}"
        )

    # ── ANALYZE TABLE ───────────────────────────────────────────
    print("Running ANALYZE TABLE on `orders_benchmark` and `customers_benchmark`...")
    res1 = db.analyze_table(config.BENCHMARK_TABLE)
    res2 = db.analyze_table("customers_benchmark")
    for r in res1 + res2:
        print(f"  ANALYZE: {r}")

    # ── Verification output ─────────────────────────────────────
    final_count = db.row_count(config.BENCHMARK_TABLE)
    cust_count = db.row_count("customers_benchmark")
    sec_indexes = db.get_non_primary_indexes(config.BENCHMARK_TABLE)
    total_elapsed = time.perf_counter() - t0

    print("\n=== Benchmark Setup Summary ===")
    print(f"Table `{config.BENCHMARK_TABLE}` rows : {final_count:,}")
    print(f"Table `customers_benchmark` rows: {cust_count:,}")
    print(f"Secondary indexes on `{config.BENCHMARK_TABLE}`: {len(sec_indexes)} ({sec_indexes})")

    stats_row = db.execute(f"""
        SELECT
            COUNT(DISTINCT order_id)       AS card_order_id,
            COUNT(DISTINCT customer_id)    AS card_customer_id,
            COUNT(DISTINCT product_id)     AS card_product_id,
            COUNT(DISTINCT status)         AS card_status,
            COUNT(DISTINCT region)         AS card_region,
            COUNT(DISTINCT order_date)     AS card_order_date,
            COUNT(DISTINCT amount)         AS card_amount,
            COUNT(DISTINCT payment_method) AS card_payment_method,
            SUM(CASE WHEN notes IS NULL THEN 1 ELSE 0 END) AS null_notes
        FROM `{config.BENCHMARK_TABLE}`
    """)[0]
    print("\n=== Live Column Cardinalities (COUNT DISTINCT) ===")
    for k, v in stats_row.items():
        print(f"  {k:<22}: {int(v):,}")

    status_dist = db.execute(f"""
        SELECT status, COUNT(*) AS cnt,
               ROUND(COUNT(*) * 100.0 / {final_count}, 2) AS pct
        FROM `{config.BENCHMARK_TABLE}`
        GROUP BY status
        ORDER BY cnt DESC
    """)
    print("\n=== Live `status` Skew Distribution ===")
    for row in status_dist:
        print(f"  {row['status']:<12}: {int(row['cnt']):>9,} rows ({row['pct']}%)")

    print(f"\nTotal setup time: {total_elapsed:.1f}s")
    db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set up orders_benchmark table")
    parser.add_argument(
        "--rows",
        type=int,
        default=None,
        help="Number of rows to insert (default: BENCHMARK_ROWS from config/.env)",
    )
    args = parser.parse_args()
    setup(args.rows)
