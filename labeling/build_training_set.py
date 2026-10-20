"""
labeling/build_training_set.py
──────────────────────────────
Builds an empirical training/evaluation dataset of >= 300 varied SQL queries
instantiated from 21 query templates against the live MySQL ``orders_benchmark``
and ``customers_benchmark`` tables.

For each (query, candidate_index):
  1. Measures baseline median execution time over 5 runs (1 warm-up discarded)
     and EXPLAIN rows examined with NO secondary indexes.
  2. Creates the candidate index on MySQL, runs ANALYZE TABLE, and measures
     median execution time over 5 runs (1 warm-up discarded) and EXPLAIN rows
     examined WITH the index.
  3. Records speedup = time_before / time_after, log_speedup = log(speedup),
     rows_before, rows_after, and all extracted catalog/query/EXPLAIN features.
  4. Drops the candidate index before moving on.

Restartable: saves progress after every measurement to ``labeling/training_data.csv``
and skips already-recorded ``(query_id, index_name)`` pairs on restart.

Template holdout guarantee:
  - Demo templates (T03, T07, T11) and held-out test templates (T16, T19) are
    tagged with ``split_group`` in {'demo_holdout', 'test'} so ``models/train.py``
    trains strictly on ``split_group == 'train'``.
"""
from __future__ import annotations

import argparse
import csv
import math
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Set, Tuple

import config
from candidates.generator import CandidateIndex, generate_candidates
from db.mysql_adapter import MySQLAdapter
from features.extract import FEATURE_NAMES, StatsCache, extract_features
from parsing.query_parser import parse_query


@dataclass(frozen=True)
class QueryInstance:
    """A single instantiated query belonging to a named template."""

    query_id: str
    template_id: str
    template_desc: str
    split_group: str  # 'train' | 'test' | 'demo_holdout'
    sql: str


STATUS_ALL = ["completed", "shipped", "processing", "cancelled", "returned"]
REGIONS = [
    "North", "South", "East", "West",
    "NorthEast", "NorthWest", "SouthEast", "SouthWest",
]
PAYMENTS = ["credit_card", "debit_card", "paypal", "bank_transfer"]
TIERS = ["bronze", "silver", "gold", "platinum"]
START_DATE = date(2022, 1, 1)


def _rand_date(rng: random.Random, max_offset: int = 1000) -> str:
    return (START_DATE + timedelta(days=rng.randint(0, max_offset))).isoformat()


def _rand_date_window(rng: random.Random, window_days: int = 14) -> Tuple[str, str]:
    d1 = START_DATE + timedelta(days=rng.randint(0, 1050 - window_days))
    d2 = d1 + timedelta(days=window_days)
    return d1.isoformat(), d2.isoformat()


def generate_workload(
    queries_per_template: int = 15,
    num_customers: int = 100_000,
    num_products: int = 5_000,
    seed: int = 42,
) -> List[QueryInstance]:
    """Generate 21 * queries_per_template (default 315) varied SQL queries."""
    rng = random.Random(seed)

    templates: List[Tuple[str, str, str, Callable[[int, random.Random], str]]] = [
        # ── Point Lookups (Equality) ────────────────────────────
        (
            "T01",
            "Point lookup on high-cardinality customer_id",
            "train",
            lambda i, r: (
                f"SELECT order_id, customer_id, product_id, amount, order_date "
                f"FROM orders_benchmark WHERE customer_id = {r.randint(1, num_customers)}"
            ),
        ),
        (
            "T02",
            "Point lookup on medium-cardinality product_id",
            "train",
            lambda i, r: (
                f"SELECT order_id, customer_id, product_id, status, amount "
                f"FROM orders_benchmark WHERE product_id = {r.randint(1, num_products)}"
            ),
        ),
        (
            "T03",
            "Low-cardinality skewed filter on status (Demo Holdout)",
            "demo_holdout",
            lambda i, r: (
                f"SELECT status, COUNT(*) AS order_cnt, AVG(amount) AS avg_amt "
                f"FROM orders_benchmark "
                f"WHERE status = '{STATUS_ALL[i % len(STATUS_ALL)]}' "
                f"GROUP BY status"
            ),
        ),
        (
            "T04",
            "Low-cardinality filter on region with aggregate",
            "train",
            lambda i, r: (
                f"SELECT region, COUNT(*) AS cnt, SUM(amount) AS total_amt "
                f"FROM orders_benchmark "
                f"WHERE region = '{REGIONS[i % len(REGIONS)]}' "
                f"GROUP BY region"
            ),
        ),
        (
            "T05",
            "Low-cardinality filter on payment_method with aggregate",
            "train",
            lambda i, r: (
                f"SELECT payment_method, COUNT(*) AS cnt, AVG(amount) AS avg_amt "
                f"FROM orders_benchmark "
                f"WHERE payment_method = '{PAYMENTS[i % len(PAYMENTS)]}' "
                f"GROUP BY payment_method"
            ),
        ),
        # ── Multi-Column Equality & IN Predicates ───────────────
        (
            "T06",
            "Composite equality on customer_id and status",
            "train",
            lambda i, r: (
                f"SELECT order_id, customer_id, status, amount, order_date "
                f"FROM orders_benchmark "
                f"WHERE customer_id = {r.randint(1, num_customers)} "
                f"AND status = '{r.choice(STATUS_ALL)}'"
            ),
        ),
        (
            "T07",
            "Composite equality on product_id and region (Demo Holdout)",
            "demo_holdout",
            lambda i, r: (
                f"SELECT order_id, customer_id, product_id, region, amount "
                f"FROM orders_benchmark "
                f"WHERE product_id = {r.randint(1, num_products)} "
                f"AND region = '{r.choice(REGIONS)}'"
            ),
        ),
        (
            "T08",
            "IN predicate on product_id with status equality",
            "train",
            lambda i, r: (
                f"SELECT order_id, product_id, status, amount "
                f"FROM orders_benchmark "
                f"WHERE product_id IN ({r.randint(1, num_products)}, "
                f"{r.randint(1, num_products)}, {r.randint(1, num_products)}) "
                f"AND status = '{r.choice(STATUS_ALL)}' LIMIT 100"
            ),
        ),
        (
            "T09",
            "IN predicate on customer_id",
            "train",
            lambda i, r: (
                f"SELECT order_id, customer_id, order_date, amount "
                f"FROM orders_benchmark "
                f"WHERE customer_id IN ({r.randint(1, num_customers)}, "
                f"{r.randint(1, num_customers)}, {r.randint(1, num_customers)}, "
                f"{r.randint(1, num_customers)})"
            ),
        ),
        # ── Range & LIKE Prefix Queries ─────────────────────────
        (
            "T10",
            "Date range window on order_date",
            "train",
            lambda i, r: (
                lambda d1, d2: (
                    f"SELECT order_date, COUNT(*) AS cnt, SUM(amount) AS sum_amt "
                    f"FROM orders_benchmark "
                    f"WHERE order_date BETWEEN '{d1}' AND '{d2}' "
                    f"GROUP BY order_date"
                )
            )(*_rand_date_window(r, window_days=r.choice([3, 7, 15, 60]))),
        ),
        (
            "T11",
            "Customer equality + order_date range + ORDER BY (Demo Holdout)",
            "demo_holdout",
            lambda i, r: (
                f"SELECT order_id, customer_id, product_id, order_date, amount "
                f"FROM orders_benchmark "
                f"WHERE customer_id = {r.randint(1, num_customers)} "
                f"AND order_date >= '{_rand_date(r, 900)}' "
                f"ORDER BY order_date DESC LIMIT 25"
            ),
        ),
        (
            "T12",
            "Product equality + amount range",
            "train",
            lambda i, r: (
                f"SELECT order_id, product_id, amount, order_date "
                f"FROM orders_benchmark "
                f"WHERE product_id = {r.randint(1, num_products)} "
                f"AND amount >= {r.randint(1000, 4500)}.00"
            ),
        ),
        (
            "T13",
            "LIKE prefix on payment_method with product_id equality",
            "train",
            lambda i, r: (
                f"SELECT order_id, product_id, payment_method, amount "
                f"FROM orders_benchmark "
                f"WHERE payment_method LIKE '{r.choice(['credit%', 'debit%', 'pay%', 'bank%'])}' "
                f"AND product_id = {r.randint(1, num_products)}"
            ),
        ),
        (
            "T14",
            "Narrow high-value range on amount with LIMIT",
            "train",
            lambda i, r: (
                lambda base: (
                    f"SELECT order_id, customer_id, amount, order_date "
                    f"FROM orders_benchmark "
                    f"WHERE amount BETWEEN {base:.2f} AND {base + 15.0:.2f} "
                    f"LIMIT 50"
                )
            )(r.uniform(4800.0, 4980.0)),
        ),
        # ── JOIN Queries on customers_benchmark ─────────────────
        (
            "T15",
            "JOIN orders_benchmark and customers_benchmark on customer_id",
            "train",
            lambda i, r: (
                f"SELECT o.order_id, o.amount, o.order_date, c.name, c.tier "
                f"FROM orders_benchmark o "
                f"JOIN customers_benchmark c ON o.customer_id = c.customer_id "
                f"WHERE c.customer_id = {r.randint(1, num_customers)}"
            ),
        ),
        (
            "T16",
            "JOIN with product_id and customer tier filter (Test Holdout)",
            "test",
            lambda i, r: (
                f"SELECT o.order_id, o.product_id, o.amount, c.name, c.tier "
                f"FROM orders_benchmark o "
                f"JOIN customers_benchmark c ON o.customer_id = c.customer_id "
                f"WHERE o.product_id = {r.randint(1, num_products)} "
                f"AND c.tier = '{r.choice(TIERS)}'"
            ),
        ),
        (
            "T17",
            "JOIN with customer_id IN list and order_date range",
            "train",
            lambda i, r: (
                f"SELECT o.order_id, o.customer_id, o.order_date, o.amount, c.tier "
                f"FROM orders_benchmark o "
                f"JOIN customers_benchmark c ON o.customer_id = c.customer_id "
                f"WHERE o.customer_id IN ({r.randint(1, num_customers)}, "
                f"{r.randint(1, num_customers)}, {r.randint(1, num_customers)}) "
                f"AND o.order_date >= '{_rand_date(r, 800)}'"
            ),
        ),
        # ── GROUP BY Queries ────────────────────────────────────
        (
            "T18",
            "GROUP BY status for a specific customer_id",
            "train",
            lambda i, r: (
                f"SELECT status, COUNT(*) AS cnt, SUM(amount) AS total_spent "
                f"FROM orders_benchmark "
                f"WHERE customer_id = {r.randint(1, num_customers)} "
                f"GROUP BY status"
            ),
        ),
        (
            "T19",
            "GROUP BY region for a specific product_id (Test Holdout)",
            "test",
            lambda i, r: (
                f"SELECT region, COUNT(*) AS cnt, AVG(amount) AS avg_amt "
                f"FROM orders_benchmark "
                f"WHERE product_id = {r.randint(1, num_products)} "
                f"GROUP BY region"
            ),
        ),
        # ── ORDER BY with LIMIT Queries ─────────────────────────
        (
            "T20",
            "Customer filter with ORDER BY amount DESC LIMIT 20",
            "train",
            lambda i, r: (
                f"SELECT order_id, customer_id, amount, order_date "
                f"FROM orders_benchmark "
                f"WHERE customer_id = {r.randint(1, num_customers)} "
                f"ORDER BY amount DESC LIMIT 20"
            ),
        ),
        (
            "T21",
            "Low-cardinality status + region filter with ORDER BY order_date LIMIT 30",
            "train",
            lambda i, r: (
                f"SELECT order_id, status, region, order_date, amount "
                f"FROM orders_benchmark "
                f"WHERE status = '{STATUS_ALL[i % len(STATUS_ALL)]}' "
                f"AND region = '{REGIONS[i % len(REGIONS)]}' "
                f"ORDER BY order_date DESC LIMIT 30"
            ),
        ),
    ]

    workload: List[QueryInstance] = []
    q_counter = 1
    for t_id, t_desc, split_grp, builder in templates:
        for i in range(queries_per_template):
            sql = builder(i, rng)
            q_id = f"Q{q_counter:03d}"
            workload.append(
                QueryInstance(
                    query_id=q_id,
                    template_id=t_id,
                    template_desc=t_desc,
                    split_group=split_grp,
                    sql=sql,
                )
            )
            q_counter += 1

    return workload


CSV_COLUMNS: List[str] = [
    "query_id",
    "template_id",
    "template_desc",
    "split_group",
    "sql",
    "template_sql",
    "table_name",
    "index_name",
    "index_columns",
    "ddl",
    "source_clauses",
    "time_before_s",
    "time_after_s",
    "speedup",
    "log_speedup",
    "rows_before",
    "rows_after",
    "access_type_before",
    "access_type_after",
    "key_used_after",
] + FEATURE_NAMES


def _load_completed_pairs(
    csv_path: Path,
) -> Tuple[Set[Tuple[str, str]], Dict[str, Tuple[float, int, str]]]:
    """Load already-measured (query_id, index_name) pairs and baseline query timings."""
    completed: Set[Tuple[str, str]] = set()
    baseline_timings: Dict[str, Tuple[float, int, str]] = {}
    if not csv_path.exists():
        return completed, baseline_timings

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = row.get("query_id", "")
            idx_name = row.get("index_name", "")
            if qid and idx_name:
                completed.add((qid, idx_name))
                try:
                    tb = float(row["time_before_s"])
                    rb = int(float(row["rows_before"]))
                    ab = str(row.get("access_type_before", "ALL"))
                    baseline_timings[qid] = (tb, rb, ab)
                except (KeyError, ValueError):
                    pass
    return completed, baseline_timings


def _explain_primary_row(
    db: MySQLAdapter, sql: str, primary_table: str = config.BENCHMARK_TABLE
) -> Tuple[int, str, str]:
    """Return (estimated_rows, access_type, key_used) for primary_table from EXPLAIN."""
    plan = db.explain(sql)
    target = None
    for r in plan:
        tbl = str(r.get("table") or "").lower()
        if tbl in (primary_table.lower(), "o"):
            target = r
            break
    if target is None and plan:
        target = plan[0]
    if target is None:
        return 0, "ALL", ""
    rows_est = int(target.get("rows") or 0)
    acc_type = str(target.get("type") or "ALL")
    key_used = str(target.get("key") or "")
    return rows_est, acc_type, key_used


_tls = threading.local()


def _get_worker_db() -> MySQLAdapter:
    """Return a thread-local MySQLAdapter connection for read-only timing runs."""
    db: MySQLAdapter | None = getattr(_tls, "db", None)
    if db is None or not db.is_connected():
        db = MySQLAdapter()
        db.connect()
        _tls.db = db
    return db


def _measure_single_query(
    sql: str, runs: int, warmup: int, fast_if_unused: bool = False
) -> Tuple[float, int, str, str]:
    """Execute *sql* over *runs* (discarding *warmup*) and return timing + EXPLAIN."""
    wdb = _get_worker_db()
    rows_est, acc_type, key_used = _explain_primary_row(wdb, sql)
    actual_runs = 2 if (fast_if_unused and not key_used and acc_type.upper() == "ALL") else runs
    med_time, _ = wdb.timed_execute(sql, runs=actual_runs, warmup=warmup)
    return med_time, rows_est, acc_type, key_used


def build_training_set(
    output_csv: str = config.LABEL_CSV,
    queries_per_template: int = 15,
    max_candidates_per_query: int = 4,
    runs: int = config.LABELING_RUNS,
    warmup: int = 1,
    workers: int = 3,
) -> None:
    """Run the full empirical labeling pipeline with restartable CSV checkpointing."""
    csv_path = Path(output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    db = MySQLAdapter()
    db.connect()
    print(f"Connected to MySQL {db.host}:{db.port}/{db.database}")

    # Ensure InnoDB buffer pool can hold the benchmark table in memory
    try:
        db.execute_ddl("SET GLOBAL innodb_buffer_pool_size = 1073741824")
    except Exception:
        pass

    # Always start with zero secondary indexes on orders_benchmark
    existing_sec = db.get_non_primary_indexes(config.BENCHMARK_TABLE)
    if existing_sec:
        print(f"Dropping leftover secondary indexes: {existing_sec}")
        db.drop_all_secondary_indexes(config.BENCHMARK_TABLE)
        db.analyze_table(config.BENCHMARK_TABLE)

    print("Collecting live catalog statistics for feature extraction...")
    stats_cache = StatsCache(db)
    t_stats = stats_cache.get_table_stats(config.BENCHMARK_TABLE)
    num_customers = max(1000, t_stats.columns["customer_id"].cardinality)
    num_products = max(100, t_stats.columns["product_id"].cardinality)
    print(
        f"Table `{t_stats.table_name}`: {t_stats.table_rows:,} rows | "
        f"customer_id card={num_customers:,} | product_id card={num_products:,}"
    )

    workload = generate_workload(
        queries_per_template=queries_per_template,
        num_customers=num_customers,
        num_products=num_products,
    )
    print(f"Generated {len(workload)} queries across 21 templates.")

    # Parse queries and generate candidates
    query_map: Dict[str, QueryInstance] = {q.query_id: q for q in workload}
    parsed_map = {q.query_id: parse_query(q.sql) for q in workload}
    candidates_by_query: Dict[str, List[CandidateIndex]] = {
        q.query_id: generate_candidates(
            parsed_map[q.query_id], max_candidates=max_candidates_per_query
        )
        for q in workload
    }

    total_pairs = sum(len(v) for v in candidates_by_query.values())
    completed_pairs, baseline_timings = _load_completed_pairs(csv_path)
    print(
        f"Total (query, candidate) pairs: {total_pairs} | "
        f"Already completed in CSV: {len(completed_pairs)}"
    )

    if len(completed_pairs) >= total_pairs:
        print("All (query, candidate) pairs are already labeled in CSV!")
        db.close()
        return

    # Pre-populate baseline EXPLAIN cache on main connection
    for q in workload:
        stats_cache.get_baseline_explain(q.sql)

    # ── Phase 1: Baseline measurements (NO secondary index) ─────
    needed_queries = [
        q
        for q in workload
        if q.query_id not in baseline_timings
        and any(
            (q.query_id, c.index_name) not in completed_pairs
            for c in candidates_by_query[q.query_id]
        )
    ]

    if needed_queries:
        print(
            f"\nPhase 1/2: Measuring baseline (no-index) timings for "
            f"{len(needed_queries)} queries ({runs} runs, {warmup} warm-up discarded, "
            f"{workers} read workers)..."
        )
        t_phase1 = time.perf_counter()
        done_b = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            fut_to_q = {
                pool.submit(_measure_single_query, q.sql, runs, warmup): q
                for q in needed_queries
            }
            for fut in as_completed(fut_to_q):
                q = fut_to_q[fut]
                med_time, rows_bef, acc_bef, _ = fut.result()
                baseline_timings[q.query_id] = (med_time, rows_bef, acc_bef)
                done_b += 1
                if done_b % 30 == 0 or done_b == len(needed_queries):
                    elapsed = time.perf_counter() - t_phase1
                    rate = done_b / max(elapsed, 1e-3)
                    rem = (len(needed_queries) - done_b) / max(rate, 1e-6)
                    print(
                        f"  [Baseline {done_b:>3}/{len(needed_queries)}] "
                        f"{q.query_id} ({q.template_id}) -> {med_time*1000:6.1f} ms, "
                        f"rows={rows_bef:,} | elapsed={elapsed:.0f}s, ETA={rem:.0f}s",
                        flush=True,
                    )

    # ── Phase 2: Candidate index measurements ───────────────────
    index_groups: Dict[str, Tuple[CandidateIndex, List[str]]] = {}
    for q in workload:
        for cand in candidates_by_query[q.query_id]:
            if (q.query_id, cand.index_name) in completed_pairs:
                continue
            if cand.index_name not in index_groups:
                index_groups[cand.index_name] = (cand, [])
            index_groups[cand.index_name][1].append(q.query_id)

    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    f_out = csv_path.open("a", encoding="utf-8", newline="")
    writer = csv.DictWriter(f_out, fieldnames=CSV_COLUMNS)
    if write_header:
        writer.writeheader()
        f_out.flush()

    print(
        f"\nPhase 2/2: Measuring {total_pairs - len(completed_pairs)} pending "
        f"(query, candidate) pairs across {len(index_groups)} distinct indexes..."
    )
    t_phase2 = time.perf_counter()
    pairs_done_now = 0
    pairs_to_do = total_pairs - len(completed_pairs)

    try:
        for g_idx, (idx_name, (cand, q_ids)) in enumerate(index_groups.items(), 1):
            db.drop_all_secondary_indexes(config.BENCHMARK_TABLE)
            t_ddl = time.perf_counter()
            db.execute_ddl(cand.ddl)
            db.analyze_table(cand.table)
            ddl_s = time.perf_counter() - t_ddl

            try:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    fut_to_qid = {
                        pool.submit(
                            _measure_single_query, query_map[qid].sql, runs, warmup, True
                        ): qid
                        for qid in q_ids
                    }
                    for fut in as_completed(fut_to_qid):
                        qid = fut_to_qid[fut]
                        q_inst = query_map[qid]
                        parsed = parsed_map[qid]
                        time_before, rows_before, acc_before = baseline_timings[qid]
                        time_after, rows_after, acc_after, key_used = fut.result()

                        speedup = time_before / max(time_after, 1e-6)
                        log_speedup = math.log(max(speedup, 1e-3))
                        feats = extract_features(parsed, cand, stats_cache)

                        row_dict: Dict[str, Any] = {
                            "query_id": q_inst.query_id,
                            "template_id": q_inst.template_id,
                            "template_desc": q_inst.template_desc,
                            "split_group": q_inst.split_group,
                            "sql": q_inst.sql,
                            "template_sql": parsed.template_sql,
                            "table_name": cand.table,
                            "index_name": cand.index_name,
                            "index_columns": ",".join(cand.columns),
                            "ddl": cand.ddl,
                            "source_clauses": "|".join(cand.source_clauses),
                            "time_before_s": round(time_before, 6),
                            "time_after_s": round(time_after, 6),
                            "speedup": round(speedup, 4),
                            "log_speedup": round(log_speedup, 6),
                            "rows_before": rows_before,
                            "rows_after": rows_after,
                            "access_type_before": acc_before,
                            "access_type_after": acc_after,
                            "key_used_after": key_used,
                        }
                        row_dict.update(feats)
                        writer.writerow(row_dict)
                        f_out.flush()
                        pairs_done_now += 1

                elapsed = time.perf_counter() - t_phase2
                print(
                    f"  [Index {g_idx:>2}/{len(index_groups)}] `{idx_name}` "
                    f"(DDL {ddl_s:.1f}s, {len(q_ids)} queries) -> "
                    f"progress {pairs_done_now}/{pairs_to_do} ({elapsed:.0f}s elapsed)",
                    flush=True,
                )
            finally:
                db.drop_index_if_exists(cand.table, cand.index_name)

        db.analyze_table(config.BENCHMARK_TABLE)
    finally:
        f_out.close()
        db.drop_all_secondary_indexes(config.BENCHMARK_TABLE)
        db.close()

    print(f"\nLabeling complete! Saved empirical dataset to {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build empirical MySQL index speedup dataset")
    parser.add_argument("--queries-per-template", type=int, default=15)
    parser.add_argument("--max-candidates", type=int, default=4)
    parser.add_argument("--runs", type=int, default=config.LABELING_RUNS)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--output", type=str, default=config.LABEL_CSV)
    args = parser.parse_args()
    build_training_set(
        output_csv=args.output,
        queries_per_template=args.queries_per_template,
        max_candidates_per_query=args.max_candidates,
        runs=args.runs,
        workers=args.workers,
    )
