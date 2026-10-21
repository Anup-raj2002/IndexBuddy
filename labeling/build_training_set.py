"""
labeling/build_training_set.py
──────────────────────────────
Schema-agnostic workload generator and live MySQL speedup labeler for real
CTU Prague Relational Learning Repository databases.
Zero hardcoded table or column names.

1. Workload Generation (``generate_workload_for_database``):
   - Introspects tables, columns, data types, real value distributions (including
     real skew), and foreign keys (from ``data/original_indexes.json`` and
     ``information_schema``) in any database.
   - Generates >= 300 deduplicated SQL SELECT queries per database covering:
       * Point lookups (high-cardinality and low-cardinality/skewed values)
       * Range predicates (>, >=, <, <=, BETWEEN)
       * IN-lists
       * LIKE prefix predicates (on string columns)
       * Multi-column equality + range
       * Equality + ORDER BY ... LIMIT
       * GROUP BY aggregations (with and without WHERE filters)
       * Foreign-key JOIN queries
2. Live Speedup Labeling (``label_workload``):
   - For each query and each generated 1- and 2-column candidate index:
       * Measures baseline median runtime over 5 runs (1 warm-up discarded)
       * Creates the candidate index on local MySQL
       * Measures indexed median runtime over 5 runs (1 warm-up discarded)
       * Records speedup = before / after and EXPLAIN rows before / after
       * Drops the candidate index immediately
   - Restartable (skips already-labeled query_ids), prints live progress, and
     saves to ``labeling/training_data.csv``.
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd

import config
from candidates.generator import CandidateIndex, generate_candidates
from db.mysql_adapter import MySQLAdapter
from features.extract import (
    ColumnStats,
    FEATURE_NAMES,
    StatsCache,
    TableStats,
    extract_features,
)
from parsing.query_parser import ParsedQuery, parse_query


def _sql_lit(val: Any) -> str:
    if val is None:
        return "NULL"
    if isinstance(val, (int, float)):
        return str(val)
    s = str(val).replace("\\", "\\\\").replace("'", "''")
    return f"'{s}'"


def _fetch_distinct_sample_values(
    db: MySQLAdapter,
    table: str,
    col: str,
    limit: int = 120,
) -> List[Any]:
    """Sample distinct non-null values from a real column (both frequent and rare)."""
    vals: List[Any] = []
    try:
        rows = db.execute(
            f"SELECT DISTINCT `{col}` AS v FROM "
            f"(SELECT `{col}` FROM `{table}` WHERE `{col}` IS NOT NULL LIMIT 15000) AS s "
            f"LIMIT {limit}"
        )
        for r in rows:
            if r.get("v") is not None:
                vals.append(r["v"])
    except Exception:
        pass
    return vals


def generate_workload_for_database(
    db: MySQLAdapter,
    db_name: str,
    target_min_queries: int = 300,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Generate >= 300 deduplicated real SQL SELECT queries for *db_name*."""
    rng = random.Random(seed + sum(ord(c) for c in db_name))
    db.use_database(db_name)
    stats_cache = StatsCache(db)

    # Load saved foreign keys from original_indexes.json if available
    saved_fks: List[Dict[str, str]] = []
    if config.ORIGINAL_INDEXES_PATH.exists():
        try:
            orig_data = json.loads(config.ORIGINAL_INDEXES_PATH.read_text(encoding="utf-8"))
            saved_fks = orig_data.get(db_name, {}).get("foreign_keys", [])
        except Exception:
            saved_fks = []

    tables_info = db.list_tables(db_name)
    # Focus on tables with at least 500 rows so queries are meaningful
    eligible_tables = [
        str(t["table_name"])
        for t in tables_info
        if int(t.get("est_rows") or 0) >= 500
    ]
    if not eligible_tables and tables_info:
        eligible_tables = [str(tables_info[0]["table_name"])]

    # Introspect eligible tables and sample real values per column
    table_stats_map: Dict[str, TableStats] = {}
    col_values_map: Dict[Tuple[str, str], List[Any]] = {}

    for t_name in eligible_tables[:6]:
        ts = stats_cache.get_table_stats(t_name)
        if ts.table_rows < 200:
            continue
        table_stats_map[t_name] = ts
        for c_name, cs in ts.columns.items():
            if cs.is_indexable and cs.cardinality >= 1:
                vals = _fetch_distinct_sample_values(db, ts.table_name, c_name, limit=120)
                if vals:
                    col_values_map[(ts.table_name, c_name)] = vals

    generated: List[Dict[str, Any]] = []
    seen_sql: Set[str] = set()

    def _add_query(sql: str, table: str, category: str) -> bool:
        norm = " ".join(sql.strip().split())
        if norm in seen_sql:
            return False
        seen_sql.add(norm)
        generated.append(
            {
                "database_name": db_name,
                "table_name": table,
                "category": category,
                "sql": norm,
            }
        )
        return True

    # Generate diverse queries across all eligible tables in the database
    for t_name, ts in table_stats_map.items():
        pk_set = {c.lower() for c in ts.primary_key_columns}
        non_pk_cols = [
            cs
            for cs in ts.columns.values()
            if cs.is_indexable
            and cs.column_name.lower() not in pk_set
            and (ts.table_name, cs.column_name) in col_values_map
        ]
        # If all columns are part of a composite PK (e.g., lapTimes or Walmart train),
        # include all indexable columns!
        if len(non_pk_cols) < 2:
            non_pk_cols = [
                cs
                for cs in ts.columns.values()
                if cs.is_indexable and (ts.table_name, cs.column_name) in col_values_map
            ]

        if not non_pk_cols:
            continue

        proj_cols = [c.column_name for c in list(ts.columns.values())[:5]]
        proj = ", ".join(f"`{c}`" for c in proj_cols)

        range_cols = [
            cs
            for cs in non_pk_cols
            if cs.data_type_code in (1, 2, 3) and len(col_values_map[(ts.table_name, cs.column_name)]) >= 4
        ]
        str_cols = [
            cs
            for cs in non_pk_cols
            if cs.data_type_code == 4 and len(col_values_map[(ts.table_name, cs.column_name)]) >= 2
        ]

        # (A) Point lookups across columns and real sampled values (including skewed values)
        for cs in non_pk_cols:
            vals = col_values_map[(ts.table_name, cs.column_name)]
            for v in vals[:45]:
                _add_query(
                    f"SELECT {proj} FROM `{ts.table_name}` WHERE `{cs.column_name}` = {_sql_lit(v)}",
                    ts.table_name,
                    "point_lookup",
                )
                _add_query(
                    f"SELECT {proj} FROM `{ts.table_name}` WHERE `{cs.column_name}` = {_sql_lit(v)} LIMIT 25",
                    ts.table_name,
                    "point_lookup_limit",
                )

        # (B) Range predicates (>, >=, <, <=, BETWEEN)
        for r_col in range_cols:
            vals = sorted(col_values_map[(ts.table_name, r_col.column_name)], key=lambda x: str(x))
            for i in range(0, min(len(vals) - 2, 35)):
                v_low = vals[i]
                v_high = vals[min(i + 2, len(vals) - 1)]
                _add_query(
                    f"SELECT {proj} FROM `{ts.table_name}` WHERE `{r_col.column_name}` >= {_sql_lit(v_low)} LIMIT 50",
                    ts.table_name,
                    "range_scan",
                )
                _add_query(
                    f"SELECT {proj} FROM `{ts.table_name}` WHERE `{r_col.column_name}` BETWEEN {_sql_lit(v_low)} AND {_sql_lit(v_high)}",
                    ts.table_name,
                    "between_range",
                )

        # (C) IN-list queries
        for cs in non_pk_cols:
            vals = col_values_map[(ts.table_name, cs.column_name)]
            if len(vals) >= 4:
                for i in range(0, min(len(vals) - 3, 30)):
                    chunk = vals[i : i + 3]
                    in_str = ", ".join(_sql_lit(x) for x in chunk)
                    _add_query(
                        f"SELECT {proj} FROM `{ts.table_name}` WHERE `{cs.column_name}` IN ({in_str})",
                        ts.table_name,
                        "in_list",
                    )

        # (D) LIKE prefix queries on string columns
        for s_col in str_cols:
            vals = col_values_map[(ts.table_name, s_col.column_name)]
            for v in vals[:20]:
                s_txt = str(v).strip()
                if len(s_txt) >= 3:
                    prefix = s_txt[:3].replace("'", "''")
                    _add_query(
                        f"SELECT {proj} FROM `{ts.table_name}` WHERE `{s_col.column_name}` LIKE '{prefix}%' LIMIT 50",
                        ts.table_name,
                        "like_prefix",
                    )

        # (E) Composite equality + range + ORDER BY ... LIMIT
        for c1 in non_pk_cols:
            vals1 = col_values_map[(ts.table_name, c1.column_name)]
            for c2 in non_pk_cols:
                if c1.column_name == c2.column_name:
                    continue
                vals2 = col_values_map[(ts.table_name, c2.column_name)]
                for v1 in vals1[:25]:
                    v2 = rng.choice(vals2)
                    if c2.data_type_code in (1, 2, 3):
                        _add_query(
                            f"SELECT {proj} FROM `{ts.table_name}` "
                            f"WHERE `{c1.column_name}` = {_sql_lit(v1)} "
                            f"AND `{c2.column_name}` >= {_sql_lit(v2)} "
                            f"ORDER BY `{c2.column_name}` DESC LIMIT 25",
                            ts.table_name,
                            "composite_eq_range_orderby",
                        )
                    _add_query(
                        f"SELECT {proj} FROM `{ts.table_name}` "
                        f"WHERE `{c1.column_name}` = {_sql_lit(v1)} "
                        f"AND `{c2.column_name}` = {_sql_lit(v2)}",
                        ts.table_name,
                        "composite_two_eq",
                    )

        # (F) GROUP BY aggregations
        for c1 in non_pk_cols:
            vals1 = col_values_map[(ts.table_name, c1.column_name)]
            _add_query(
                f"SELECT `{c1.column_name}`, COUNT(*) AS cnt FROM `{ts.table_name}` GROUP BY `{c1.column_name}` LIMIT 30",
                ts.table_name,
                "group_by_unfiltered",
            )
            for c2 in non_pk_cols:
                if c1.column_name == c2.column_name:
                    continue
                for v1 in vals1[:20]:
                    _add_query(
                        f"SELECT `{c2.column_name}`, COUNT(*) AS cnt FROM `{ts.table_name}` "
                        f"WHERE `{c1.column_name}` = {_sql_lit(v1)} GROUP BY `{c2.column_name}`",
                        ts.table_name,
                        "group_by_filtered",
                    )

        # (G) Foreign-key JOIN queries
        for fk in saved_fks:
            if fk["table_name"].lower() == ts.table_name.lower():
                ref_tbl = fk["ref_table_name"]
                fk_col = fk["column_name"]
                ref_col = fk["ref_column_name"]
                for c1 in non_pk_cols:
                    vals1 = col_values_map[(ts.table_name, c1.column_name)]
                    for v1 in vals1[:25]:
                        _add_query(
                            f"SELECT t1.`{fk_col}`, t1.`{c1.column_name}` "
                            f"FROM `{ts.table_name}` t1 "
                            f"JOIN `{ref_tbl}` t2 ON t1.`{fk_col}` = t2.`{ref_col}` "
                            f"WHERE t1.`{c1.column_name}` = {_sql_lit(v1)} LIMIT 30",
                            ts.table_name,
                            "fk_join",
                        )

    # Shuffle deterministically while ensuring all categories are well-represented
    rng.shuffle(generated)
    if len(generated) < target_min_queries:
        # Pad with additional value combinations if needed
        for t_name, ts in table_stats_map.items():
            for (tbl_k, col_k), vals in col_values_map.items():
                if tbl_k != ts.table_name:
                    continue
                for v in vals:
                    for lim in (10, 20, 50, 100):
                        _add_query(
                            f"SELECT * FROM `{ts.table_name}` WHERE `{col_k}` = {_sql_lit(v)} LIMIT {lim}",
                            ts.table_name,
                            "point_lookup_limit",
                        )
                        if len(generated) >= target_min_queries:
                            break
                    if len(generated) >= target_min_queries:
                        break

    return generated


def build_all_workloads(out_csv: str = config.WORKLOAD_CSV) -> pd.DataFrame:
    """Generate >= 300 deduplicated queries for each of the 6 real CTU Prague databases."""
    db = MySQLAdapter()
    db.connect()
    all_rows: List[Dict[str, Any]] = []
    try:
        for db_name in config.SELECTED_CTU_DATABASES:
            qs = generate_workload_for_database(db, db_name, target_min_queries=310)
            split_group = "train" if db_name in config.TRAIN_DATABASES else "test"
            for idx, q in enumerate(qs, 1):
                q["query_id"] = f"{db_name}_Q{idx:03d}"
                q["split_group"] = split_group
                all_rows.append(q)
            print(
                f"  [WORKLOAD] `{db_name:<14}` ({split_group:<5}): "
                f"generated {len(qs)} deduplicated real queries"
            )
    finally:
        db.close()

    df = pd.DataFrame(all_rows)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved {len(df)} total generated queries across {len(config.SELECTED_CTU_DATABASES)} databases to {out_csv}")
    return df


def select_representative_queries_for_labeling(
    workload_df: pd.DataFrame,
    queries_per_db: int = 28,
) -> pd.DataFrame:
    """Select stratified queries across categories from each database for physical index labeling."""
    selected_chunks: List[pd.DataFrame] = []
    for db_name, grp in workload_df.groupby("database_name", sort=False):
        # Prioritize largest table in that database and diverse query categories
        cats = grp["category"].unique().tolist()
        per_cat = max(3, queries_per_db // max(1, len(cats)))
        picked_indices: List[int] = []
        for cat in cats:
            sub = grp[grp["category"] == cat]
            picked_indices.extend(sub.index[:per_cat].tolist())
        if len(picked_indices) < queries_per_db:
            remaining = [i for i in grp.index if i not in picked_indices]
            picked_indices.extend(remaining[: queries_per_db - len(picked_indices)])
        selected_chunks.append(grp.loc[picked_indices[:queries_per_db]])
    return pd.concat(selected_chunks, ignore_index=True)


def label_workload(
    workload_csv: str = config.WORKLOAD_CSV,
    output_csv: str = config.LABEL_CSV,
    queries_per_db: int = 26,
    runs: int = config.LABELING_RUNS,
) -> pd.DataFrame:
    """Measure real before/after execution times and EXPLAIN rows on local MySQL."""
    if not Path(workload_csv).exists():
        build_all_workloads(workload_csv)

    w_df = pd.read_csv(workload_csv)
    target_df = select_representative_queries_for_labeling(w_df, queries_per_db=queries_per_db)

    out_path = Path(output_csv)
    existing_rows: List[Dict[str, Any]] = []
    completed_qids: Set[str] = set()
    if out_path.exists():
        try:
            old_df = pd.read_csv(out_path)
            if "database_name" in old_df.columns and set(old_df["database_name"]).issubset(
                set(config.SELECTED_CTU_DATABASES)
            ):
                existing_rows = old_df.to_dict("records")
                completed_qids = set(old_df["query_id"].astype(str).unique())
                print(f"Resuming from {len(existing_rows)} existing rows ({len(completed_qids)} queries)...")
        except Exception:
            pass

    db = MySQLAdapter()
    db.connect()
    stats_cache = StatsCache(db)

    total_queries = len(target_df)
    t_start = time.perf_counter()

    try:
        for q_idx, row in target_df.iterrows():
            qid = str(row["query_id"])
            if qid in completed_qids:
                continue

            db_name = str(row["database_name"])
            tbl_name = str(row["table_name"])
            sql = str(row["sql"])
            split_group = str(row["split_group"])
            category = str(row["category"])

            db.use_database(db_name)
            # Ensure clean state (no leftover secondary indexes on target table)
            for leftover_idx in db.get_non_primary_indexes(tbl_name):
                db.drop_index_if_exists(tbl_name, leftover_idx)

            t_stats = stats_cache.get_table_stats(tbl_name)
            parsed = parse_query(
                sql,
                target_table=t_stats.table_name,
                schema_columns=t_stats.canonical_columns,
            )
            candidates = generate_candidates(
                parsed,
                max_candidates=4,
                excluded_columns=t_stats.excluded_columns(),
                single_pk_column=t_stats.single_pk_column(),
            )
            if not candidates:
                continue

            # Measure unindexed baseline (5 runs, discard 1 warm-up)
            try:
                time_before, rows_before, _ = db.timed_execute(sql, runs=runs, warmup=1)
                base_exp = stats_cache.get_baseline_explain(sql, t_stats.table_name)
            except Exception as exc:
                print(f"  [SKIP QUERY] {qid} failed baseline execution: {exc}")
                continue

            for cand in candidates:
                try:
                    feats = extract_features(parsed, cand, stats_cache, baseline_explain=base_exp)
                    # Create candidate index
                    db.execute_ddl(cand.ddl)
                    time_after, rows_after, _ = db.timed_execute(sql, runs=runs, warmup=1)
                except Exception as exc:
                    print(f"    [SKIP CAND] {cand.index_name}: {exc}")
                    db.drop_index_if_exists(cand.table, cand.index_name)
                    continue
                finally:
                    db.drop_index_if_exists(cand.table, cand.index_name)

                speedup = max(0.05, time_before / max(time_after, 1e-6))
                log_speedup = math.log(speedup)

                rec: Dict[str, Any] = {
                    "query_id": qid,
                    "database_name": db_name,
                    "table_name": t_stats.table_name,
                    "category": category,
                    "split_group": split_group,
                    "template_id": f"{db_name}:{category}",
                    "template_sql": parsed.template_sql,
                    "sql": sql,
                    "index_name": cand.index_name,
                    "candidate_columns": cand.columns_display,
                    "ddl": cand.ddl,
                    "time_before_ms": round(time_before * 1000.0, 4),
                    "time_after_ms": round(time_after * 1000.0, 4),
                    "rows_before": int(rows_before),
                    "rows_after": int(rows_after),
                    "speedup": round(speedup, 4),
                    "log_speedup": round(log_speedup, 4),
                }
                for fname in FEATURE_NAMES:
                    rec[fname] = float(feats[fname])
                existing_rows.append(rec)

            completed_qids.add(qid)
            if len(completed_qids) % 5 == 0 or len(completed_qids) == total_queries:
                out_df = pd.DataFrame(existing_rows)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_df.to_csv(out_path, index=False)
                elapsed = time.perf_counter() - t_start
                print(
                    f"  [LABELING PROGRESS] {len(completed_qids)}/{total_queries} queries "
                    f"({len(existing_rows)} pairs) | last={db_name}.{tbl_name} | elapsed={elapsed:.1f}s"
                )
    finally:
        db.close()

    final_df = pd.DataFrame(existing_rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(out_path, index=False)
    print(f"Finished labeling: {len(final_df)} pairs saved to {out_path}")
    return final_df


if __name__ == "__main__":
    label_workload()
