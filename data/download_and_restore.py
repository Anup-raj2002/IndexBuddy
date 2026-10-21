"""
data/download_and_restore.py
────────────────────────────
Dumps each selected real database from the CTU Prague Relational Learning
Repository (relational.fel.cvut.cz:3306) once into ``data/raw/<db>.sql``
(with polite pauses, skipping if already dumped), restores each into the
local MySQL 8.x server as its own database, saves the original index list
to ``data/original_indexes.json``, drops all secondary indexes (keeping only
PRIMARY KEYs) so candidates are measured from a clean state, and runs
``ANALYZE TABLE`` on every restored table.

 Also drops the legacy synthetic ``indexbuddy`` database if present.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List

import mysql.connector

import config


def _find_mysql_binary(name: str) -> str:
    """Locate mysql.exe or mysqldump.exe on Windows or PATH."""
    found = shutil.which(name)
    if found:
        return found
    candidates = [
        Path(rf"C:\Program Files\MySQL\MySQL Server 8.0\bin\{name}.exe"),
        Path(rf"C:\Program Files\MySQL\MySQL Server 8.4\bin\{name}.exe"),
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    raise FileNotFoundError(f"Could not find {name} executable.")


def dump_ctu_database(db_name: str, out_dir: Path = config.RAW_DUMP_DIR) -> Path:
    """Dump *db_name* from CTU Prague into ``data/raw/<db_name>.sql`` if not already present."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_file = out_dir / f"{db_name}.sql"
    if dump_file.exists() and dump_file.stat().st_size > 10_000:
        size_mb = dump_file.stat().st_size / (1024 * 1024)
        print(f"  [SKIP DUMP] {dump_file} already exists ({size_mb:.2f} MB)")
        return dump_file

    mysqldump_bin = _find_mysql_binary("mysqldump")
    print(f"  [DUMPING] {db_name} from {config.CTU_HOST}:{config.CTU_PORT} -> {dump_file} ...")
    t0 = time.perf_counter()
    cmd = [
        mysqldump_bin,
        f"--host={config.CTU_HOST}",
        f"--port={config.CTU_PORT}",
        f"--user={config.CTU_USER}",
        f"--password={config.CTU_PASSWORD}",
        "--single-transaction",
        "--quick",
        "--skip-lock-tables",
        "--no-tablespaces",
        "--set-gtid-purged=OFF",
        "--column-statistics=0",
        db_name,
    ]
    # Check if --column-statistics=0 is supported by this mysqldump build
    help_out = subprocess.run(
        [mysqldump_bin, "--help"], capture_output=True, text=True, check=False
    ).stdout
    if "column-statistics" not in help_out:
        cmd = [c for c in cmd if not c.startswith("--column-statistics")]

    with open(dump_file, "wb") as f_out:
        proc = subprocess.run(cmd, stdout=f_out, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        err_msg = proc.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"mysqldump failed for {db_name}: {err_msg}")

    elapsed = time.perf_counter() - t0
    size_mb = dump_file.stat().st_size / (1024 * 1024)
    print(f"  [DUMPED] {db_name}: {size_mb:.2f} MB in {elapsed:.1f}s")
    time.sleep(2.0)  # polite pause between dumps against public server
    return dump_file


def restore_to_local_mysql(db_name: str, dump_file: Path) -> bool:
    """Restore *dump_file* into local MySQL as database *db_name* if not already populated."""
    conn = mysql.connector.connect(
        host=config.MYSQL_HOST,
        port=config.MYSQL_PORT,
        user=config.MYSQL_USER,
        password=config.MYSQL_PASSWORD,
        autocommit=True,
        use_pure=True,
    )
    cur = conn.cursor(dictionary=True)
    try:
        # Check if database already exists and has tables with rows
        cur.execute(
            "SELECT COUNT(*) AS tbl_cnt, COALESCE(SUM(TABLE_ROWS), 0) AS row_sum "
            "FROM information_schema.TABLES "
            "WHERE TABLE_SCHEMA = %s AND TABLE_TYPE = 'BASE TABLE'",
            (db_name,),
        )
        stat = cur.fetchone()
        if stat and int(stat["tbl_cnt"]) > 0 and int(stat["row_sum"]) > 10_000:
            print(
                f"  [SKIP RESTORE] Local MySQL `{db_name}` already has "
                f"{stat['tbl_cnt']} tables (~{int(stat['row_sum']):,} rows)"
            )
            return False

        cur.execute(f"CREATE DATABASE IF NOT EXISTS `{db_name}` DEFAULT CHARACTER SET utf8mb4")
    finally:
        cur.close()
        conn.close()

    mysql_bin = _find_mysql_binary("mysql")
    print(f"  [RESTORING] {dump_file.name} -> local MySQL database `{db_name}` ...")
    t0 = time.perf_counter()
    cmd = [
        mysql_bin,
        f"--host={config.MYSQL_HOST}",
        f"--port={config.MYSQL_PORT}",
        f"--user={config.MYSQL_USER}",
        f"--password={config.MYSQL_PASSWORD}",
        "--init-command=SET SESSION sql_mode=''; SET FOREIGN_KEY_CHECKS=0;",
        db_name,
    ]
    with open(dump_file, "rb") as f_in:
        proc = subprocess.run(cmd, stdin=f_in, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        err_msg = proc.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"mysql restore failed for {db_name}: {err_msg}")

    elapsed = time.perf_counter() - t0
    print(f"  [RESTORED] `{db_name}` in {elapsed:.1f}s")
    return True


def strip_secondary_indexes_and_analyze(
    db_name: str,
    original_indexes_store: Dict[str, Any],
) -> Dict[str, Any]:
    """Save original foreign keys and secondary indexes, drop them so only PRIMARY KEYs remain, and run ANALYZE TABLE."""
    conn = mysql.connector.connect(
        host=config.MYSQL_HOST,
        port=config.MYSQL_PORT,
        user=config.MYSQL_USER,
        password=config.MYSQL_PASSWORD,
        database=db_name,
        autocommit=True,
        use_pure=True,
    )
    cur = conn.cursor(dictionary=True)
    db_record: Dict[str, Any] = original_indexes_store.get(
        db_name, {"foreign_keys": [], "secondary_indexes": [], "tables_analyzed": []}
    )
    try:
        cur.execute("SET FOREIGN_KEY_CHECKS = 0")

        # 1. Record & drop foreign key constraints (so secondary FK indexes can be dropped cleanly)
        cur.execute(
            "SELECT TABLE_NAME, CONSTRAINT_NAME, COLUMN_NAME, "
            "REFERENCED_TABLE_NAME, REFERENCED_COLUMN_NAME "
            "FROM information_schema.KEY_COLUMN_USAGE "
            "WHERE TABLE_SCHEMA = %s AND REFERENCED_TABLE_NAME IS NOT NULL",
            (db_name,),
        )
        fks = cur.fetchall()
        seen_fk_constraints = set()
        for fk in fks:
            t_name = str(fk["TABLE_NAME"])
            c_name = str(fk["CONSTRAINT_NAME"])
            fk_entry = {
                "table_name": t_name,
                "constraint_name": c_name,
                "column_name": str(fk["COLUMN_NAME"]),
                "ref_table_name": str(fk["REFERENCED_TABLE_NAME"]),
                "ref_column_name": str(fk["REFERENCED_COLUMN_NAME"]),
            }
            if fk_entry not in db_record["foreign_keys"]:
                db_record["foreign_keys"].append(fk_entry)
            if (t_name, c_name) not in seen_fk_constraints:
                seen_fk_constraints.add((t_name, c_name))
                try:
                    cur.execute(f"ALTER TABLE `{t_name}` DROP FOREIGN KEY `{c_name}`")
                except Exception:
                    pass

        # 2. Record & drop all non-PRIMARY indexes
        cur.execute(
            "SELECT TABLE_NAME, INDEX_NAME, NON_UNIQUE, INDEX_TYPE, "
            "GROUP_CONCAT(COLUMN_NAME ORDER BY SEQ_IN_INDEX SEPARATOR ',') AS cols "
            "FROM information_schema.STATISTICS "
            "WHERE TABLE_SCHEMA = %s AND INDEX_NAME != 'PRIMARY' "
            "GROUP BY TABLE_NAME, INDEX_NAME, NON_UNIQUE, INDEX_TYPE",
            (db_name,),
        )
        sec_idxs = cur.fetchall()
        dropped_count = 0
        for row in sec_idxs:
            t_name = str(row["TABLE_NAME"])
            idx_name = str(row["INDEX_NAME"])
            idx_entry = {
                "table_name": t_name,
                "index_name": idx_name,
                "non_unique": int(row["NON_UNIQUE"]),
                "index_type": str(row["INDEX_TYPE"]),
                "columns": str(row["cols"]).split(","),
            }
            if idx_entry not in db_record["secondary_indexes"]:
                db_record["secondary_indexes"].append(idx_entry)
            try:
                cur.execute(f"DROP INDEX `{idx_name}` ON `{t_name}`")
                dropped_count += 1
            except Exception as exc:
                print(f"    [WARN] Could not drop index `{idx_name}` on `{t_name}`: {exc}")

        # 3. Run ANALYZE TABLE on all tables
        cur.execute(
            "SELECT TABLE_NAME FROM information_schema.TABLES "
            "WHERE TABLE_SCHEMA = %s AND TABLE_TYPE = 'BASE TABLE'",
            (db_name,),
        )
        tables = [str(r["TABLE_NAME"]) for r in cur.fetchall()]
        for t_name in tables:
            cur.execute(f"ANALYZE TABLE `{t_name}`")
            cur.fetchall()
        db_record["tables_analyzed"] = tables

        cur.execute("SET FOREIGN_KEY_CHECKS = 1")
        print(
            f"  [CLEAN STATE] `{db_name}`: dropped {len(seen_fk_constraints)} FK constraints "
            f"& {dropped_count} secondary indexes; analyzed {len(tables)} tables."
        )
    finally:
        cur.close()
        conn.close()

    original_indexes_store[db_name] = db_record
    return db_record


def drop_legacy_synthetic_db() -> None:
    """Drop the legacy synthetic `indexbuddy` database so only real databases exist."""
    conn = mysql.connector.connect(
        host=config.MYSQL_HOST,
        port=config.MYSQL_PORT,
        user=config.MYSQL_USER,
        password=config.MYSQL_PASSWORD,
        autocommit=True,
        use_pure=True,
    )
    cur = conn.cursor()
    try:
        cur.execute("DROP DATABASE IF EXISTS `indexbuddy`")
        print("  [CLEANUP] Dropped legacy synthetic database `indexbuddy` from local MySQL.")
    finally:
        cur.close()
        conn.close()


def run_all() -> None:
    print("=== Stage 3: Dump from CTU Prague, Restore to Local MySQL & Clean-State Preparation ===")
    drop_legacy_synthetic_db()

    orig_path = config.ORIGINAL_INDEXES_PATH
    if orig_path.exists():
        try:
            orig_store = json.loads(orig_path.read_text(encoding="utf-8"))
        except Exception:
            orig_store = {}
    else:
        orig_store = {}

    for db_name in config.SELECTED_CTU_DATABASES:
        print(f"\n--- Processing Database: `{db_name}` ---")
        dump_file = dump_ctu_database(db_name)
        restore_to_local_mysql(db_name, dump_file)
        strip_secondary_indexes_and_analyze(db_name, orig_store)
        orig_path.parent.mkdir(parents=True, exist_ok=True)
        orig_path.write_text(json.dumps(orig_store, indent=2), encoding="utf-8")

    # Verify final state across all selected databases on local MySQL
    print("\n=== Final Verification on Local MySQL Server ===")
    conn = mysql.connector.connect(
        host=config.MYSQL_HOST,
        port=config.MYSQL_PORT,
        user=config.MYSQL_USER,
        password=config.MYSQL_PASSWORD,
        use_pure=True,
    )
    cur = conn.cursor(dictionary=True)
    for db_name in config.SELECTED_CTU_DATABASES:
        cur.execute(
            "SELECT COUNT(*) AS num_tables, MAX(TABLE_ROWS) AS max_rows, SUM(TABLE_ROWS) AS sum_rows "
            "FROM information_schema.TABLES WHERE TABLE_SCHEMA = %s AND TABLE_TYPE = 'BASE TABLE'",
            (db_name,),
        )
        t_stat = cur.fetchone()
        cur.execute(
            "SELECT COUNT(DISTINCT INDEX_NAME) AS sec_idx "
            "FROM information_schema.STATISTICS WHERE TABLE_SCHEMA = %s AND INDEX_NAME != 'PRIMARY'",
            (db_name,),
        )
        i_stat = cur.fetchone()
        print(
            f"  Local DB `{db_name:<14}` | tables={int(t_stat['num_tables']):>2d} | "
            f"max_rows={int(t_stat['max_rows'] or 0):>10,d} | "
            f"total_rows={int(t_stat['sum_rows'] or 0):>10,d} | "
            f"secondary_indexes={int(i_stat['sec_idx'] or 0)}"
        )
    cur.close()
    conn.close()
    print(f"Saved original index & FK metadata to {orig_path}")


if __name__ == "__main__":
    run_all()
