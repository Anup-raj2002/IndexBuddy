"""
data/select_ctu_datasets.py
───────────────────────────
Connects to the public CTU Prague Relational Learning Repository
(relational.fel.cvut.cz:3306, user=guest, password=ctu-relational),
queries ``meta.information`` for databases marked Type = Real
(``is_artificial = 0``) where the largest table has between 50,000 and
5,000,000 rows, and prints the selected diverse-domain databases.
"""
from __future__ import annotations

from typing import Any, Dict, List

import mysql.connector

import config


def fetch_selected_ctu_metadata() -> List[Dict[str, Any]]:
    """Query live CTU Prague metadata for our selected real databases."""
    conn = mysql.connector.connect(
        host=config.CTU_HOST,
        port=config.CTU_PORT,
        user=config.CTU_USER,
        password=config.CTU_PASSWORD,
        connection_timeout=20,
        use_pure=True,
    )
    cur = conn.cursor(dictionary=True)
    results: List[Dict[str, Any]] = []
    try:
        for db_name in config.SELECTED_CTU_DATABASES:
            cur.execute(
                "SELECT database_name, dataset_name, domain, is_artificial, "
                "table_count, row_max, row_count, database_size, "
                "primary_key_count, foreign_key_count "
                "FROM meta.information WHERE database_name = %s",
                (db_name,),
            )
            info = cur.fetchone()
            if not info:
                continue

            cur.execute(
                "SELECT TABLE_NAME, TABLE_ROWS, "
                "ROUND((DATA_LENGTH + INDEX_LENGTH)/1024/1024, 2) AS size_mb "
                "FROM information_schema.TABLES "
                "WHERE TABLE_SCHEMA = %s AND TABLE_TYPE = 'BASE TABLE' "
                "ORDER BY TABLE_ROWS DESC",
                (db_name,),
            )
            tables = cur.fetchall()
            largest_tbl = tables[0]["TABLE_NAME"] if tables else ""
            split_role = (
                "TRAIN"
                if db_name in config.TRAIN_DATABASES
                else "HELD-OUT TEST / DEMO"
            )
            results.append(
                {
                    "database_name": str(info["database_name"]),
                    "domain": str(info["domain"]),
                    "type": "Real" if int(info["is_artificial"]) == 0 else "Synthetic",
                    "table_count": int(info["table_count"]),
                    "largest_table": str(largest_tbl),
                    "largest_table_rows": int(info["row_max"]),
                    "total_rows": int(info["row_count"]),
                    "size_mb": float(info["database_size"]),
                    "split_role": split_role,
                    "tables": [
                        {
                            "name": str(t["TABLE_NAME"]),
                            "rows": int(t["TABLE_ROWS"] or 0),
                            "size_mb": float(t["size_mb"] or 0.0),
                        }
                        for t in tables
                    ],
                }
            )
    finally:
        cur.close()
        conn.close()
    return results


def print_selected_datasets() -> None:
    rows = fetch_selected_ctu_metadata()
    print("=" * 105)
    print(
        f"{'Database':<15} | {'Domain':<14} | {'Type':<6} | {'Tables':>6} | "
        f"{'Largest Table':<16} | {'Max Rows':>10} | {'Total Rows':>10} | {'Size (MB)':>9} | {'Split Role'}"
    )
    print("-" * 105)
    for r in rows:
        print(
            f"{r['database_name']:<15} | {r['domain']:<14} | {r['type']:<6} | "
            f"{r['table_count']:>6d} | {r['largest_table']:<16} | "
            f"{r['largest_table_rows']:>10,d} | {r['total_rows']:>10,d} | "
            f"{r['size_mb']:>8.1f}M | {r['split_role']}"
        )
    print("=" * 105)


if __name__ == "__main__":
    print_selected_datasets()
