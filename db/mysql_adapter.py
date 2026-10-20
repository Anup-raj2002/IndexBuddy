"""
Abstract database adapter — concrete implementations for MySQL (and later PostgreSQL).
"""
from __future__ import annotations

import time
import statistics
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import mysql.connector
from mysql.connector import Error as MySQLError

import config


# ── Abstract base ───────────────────────────────────────────────


class DBAdapter(ABC):
    """Base class for database adapters.  Sub-class for each RDBMS."""

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def execute(self, sql: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]: ...

    @abstractmethod
    def execute_scalar(self, sql: str, params: Optional[tuple] = None) -> Any: ...

    @abstractmethod
    def execute_ddl(self, sql: str) -> None: ...

    @abstractmethod
    def explain(self, sql: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]: ...

    @abstractmethod
    def explain_analyze(self, sql: str, params: Optional[tuple] = None) -> str: ...

    @abstractmethod
    def table_exists(self, table: str) -> bool: ...

    @abstractmethod
    def row_count(self, table: str) -> int: ...

    # ── Shared helpers (use the abstract primitives) ────────────

    def timed_execute(self, sql: str, params: Optional[tuple] = None,
                      runs: int = 5, warmup: int = 1) -> Tuple[float, int]:
        """Execute *sql* multiple times and return (median_seconds, rows_examined).

        The first *warmup* runs are discarded.  *rows_examined* comes from
        EXPLAIN of the query (estimated rows).
        """
        times: list[float] = []
        for i in range(runs):
            start = time.perf_counter()
            self.execute(sql, params)
            elapsed = time.perf_counter() - start
            if i >= warmup:
                times.append(elapsed)
        median_time = statistics.median(times) if times else 0.0

        # rows examined via EXPLAIN
        rows_examined = 0
        try:
            plan = self.explain(sql, params)
            for row in plan:
                rows_examined += int(row.get("rows", 0) or 0)
        except Exception:
            pass

        return median_time, rows_examined


# ── MySQL adapter ───────────────────────────────────────────────


class MySQLAdapter(DBAdapter):
    """Concrete adapter for MySQL 8.x using mysql-connector-python."""

    def __init__(self, host: Optional[str] = None, port: Optional[int] = None,
                 user: Optional[str] = None, password: Optional[str] = None,
                 database: Optional[str] = None) -> None:
        self.host = host or config.MYSQL_HOST
        self.port = port or config.MYSQL_PORT
        self.user = user or config.MYSQL_USER
        self.password = password or config.MYSQL_PASSWORD
        self.database = database or config.MYSQL_DB
        self._conn: Optional[mysql.connector.MySQLConnection] = None

    # ── connection management ───────────────────────────────────

    def connect(self) -> None:
        if self._conn and self._conn.is_connected():
            return
        self._conn = mysql.connector.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            database=self.database,
            autocommit=True,
            connection_timeout=10,
        )

    def close(self) -> None:
        if self._conn and self._conn.is_connected():
            self._conn.close()
            self._conn = None

    @property
    def conn(self) -> mysql.connector.MySQLConnection:
        if self._conn is None or not self._conn.is_connected():
            self.connect()
        assert self._conn is not None
        return self._conn

    # ── primitives ──────────────────────────────────────────────

    def execute(self, sql: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        cur = self.conn.cursor(dictionary=True)
        try:
            cur.execute(sql, params)
            if cur.with_rows:
                return cur.fetchall()
            return []
        finally:
            cur.close()

    def execute_scalar(self, sql: str, params: Optional[tuple] = None) -> Any:
        rows = self.execute(sql, params)
        if rows:
            return list(rows[0].values())[0]
        return None

    def execute_ddl(self, sql: str) -> None:
        cur = self.conn.cursor()
        try:
            cur.execute(sql)
            if cur.with_rows:
                cur.fetchall()
        finally:
            cur.close()

    def explain(self, sql: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        return self.execute(f"EXPLAIN {sql}", params)

    def explain_analyze(self, sql: str, params: Optional[tuple] = None) -> str:
        """Return the tree-format output of EXPLAIN ANALYZE (MySQL 8.0.18+)."""
        rows = self.execute(f"EXPLAIN ANALYZE {sql}", params)
        if rows:
            # MySQL returns a single row with one column named "EXPLAIN"
            first = rows[0]
            return str(list(first.values())[0])
        return ""

    def table_exists(self, table: str) -> bool:
        rows = self.execute(
            "SELECT COUNT(*) AS cnt FROM information_schema.TABLES "
            "WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s",
            (self.database, table),
        )
        return bool(rows and rows[0]["cnt"] > 0)

    def row_count(self, table: str) -> int:
        val = self.execute_scalar(f"SELECT COUNT(*) FROM `{table}`")
        return int(val) if val is not None else 0

    def index_exists(self, table: str, index_name: str) -> bool:
        rows = self.execute(
            "SELECT COUNT(*) AS cnt FROM information_schema.STATISTICS "
            "WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s AND INDEX_NAME = %s",
            (self.database, table, index_name),
        )
        return bool(rows and rows[0]["cnt"] > 0)

    def drop_index_if_exists(self, table: str, index_name: str) -> None:
        if self.index_exists(table, index_name):
            self.execute_ddl(f"DROP INDEX `{index_name}` ON `{table}`")

    def get_non_primary_indexes(self, table: str) -> List[str]:
        """Return names of all non-PRIMARY indexes on *table*."""
        rows = self.execute(
            "SELECT DISTINCT INDEX_NAME FROM information_schema.STATISTICS "
            "WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s AND INDEX_NAME != 'PRIMARY'",
            (self.database, table),
        )
        return [r["INDEX_NAME"] for r in rows]

    def drop_all_secondary_indexes(self, table: str) -> None:
        for idx in self.get_non_primary_indexes(table):
            self.execute_ddl(f"DROP INDEX `{idx}` ON `{table}`")

    def column_distinct_count(self, table: str, column: str) -> int:
        val = self.execute_scalar(f"SELECT COUNT(DISTINCT `{column}`) FROM `{table}`")
        return int(val) if val is not None else 0

    def column_null_fraction(self, table: str, column: str) -> float:
        total = self.row_count(table)
        if total == 0:
            return 0.0
        nulls = self.execute_scalar(
            f"SELECT COUNT(*) FROM `{table}` WHERE `{column}` IS NULL"
        )
        return float(nulls) / total if nulls else 0.0

    def analyze_table(self, table: str) -> List[Dict[str, Any]]:
        return self.execute(f"ANALYZE TABLE `{table}`")

    def flush_status(self) -> None:
        """Reset session status counters (useful before measuring)."""
        try:
            self.execute_ddl("FLUSH STATUS")
        except MySQLError:
            pass  # requires RELOAD privilege

    def is_connected(self) -> bool:
        try:
            if self._conn and self._conn.is_connected():
                self._conn.ping(reconnect=False)
                return True
        except Exception:
            pass
        return False
