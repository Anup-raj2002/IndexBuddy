"""
db/mysql_adapter.py
───────────────────
Abstract database adapter (DBAdapter) and concrete MySQL 8.x implementation
(MySQLAdapter) with dynamic schema introspection across any database and table.
Zero hardcoded table or column names.
"""
from __future__ import annotations

import statistics
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import mysql.connector
from mysql.connector import Error as MySQLError

import config


SYSTEM_SCHEMAS = {
    "information_schema",
    "mysql",
    "performance_schema",
    "sys",
}


class DBAdapter(ABC):
    """Base class for database adapters. Sub-class for each RDBMS (MySQL, PostgreSQL)."""

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
    def list_databases(self) -> List[str]: ...

    @abstractmethod
    def list_tables(self, database: Optional[str] = None) -> List[Dict[str, Any]]: ...

    @abstractmethod
    def table_exists(self, table: str) -> bool: ...

    @abstractmethod
    def row_count(self, table: str) -> int: ...

    def timed_execute(
        self,
        sql: str,
        params: Optional[tuple] = None,
        runs: int = 5,
        warmup: int = 1,
    ) -> Tuple[float, int, List[float]]:
        """Execute *sql* multiple times and return (median_seconds, rows_examined, run_times_sec).

        The first *warmup* runs are discarded. *rows_examined* comes from EXPLAIN.
        """
        times: List[float] = []
        for i in range(runs):
            start = time.perf_counter()
            self.execute(sql, params)
            elapsed = time.perf_counter() - start
            if i >= warmup:
                times.append(elapsed)
        median_time = statistics.median(times) if times else 0.0

        rows_examined = 0
        try:
            plan = self.explain(sql, params)
            for row in plan:
                rows_examined += int(row.get("rows", 0) or 0)
        except Exception:
            pass

        return median_time, rows_examined, times


class MySQLAdapter(DBAdapter):
    """Concrete adapter for MySQL 8.x using mysql-connector-python."""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
    ) -> None:
        self.host = host or config.MYSQL_HOST
        self.port = int(port or config.MYSQL_PORT)
        self.user = user or config.MYSQL_USER
        self.password = password if password is not None else config.MYSQL_PASSWORD
        self.database = database if database is not None else config.MYSQL_DB
        self._conn: Optional[mysql.connector.MySQLConnection] = None

    def connect(self) -> None:
        if self._conn and self._conn.is_connected():
            return
        kwargs: Dict[str, Any] = {
            "host": self.host,
            "port": self.port,
            "user": self.user,
            "password": self.password,
            "autocommit": True,
            "connection_timeout": 15,
            "use_pure": True,
        }
        if self.database:
            kwargs["database"] = self.database
        self._conn = mysql.connector.connect(**kwargs)

    def use_database(self, database: str) -> None:
        """Switch active database on the connection."""
        self.database = database
        if self._conn and self._conn.is_connected():
            self.execute_ddl(f"USE `{database}`")
        else:
            self.connect()

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
            first = rows[0]
            return str(list(first.values())[0])
        return ""

    def list_databases(self) -> List[str]:
        """Return all user databases available on the MySQL server."""
        rows = self.execute("SHOW DATABASES")
        dbs: List[str] = []
        for r in rows:
            name = str(list(r.values())[0])
            if name.lower() not in SYSTEM_SCHEMAS and name.lower() != "indexbuddy":
                dbs.append(name)
        return sorted(dbs, key=lambda s: s.lower())

    def list_tables(self, database: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return tables and estimated row counts for *database* via information_schema."""
        db_name = database or self.database
        if not db_name:
            return []
        rows = self.execute(
            "SELECT TABLE_NAME AS table_name, "
            "COALESCE(TABLE_ROWS, 0) AS est_rows, "
            "ROUND(COALESCE(DATA_LENGTH + INDEX_LENGTH, 0) / 1048576, 2) AS size_mb "
            "FROM information_schema.TABLES "
            "WHERE TABLE_SCHEMA = %s AND TABLE_TYPE = 'BASE TABLE' "
            "ORDER BY TABLE_ROWS DESC, TABLE_NAME ASC",
            (db_name,),
        )
        return rows

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

    def get_primary_key_columns(self, table: str, database: Optional[str] = None) -> List[str]:
        db_name = database or self.database
        rows = self.execute(
            "SELECT COLUMN_NAME FROM information_schema.STATISTICS "
            "WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s AND INDEX_NAME = 'PRIMARY' "
            "ORDER BY SEQ_IN_INDEX",
            (db_name, table),
        )
        return [str(r["COLUMN_NAME"]) for r in rows]

    def get_foreign_keys(self, database: Optional[str] = None) -> List[Dict[str, str]]:
        """Discover foreign key relationships in *database* from information_schema."""
        db_name = database or self.database
        rows = self.execute(
            "SELECT TABLE_NAME AS table_name, COLUMN_NAME AS column_name, "
            "REFERENCED_TABLE_NAME AS ref_table_name, "
            "REFERENCED_COLUMN_NAME AS ref_column_name "
            "FROM information_schema.KEY_COLUMN_USAGE "
            "WHERE TABLE_SCHEMA = %s AND REFERENCED_TABLE_NAME IS NOT NULL",
            (db_name,),
        )
        return [
            {
                "table_name": str(r["table_name"]),
                "column_name": str(r["column_name"]),
                "ref_table_name": str(r["ref_table_name"]),
                "ref_column_name": str(r["ref_column_name"]),
            }
            for r in rows
        ]

    def get_existing_indexes(self, table: str, database: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return detailed index metadata on *table*."""
        db_name = database or self.database
        rows = self.execute(
            "SELECT INDEX_NAME AS index_name, NON_UNIQUE AS non_unique, "
            "GROUP_CONCAT(COLUMN_NAME ORDER BY SEQ_IN_INDEX SEPARATOR ', ') AS columns, "
            "MAX(CARDINALITY) AS cardinality, INDEX_TYPE AS index_type "
            "FROM information_schema.STATISTICS "
            "WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s "
            "GROUP BY INDEX_NAME, NON_UNIQUE, INDEX_TYPE "
            "ORDER BY INDEX_NAME",
            (db_name, table),
        )
        return rows

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
        return [str(r["INDEX_NAME"]) for r in rows]

    def get_indexbuddy_indexes(self, table: str) -> List[str]:
        """Return only indexes created by IndexBuddy (prefixed with 'idx_ib_')."""
        return [idx for idx in self.get_non_primary_indexes(table) if idx.startswith("idx_ib_")]

    def drop_indexbuddy_indexes(self, table: str) -> List[str]:
        """Drop only indexes that IndexBuddy created on *table*."""
        dropped: List[str] = []
        for idx in self.get_indexbuddy_indexes(table):
            self.execute_ddl(f"DROP INDEX `{idx}` ON `{table}`")
            dropped.append(idx)
        return dropped

    def analyze_table(self, table: str) -> List[Dict[str, Any]]:
        return self.execute(f"ANALYZE TABLE `{table}`")

    def is_connected(self) -> bool:
        try:
            if self._conn and self._conn.is_connected():
                self._conn.ping(reconnect=False)
                return True
        except Exception:
            pass
        return False
