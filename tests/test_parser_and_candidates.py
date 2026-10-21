"""
tests/test_parser_and_candidates.py
───────────────────────────────────
Unit tests for SQL AST parsing, candidate generation, and schema-agnostic
feature extraction across at least two distinct database schemas (Formula 1
racing schema and Banking transactions schema).
"""
from __future__ import annotations

import pytest

from candidates.generator import generate_candidates
from features.extract import (
    BaselineExplainInfo,
    ColumnStats,
    FEATURE_NAMES,
    StatsCache,
    TableStats,
    extract_features,
)
from parsing.query_parser import extract_query_template, parse_query, validate_select_only


class DummyStatsCache(StatsCache):
    """In-memory mock of StatsCache for fast unit testing across multiple schemas."""

    def __init__(self, tables: dict[str, TableStats]) -> None:
        self._tables_map = {k.lower(): v for k, v in tables.items()}
        self._explain_cache = {}

    def get_table_stats(self, table: str) -> TableStats:
        return self._tables_map[table.lower()]

    def get_baseline_explain(self, sql: str, primary_table: str) -> BaselineExplainInfo:
        t_stats = self.get_table_stats(primary_table)
        return BaselineExplainInfo(
            access_type="ALL",
            access_type_code=0,
            estimated_rows=t_stats.table_rows,
            filtered_pct=10.0,
            uses_filesort=1 if "ORDER BY" in sql.upper() else 0,
            uses_temporary=1 if "GROUP BY" in sql.upper() else 0,
            raw_rows=[],
        )


def _build_f1_laptimes_schema() -> TableStats:
    """Schema 1: Formula 1 lapTimes table (419,328 rows)."""
    cols = {
        "raceId": ColumnStats(
            column_name="raceId",
            data_type="int",
            data_type_code=1,
            byte_width=4,
            is_nullable=False,
            is_primary_key=True,
            is_indexable=True,
            cardinality=976,
            distinct_ratio=976 / 419328,
            null_fraction=0.0,
            min_value=1,
            max_value=988,
        ),
        "driverId": ColumnStats(
            column_name="driverId",
            data_type="int",
            data_type_code=1,
            byte_width=4,
            is_nullable=False,
            is_primary_key=True,
            is_indexable=True,
            cardinality=840,
            distinct_ratio=840 / 419328,
            null_fraction=0.0,
            min_value=1,
            max_value=843,
        ),
        "lap": ColumnStats(
            column_name="lap",
            data_type="int",
            data_type_code=1,
            byte_width=4,
            is_nullable=False,
            is_primary_key=True,
            is_indexable=True,
            cardinality=78,
            distinct_ratio=78 / 419328,
            null_fraction=0.0,
            min_value=1,
            max_value=78,
        ),
        "position": ColumnStats(
            column_name="position",
            data_type="int",
            data_type_code=1,
            byte_width=4,
            is_nullable=True,
            is_primary_key=False,
            is_indexable=True,
            cardinality=24,
            distinct_ratio=24 / 419328,
            null_fraction=0.0,
            min_value=1,
            max_value=24,
        ),
        "milliseconds": ColumnStats(
            column_name="milliseconds",
            data_type="int",
            data_type_code=1,
            byte_width=4,
            is_nullable=True,
            is_primary_key=False,
            is_indexable=True,
            cardinality=72540,
            distinct_ratio=72540 / 419328,
            null_fraction=0.0,
            min_value=55000,
            max_value=350000,
        ),
    }
    return TableStats(
        database_name="ErgastF1",
        table_name="lapTimes",
        table_rows=419328,
        avg_row_length=48,
        data_length_mb=39.5,
        primary_key_columns=["raceId", "driverId", "lap"],
        columns=cols,
        canonical_columns={k.lower(): k for k in cols},
    )


def _build_financial_trans_schema() -> TableStats:
    """Schema 2: Financial banking trans table (1,056,320 rows)."""
    cols = {
        "trans_id": ColumnStats(
            column_name="trans_id",
            data_type="int",
            data_type_code=1,
            byte_width=4,
            is_nullable=False,
            is_primary_key=True,
            is_indexable=True,
            cardinality=1056320,
            distinct_ratio=1.0,
            null_fraction=0.0,
            min_value=1,
            max_value=3682987,
        ),
        "account_id": ColumnStats(
            column_name="account_id",
            data_type="int",
            data_type_code=1,
            byte_width=4,
            is_nullable=False,
            is_primary_key=False,
            is_indexable=True,
            cardinality=4500,
            distinct_ratio=4500 / 1056320,
            null_fraction=0.0,
            min_value=1,
            max_value=11382,
        ),
        "date": ColumnStats(
            column_name="date",
            data_type="date",
            data_type_code=3,
            byte_width=3,
            is_nullable=False,
            is_primary_key=False,
            is_indexable=True,
            cardinality=2191,
            distinct_ratio=2191 / 1056320,
            null_fraction=0.0,
            min_value="1993-01-01",
            max_value="1998-12-31",
        ),
        "type": ColumnStats(
            column_name="type",
            data_type="varchar",
            data_type_code=4,
            byte_width=10,
            is_nullable=False,
            is_primary_key=False,
            is_indexable=True,
            cardinality=3,
            distinct_ratio=3 / 1056320,
            null_fraction=0.0,
            value_frequencies={"vydej": 0.60, "prijem": 0.38, "vyber": 0.02},
        ),
        "amount": ColumnStats(
            column_name="amount",
            data_type="int",
            data_type_code=1,
            byte_width=4,
            is_nullable=False,
            is_primary_key=False,
            is_indexable=True,
            cardinality=40400,
            distinct_ratio=40400 / 1056320,
            null_fraction=0.0,
            min_value=0,
            max_value=87400,
        ),
    }
    return TableStats(
        database_name="financial",
        table_name="trans",
        table_rows=1056320,
        avg_row_length=64,
        data_length_mb=89.1,
        primary_key_columns=["trans_id"],
        columns=cols,
        canonical_columns={k.lower(): k for k in cols},
    )


def test_validate_select_only_rejects_mutations() -> None:
    validate_select_only("SELECT * FROM lapTimes WHERE driverId = 1")
    with pytest.raises(ValueError):
        validate_select_only("DROP TABLE lapTimes")
    with pytest.raises(ValueError):
        validate_select_only("DELETE FROM trans WHERE account_id = 5")


def test_parse_equality_range_and_order_by_on_schema_1() -> None:
    sql = (
        "SELECT raceId, driverId, lap, milliseconds "
        "FROM lapTimes "
        "WHERE driverId = 44 AND milliseconds <= 90000 "
        "ORDER BY milliseconds ASC LIMIT 20"
    )
    parsed = parse_query(sql)
    assert parsed.primary_table == "lapTimes"
    assert parsed.where_columns == ["driverId", "milliseconds"]
    assert parsed.order_by_columns == ["milliseconds"]
    assert parsed.has_order_by is True
    assert parsed.has_limit is True
    assert parsed.limit_value == 20
    assert parsed.predicate_types["driverId"] == "equality"
    assert parsed.predicate_types["milliseconds"] == "range"


def test_parse_in_between_like_prefix_on_schema_2() -> None:
    sql = (
        "SELECT trans_id, account_id, amount "
        "FROM trans "
        "WHERE account_id IN (10, 20, 30) "
        "AND date BETWEEN '1996-01-01' AND '1996-12-31' "
        "AND type LIKE 'pri%'"
    )
    parsed = parse_query(sql)
    assert parsed.primary_table == "trans"
    assert parsed.predicate_types["account_id"] == "in"
    assert parsed.predicate_types["date"] == "range"
    assert parsed.predicate_types["type"] == "like_prefix"


def test_parse_join_with_aliases() -> None:
    sql = (
        "SELECT t.trans_id, t.amount, a.district_id "
        "FROM trans t "
        "JOIN account a ON t.account_id = a.account_id "
        "WHERE t.amount > 10000 AND a.district_id = 1"
    )
    parsed = parse_query(sql, target_table="trans")
    assert parsed.primary_table == "trans"
    assert parsed.has_join is True
    assert "account_id" in parsed.join_columns
    assert "amount" in parsed.where_columns
    assert "district_id" not in parsed.where_columns


def test_template_normalization_masks_literals() -> None:
    q1 = "SELECT * FROM trans WHERE account_id = 105 AND type = 'vydej'"
    q2 = "SELECT * FROM trans WHERE account_id = 9981 AND type = 'prijem'"
    assert extract_query_template(q1) == extract_query_template(q2)


def test_candidate_generation_on_two_schemas() -> None:
    s1 = _build_f1_laptimes_schema()
    s2 = _build_financial_trans_schema()

    q1 = "SELECT * FROM lapTimes WHERE driverId = 44 AND position <= 3 ORDER BY milliseconds ASC LIMIT 10"
    p1 = parse_query(q1, target_table=s1.table_name, schema_columns=s1.canonical_columns)
    cands1 = generate_candidates(p1, excluded_columns=s1.excluded_columns(), single_pk_column=s1.single_pk_column())
    col_tuples1 = [c.columns for c in cands1]
    assert ("driverId",) in col_tuples1
    assert ("driverId", "position") in col_tuples1
    assert ("driverId", "milliseconds") in col_tuples1

    q2 = "SELECT * FROM trans WHERE trans_id = 100 AND account_id = 55 AND date >= '1997-01-01'"
    p2 = parse_query(q2, target_table=s2.table_name, schema_columns=s2.canonical_columns)
    cands2 = generate_candidates(p2, excluded_columns=s2.excluded_columns(), single_pk_column=s2.single_pk_column())
    col_tuples2 = [c.columns for c in cands2]
    # Single-column PK trans_id is skipped as a standalone 1-col candidate
    assert ("trans_id",) not in col_tuples2
    assert ("account_id",) in col_tuples2
    assert ("account_id", "date") in col_tuples2


def test_feature_extractor_schema_agnostic_on_two_schemas() -> None:
    s1 = _build_f1_laptimes_schema()
    s2 = _build_financial_trans_schema()
    cache = DummyStatsCache({"lapTimes": s1, "trans": s2})

    q1 = "SELECT * FROM lapTimes WHERE driverId = 44 AND position = 1"
    p1 = parse_query(q1, target_table=s1.table_name, schema_columns=s1.canonical_columns)
    c1 = generate_candidates(p1)[0]
    f1 = extract_features(p1, c1, cache)

    q2 = "SELECT * FROM trans WHERE account_id = 900 AND type = 'vydej'"
    p2 = parse_query(q2, target_table=s2.table_name, schema_columns=s2.canonical_columns)
    c2 = generate_candidates(p2)[0]
    f2 = extract_features(p2, c2, cache)

    assert set(f1.keys()) == set(FEATURE_NAMES)
    assert set(f2.keys()) == set(FEATURE_NAMES)
    assert all(isinstance(v, float) for v in f1.values())
    assert all(isinstance(v, float) for v in f2.values())
    assert f1["table_rows"] == 419328.0
    assert f2["table_rows"] == 1056320.0
