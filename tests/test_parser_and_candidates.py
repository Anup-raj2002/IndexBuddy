"""
tests/test_parser_and_candidates.py
───────────────────────────────────
Unit tests for sqlglot MySQL query parser (parsing/query_parser.py) and
candidate index generator (candidates/generator.py).
"""
from __future__ import annotations

import pytest

from candidates.generator import (
    CandidateIndex,
    build_create_index_ddl,
    generate_candidates,
)
from parsing.query_parser import extract_query_template, parse_query


def test_parse_equality_and_range_predicates() -> None:
    sql = (
        "SELECT order_id, amount FROM orders_benchmark "
        "WHERE customer_id = 42310 AND order_date >= '2023-06-01' "
        "ORDER BY amount DESC LIMIT 20"
    )
    parsed = parse_query(sql)
    assert parsed.primary_table == "orders_benchmark"
    assert parsed.where_columns == ["customer_id", "order_date"]
    assert parsed.order_by_columns == ["amount"]
    assert parsed.has_order_by is True
    assert parsed.has_limit is True
    assert parsed.limit_value == 20
    assert parsed.predicate_types["customer_id"] == "equality"
    assert parsed.predicate_types["order_date"] == "range"


def test_parse_in_between_and_like_prefix() -> None:
    sql = (
        "SELECT * FROM orders_benchmark "
        "WHERE region IN ('North', 'South') "
        "AND amount BETWEEN 100.0 AND 500.0 "
        "AND payment_method LIKE 'credit%'"
    )
    parsed = parse_query(sql)
    assert parsed.predicate_types["region"] == "in"
    assert parsed.predicate_types["amount"] == "range"
    assert parsed.predicate_types["payment_method"] == "like_prefix"


def test_parse_like_wildcard_distinction() -> None:
    sql = "SELECT * FROM orders_benchmark WHERE region LIKE '%East%'"
    parsed = parse_query(sql)
    assert parsed.predicate_types["region"] == "like_wildcard"
    # Leading wildcard LIKE alone should not produce a useless B-tree candidate
    cands = generate_candidates(parsed)
    assert cands == []


def test_parse_join_with_aliases_and_group_by() -> None:
    sql = (
        "SELECT o.region, COUNT(*) AS cnt "
        "FROM orders_benchmark o "
        "JOIN customers_benchmark c ON o.customer_id = c.customer_id "
        "WHERE c.tier = 'gold' AND o.status = 'completed' "
        "GROUP BY o.region "
        "ORDER BY cnt DESC"
    )
    parsed = parse_query(sql, target_table="orders_benchmark")
    assert parsed.primary_table == "orders_benchmark"
    assert parsed.has_join is True
    assert parsed.has_group_by is True
    assert "customer_id" in parsed.join_columns
    assert "status" in parsed.where_columns
    # c.tier belongs to customers_benchmark, not orders_benchmark
    assert "tier" not in parsed.where_columns
    assert parsed.group_by_columns == ["region"]


def test_extract_query_template_normalizes_literals() -> None:
    q1 = "SELECT * FROM orders_benchmark WHERE customer_id = 105 AND status = 'shipped'"
    q2 = "SELECT * FROM orders_benchmark WHERE customer_id = 99812 AND status = 'completed'"
    assert extract_query_template(q1) == extract_query_template(q2)


def test_generate_single_and_two_column_candidates() -> None:
    sql = (
        "SELECT order_id, customer_id, order_date, amount "
        "FROM orders_benchmark "
        "WHERE customer_id = 12345 AND order_date >= '2024-01-01' "
        "ORDER BY amount DESC"
    )
    cands = generate_candidates(sql)
    col_tuples = [c.columns for c in cands]

    # Single-column candidates
    assert ("customer_id",) in col_tuples
    assert ("order_date",) in col_tuples
    assert ("amount",) in col_tuples

    # Composite two-column candidates
    assert ("customer_id", "order_date") in col_tuples
    assert ("customer_id", "amount") in col_tuples

    # Check DDL format
    first = cands[0]
    assert isinstance(first, CandidateIndex)
    assert first.ddl == "CREATE INDEX `idx_orders_benchmark_customer_id` ON `orders_benchmark` (`customer_id`);"


def test_candidate_cap_and_exclusions() -> None:
    sql = (
        "SELECT * FROM orders_benchmark "
        "WHERE order_id = 10 AND customer_id = 50 AND product_id = 20 "
        "AND status = 'completed' AND region = 'North' AND notes IS NOT NULL "
        "GROUP BY payment_method ORDER BY order_date, amount"
    )
    cands = generate_candidates(sql, max_candidates=5)
    assert len(cands) == 5
    for c in cands:
        assert "order_id" not in c.columns
        assert "notes" not in c.columns


def test_empty_query_raises() -> None:
    with pytest.raises(ValueError):
        parse_query("   ")
