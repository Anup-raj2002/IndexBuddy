"""
parsing/query_parser.py
───────────────────────
SQL AST parser built on sqlglot (dialect="mysql").

Extracts:
  - Target table(s) and alias mapping
  - Columns referenced in WHERE, JOIN ON, GROUP BY, and ORDER BY clauses
  - Predicate types per column: equality, range, in, like_prefix, like_wildcard, other
  - Literal values associated with predicates (used for selectivity estimation)
  - Canonical SQL template (literals replaced with '?') for template-level holdout splitting
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import sqlglot
from sqlglot import exp


# Priority order when a column appears in multiple predicates in the same query
PREDICATE_PRIORITY: Dict[str, int] = {
    "equality": 5,
    "in": 4,
    "like_prefix": 3,
    "range": 2,
    "other": 1,
    "like_wildcard": 0,
}


@dataclass
class PredicateInfo:
    """Represents a single filter predicate extracted from WHERE or JOIN."""

    column: str
    predicate_type: str  # 'equality' | 'range' | 'in' | 'like_prefix' | 'like_wildcard' | 'other'
    operator: str
    literals: List[Any] = field(default_factory=list)
    table: Optional[str] = None


@dataclass
class ParsedQuery:
    """Structured representation of a parsed MySQL SELECT query."""

    raw_sql: str
    primary_table: str
    tables: List[str] = field(default_factory=list)
    where_columns: List[str] = field(default_factory=list)
    join_columns: List[str] = field(default_factory=list)
    group_by_columns: List[str] = field(default_factory=list)
    order_by_columns: List[str] = field(default_factory=list)
    predicates: List[PredicateInfo] = field(default_factory=list)
    predicate_types: Dict[str, str] = field(default_factory=dict)
    has_join: bool = False
    has_group_by: bool = False
    has_order_by: bool = False
    has_limit: bool = False
    limit_value: Optional[int] = None
    template_sql: str = ""

    def all_referenced_columns(self) -> List[str]:
        """Return deduplicated list of columns across WHERE, JOIN, GROUP BY, ORDER BY."""
        seen: Set[str] = set()
        ordered: List[str] = []
        for col in (
            self.where_columns
            + self.join_columns
            + self.group_by_columns
            + self.order_by_columns
        ):
            if col not in seen:
                seen.add(col)
                ordered.append(col)
        return ordered


def _extract_literal_value(node: exp.Expression) -> Any:
    """Extract Python primitive from a sqlglot literal/negation expression."""
    if isinstance(node, exp.Literal):
        text = node.this
        if node.is_number:
            try:
                return int(text) if "." not in str(text) else float(text)
            except ValueError:
                return text
        return str(text)
    if isinstance(node, exp.Neg) and isinstance(node.this, exp.Literal):
        val = _extract_literal_value(node.this)
        if isinstance(val, (int, float)):
            return -val
    return None


def _resolve_column_table(
    col_expr: exp.Column,
    alias_map: Dict[str, str],
    default_table: str,
) -> Tuple[str, str]:
    """Return (resolved_table_name, column_name) in lowercase."""
    col_name = col_expr.name.lower()
    tbl_qualifier = (col_expr.table or "").lower()
    if tbl_qualifier:
        resolved_tbl = alias_map.get(tbl_qualifier, tbl_qualifier)
    else:
        resolved_tbl = default_table
    return resolved_tbl, col_name


def _append_unique(target: List[str], item: str) -> None:
    if item and item not in target:
        target.append(item)


def extract_query_template(sql: str) -> str:
    """Normalize a SQL query by replacing all literals and IN-lists with '?'."""
    try:
        ast = sqlglot.parse_one(sql, read="mysql")
    except Exception:
        return sql.strip()

    def _mask_node(node: exp.Expression) -> exp.Expression:
        if isinstance(node, exp.In):
            # Collapse IN (a, b, c) into IN (?)
            clone = node.copy()
            clone.set("expressions", [exp.Placeholder()])
            return clone
        if isinstance(node, exp.Literal):
            return exp.Placeholder()
        return node

    masked = ast.transform(_mask_node)
    return masked.sql(dialect="mysql")


def parse_query(
    sql: str,
    target_table: Optional[str] = "orders_benchmark",
) -> ParsedQuery:
    """Parse a MySQL query using sqlglot and extract tables, columns, and predicates.

    Parameters
    ----------
    sql : str
        Raw SQL SELECT statement.
    target_table : Optional[str]
        If the query joins multiple tables and *target_table* is among them,
        filter extracted candidate columns to those belonging to *target_table*
        (or unqualified columns). Defaults to ``"orders_benchmark"``.
    """
    if not sql or not sql.strip():
        raise ValueError("Empty SQL query passed to parse_query().")

    ast = sqlglot.parse_one(sql, read="mysql")

    # 1. Extract tables & aliases
    alias_map: Dict[str, str] = {}
    tables: List[str] = []
    for tbl_node in ast.find_all(exp.Table):
        t_name = tbl_node.name.lower()
        if not t_name:
            continue
        _append_unique(tables, t_name)
        alias_map[t_name] = t_name
        alias = tbl_node.alias.lower() if tbl_node.alias else ""
        if alias:
            alias_map[alias] = t_name

    if not tables:
        primary_table = (target_table or "unknown").lower()
    elif target_table and target_table.lower() in tables:
        primary_table = target_table.lower()
    else:
        primary_table = tables[0]

    def _belongs_to_primary(col_node: exp.Column) -> Tuple[bool, str, str]:
        resolved_tbl, col_name = _resolve_column_table(col_node, alias_map, primary_table)
        # If query has multiple tables and column is unqualified, accept it;
        # if qualified, require it to match primary_table.
        return (resolved_tbl == primary_table), resolved_tbl, col_name

    where_columns: List[str] = []
    join_columns: List[str] = []
    group_by_columns: List[str] = []
    order_by_columns: List[str] = []
    predicates: List[PredicateInfo] = []
    predicate_types: Dict[str, str] = {}

    def _record_predicate(
        col_node: exp.Column,
        pred_type: str,
        operator: str,
        literals: List[Any],
    ) -> None:
        is_primary, resolved_tbl, col_name = _belongs_to_primary(col_node)
        if not is_primary:
            return
        _append_unique(where_columns, col_name)
        predicates.append(
            PredicateInfo(
                column=col_name,
                predicate_type=pred_type,
                operator=operator,
                literals=[lit for lit in literals if lit is not None],
                table=resolved_tbl,
            )
        )
        prev = predicate_types.get(col_name)
        if prev is None or PREDICATE_PRIORITY.get(pred_type, 0) > PREDICATE_PRIORITY.get(prev, 0):
            predicate_types[col_name] = pred_type

    # 2. Extract JOIN columns and ON predicates
    join_nodes = list(ast.find_all(exp.Join))
    has_join = len(join_nodes) > 0
    for j_node in join_nodes:
        on_expr = j_node.args.get("on")
        if on_expr is None:
            continue
        for eq_node in on_expr.find_all(exp.EQ):
            left, right = eq_node.left, eq_node.right
            if isinstance(left, exp.Column) and isinstance(right, exp.Column):
                for c_node in (left, right):
                    is_primary, _, col_name = _belongs_to_primary(c_node)
                    if is_primary:
                        _append_unique(join_columns, col_name)
                        if col_name not in predicate_types:
                            predicate_types[col_name] = "equality"
            elif isinstance(left, exp.Column):
                _record_predicate(left, "equality", "=", [_extract_literal_value(right)])
            elif isinstance(right, exp.Column):
                _record_predicate(right, "equality", "=", [_extract_literal_value(left)])

        # Also catch any other columns inside JOIN ON
        for col_node in on_expr.find_all(exp.Column):
            is_primary, _, col_name = _belongs_to_primary(col_node)
            if is_primary and col_name not in where_columns:
                _append_unique(join_columns, col_name)

    # 3. Extract WHERE predicates
    where_node = ast.find(exp.Where)
    if where_node is not None:
        for node in where_node.walk():
            if isinstance(node, exp.Between):
                col_expr = node.this
                if isinstance(col_expr, exp.Column):
                    low = _extract_literal_value(node.args.get("low"))
                    high = _extract_literal_value(node.args.get("high"))
                    _record_predicate(col_expr, "range", "BETWEEN", [low, high])

            elif isinstance(node, exp.In):
                col_expr = node.this
                if isinstance(col_expr, exp.Column):
                    lits = [
                        _extract_literal_value(e)
                        for e in (node.expressions or [])
                    ]
                    _record_predicate(col_expr, "in", "IN", lits)

            elif isinstance(node, (exp.Like, exp.ILike)):
                col_expr = node.this
                pattern_expr = node.expression
                if isinstance(col_expr, exp.Column):
                    pat_val = _extract_literal_value(pattern_expr)
                    pat_str = str(pat_val) if pat_val is not None else ""
                    if pat_str and not pat_str.startswith(("%", "_")):
                        p_type = "like_prefix"
                    else:
                        p_type = "like_wildcard"
                    _record_predicate(col_expr, p_type, "LIKE", [pat_val])

            elif isinstance(node, exp.EQ):
                left, right = node.left, node.right
                if isinstance(left, exp.Column) and not isinstance(right, exp.Column):
                    _record_predicate(left, "equality", "=", [_extract_literal_value(right)])
                elif isinstance(right, exp.Column) and not isinstance(left, exp.Column):
                    _record_predicate(right, "equality", "=", [_extract_literal_value(left)])
                elif isinstance(left, exp.Column) and isinstance(right, exp.Column):
                    for c_node in (left, right):
                        is_primary, _, col_name = _belongs_to_primary(c_node)
                        if is_primary:
                            _append_unique(join_columns, col_name)

            elif isinstance(node, (exp.GT, exp.GTE, exp.LT, exp.LTE)):
                op_map = {
                    exp.GT: ">",
                    exp.GTE: ">=",
                    exp.LT: "<",
                    exp.LTE: "<=",
                }
                op_str = op_map.get(type(node), "range")
                left, right = node.left, node.right
                if isinstance(left, exp.Column) and not isinstance(right, exp.Column):
                    _record_predicate(left, "range", op_str, [_extract_literal_value(right)])
                elif isinstance(right, exp.Column) and not isinstance(left, exp.Column):
                    _record_predicate(right, "range", op_str, [_extract_literal_value(left)])

            elif isinstance(node, (exp.NEQ, exp.Is)):
                col_expr = node.this
                if isinstance(col_expr, exp.Column):
                    op_str = "IS" if isinstance(node, exp.Is) else "!="
                    _record_predicate(col_expr, "other", op_str, [])

    # 4. Extract GROUP BY columns
    group_node = ast.find(exp.Group)
    has_group_by = group_node is not None
    if group_node is not None:
        for col_node in group_node.find_all(exp.Column):
            is_primary, _, col_name = _belongs_to_primary(col_node)
            if is_primary:
                _append_unique(group_by_columns, col_name)

    # 5. Extract ORDER BY columns
    order_node = ast.find(exp.Order)
    has_order_by = order_node is not None
    if order_node is not None:
        for col_node in order_node.find_all(exp.Column):
            is_primary, _, col_name = _belongs_to_primary(col_node)
            if is_primary:
                _append_unique(order_by_columns, col_name)

    # 6. Extract LIMIT
    limit_node = ast.find(exp.Limit)
    has_limit = limit_node is not None
    limit_value: Optional[int] = None
    if limit_node is not None and limit_node.expression is not None:
        val = _extract_literal_value(limit_node.expression)
        if isinstance(val, int):
            limit_value = val

    return ParsedQuery(
        raw_sql=sql.strip(),
        primary_table=primary_table,
        tables=tables,
        where_columns=where_columns,
        join_columns=join_columns,
        group_by_columns=group_by_columns,
        order_by_columns=order_by_columns,
        predicates=predicates,
        predicate_types=predicate_types,
        has_join=has_join,
        has_group_by=has_group_by,
        has_order_by=has_order_by,
        has_limit=has_limit,
        limit_value=limit_value,
        template_sql=extract_query_template(sql),
    )
