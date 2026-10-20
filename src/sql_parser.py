import re
import sqlparse

def parse_sql_query(sql_string):
    if not sql_string or not sql_string.strip():
        return {"table_name": "unknown", "candidate_column": "unknown", "has_join": 0, "is_range_scan": 0}

    sql_upper = sql_string.upper()
    
    has_join = 1 if "JOIN" in sql_upper else 0
    
    range_operators = [">", "<", ">=", "<=", "BETWEEN"]
    is_range_scan = 1 if any(op in sql_upper for op in range_operators) else 0

    table_match = re.search(r"(?:FROM|JOIN)\s+([a-zA-Z0-9_]+)", sql_upper)
    table_name = table_match.group(1).lower() if table_match else "unknown"

    col_match = re.search(r"(?:WHERE|AND|OR|ON)\s+(?:[a-zA-Z0-9_]+\.)?([a-zA-Z0-9_]+)\s*(?:=|>|<|>=|<=|!=|LIKE|IN|BETWEEN)", sql_upper)
    candidate_column = col_match.group(1).lower() if col_match else "unknown"

    return {
        "table_name": table_name,
        "candidate_column": candidate_column,
        "has_join": has_join,
        "is_range_scan": is_range_scan
    }