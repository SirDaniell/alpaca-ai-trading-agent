"""
SQL Validator for AI Tool Calls
Ensures only safe SELECT queries are executed
"""

import sqlparse
from sqlparse.sql import IdentifierList, Identifier, Token, TokenList
from sqlparse.tokens import Keyword, DML, Whitespace
from typing import List, Dict, Set, Tuple


class SQLValidation:
    """Result of SQL validation"""
    def __init__(self, valid: bool, error: str = None):
        self.valid = valid
        self.error = error


# Allowed tables and columns for AI queries
ALLOWED_TABLES = {
    "assets": ["asset_id", "symbol", "name", "asset_class", "sector", "active"],
    "transactions": ["transaction_id", "user_id", "symbol", "quantity", "price", "timestamp", "type", "status"],
    "portfolios": ["portfolio_id", "user_id", "name", "total_value", "created_at"],
    "market_data_ohlcv": ["symbol", "time", "open", "high", "low", "close", "volume", "timeframe"]
}


def validate_sql(query: str, user_id: str = None) -> SQLValidation:
    """
    Validate SQL query for security
    
    Args:
        query: SQL query string
        user_id: Optional user ID for additional validation
        
    Returns:
        SQLValidation object with valid flag and error message
    """
    
    # 1. Parse SQL
    try:
        parsed = sqlparse.parse(query)
    except Exception as e:
        return SQLValidation(False, f"SQL parsing error: {str(e)}")
    
    if not parsed:
        return SQLValidation(False, "Invalid SQL syntax")
    
    stmt = parsed[0]
    
    # 2. Check for dangerous keywords
    dangerous_keywords = [
        'DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE', 
        'TRUNCATE', 'EXEC', 'EXECUTE', 'GRANT', 'REVOKE'
    ]
    
    tokens = [str(t).upper().strip() for t in stmt.flatten() if t.ttype in (Keyword, DML)]
    
    for dangerous in dangerous_keywords:
        if dangerous in tokens:
            return SQLValidation(False, f"Forbidden keyword: {dangerous}")
    
    # 3. Ensure it's a SELECT statement
    first_keyword = None
    for token in stmt.tokens:
        if token.ttype is DML:
            first_keyword = str(token).upper().strip()
            break
    
    if first_keyword != 'SELECT':
        return SQLValidation(False, "Only SELECT queries are allowed")
    
    # 4. Extract and validate tables
    tables = extract_tables(stmt)
    for table in tables:
        if table not in ALLOWED_TABLES:
            return SQLValidation(False, f"Table '{table}' is not allowed")
    
    # 5. Extract and validate columns
    columns_by_table = extract_columns(stmt, tables)
    for table, cols in columns_by_table.items():
        allowed_cols = ALLOWED_TABLES.get(table, [])
        for col in cols:
            if col != '*' and col not in allowed_cols:
                return SQLValidation(False, f"Column '{col}' not allowed in table '{table}'")
    
    # 6. Check for LIMIT clause
    has_limit = 'LIMIT' in [str(t).upper() for t in stmt.flatten() if t.ttype is Keyword]
    if not has_limit:
        # Will be added automatically
        pass
    
    return SQLValidation(True)


def extract_tables(stmt: TokenList) -> Set[str]:
    """Extract table names from SQL statement"""
    tables = set()
    
    from_seen = False
    for token in stmt.tokens:
        if from_seen:
            if isinstance(token, Identifier):
                tables.add(str(token.get_real_name()).lower())
            elif isinstance(token, IdentifierList):
                for identifier in token.get_identifiers():
                    tables.add(str(identifier.get_real_name()).lower())
            from_seen = False
        
        if token.ttype is Keyword and str(token).upper() == 'FROM':
            from_seen = True
    
    return tables


def extract_columns(stmt: TokenList, tables: Set[str]) -> Dict[str, List[str]]:
    """Extract column names per table (simplified)"""
    # For simplicity, we'll assume all columns belong to all tables
    # A more sophisticated implementation would parse JOIN conditions
    columns = {}
    
    for table in tables:
        columns[table] = []
    
    # Extract SELECT columns
    select_seen = False
    for token in stmt.tokens:
        if token.ttype is DML and str(token).upper() == 'SELECT':
            select_seen = True
            continue
        
        if select_seen and token.ttype is not Whitespace:
            if isinstance(token, IdentifierList):
                for identifier in token.get_identifiers():
                    col_name = str(identifier.get_real_name()).lower()
                    for table in tables:
                        if col_name not in columns[table]:
                            columns[table].append(col_name)
            elif isinstance(token, Identifier):
                col_name = str(token.get_real_name()).lower()
                for table in tables:
                    if col_name not in columns[table]:
                        columns[table].append(col_name)
            break
    
    return columns


def sanitize_query(query: str, max_limit: int = 100) -> str:
    """
    Sanitize SQL query by adding LIMIT if missing
    
    Args:
        query: SQL query string
        max_limit: Maximum number of rows to return
        
    Returns:
        Sanitized query string
    """
    query = query.strip()
    
    # Add LIMIT if not present
    if 'LIMIT' not in query.upper():
        query += f" LIMIT {max_limit}"
    
    return query
