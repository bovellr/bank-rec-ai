"""
Shared utility functions for bank reconciliation system.

Common functions used across multiple modules for data processing,
validation, and formatting.
"""

import re
import pandas as pd
from typing import Union, Optional
import logging

logger = logging.getLogger(__name__)

def parse_amount(amount: Union[str, float, int]) -> float:
    """
    Parse amount string or number into float.
    
    Handles currency symbols, commas, and various formats.
    
    Args:
        amount: Amount to parse (string, float, or int)
        
    Returns:
        Parsed amount as float
        
    Examples:
        parse_amount("£1,234.56") -> 1234.56
        parse_amount("-150.00") -> -150.0
        parse_amount("") -> 0.0
    """
    if pd.isna(amount) or amount == "":
        return 0.0
    
    # If already a number, return as float
    if isinstance(amount, (int, float)):
        return float(amount)
    
    # Convert to string and clean
    amount_str = str(amount).strip()
    if not amount_str:
        return 0.0
    
    # Remove currency symbols and formatting
    cleaned = re.sub(r'[£$€,\s]', '', amount_str)
    
    # Handle parentheses as negative (accounting format)
    if cleaned.startswith('(') and cleaned.endswith(')'):
        cleaned = '-' + cleaned[1:-1]
    
    try:
        return float(cleaned)
    except ValueError:
        logger.warning(f"Could not parse amount: '{amount}'")
        return 0.0

def clean_description(description: str) -> str:
    """
    Clean and standardize transaction description.
    
    Args:
        description: Raw description string
        
    Returns:
        Cleaned description string
    """
    if pd.isna(description) or not description:
        return "Transaction"
    
    desc = str(description).strip()
    
    # Remove extra whitespace and normalize
    desc = re.sub(r'\s+', ' ', desc)
    
    # Remove common noise patterns
    noise_patterns = [
        r'^-\s*',           # Leading dash
        r'\s*-$',           # Trailing dash  
        r'^null$',          # Null values
        r'^none$',          # None values
        r'^\s*$'            # Empty/whitespace only
    ]
    
    for pattern in noise_patterns:
        desc = re.sub(pattern, '', desc, flags=re.IGNORECASE)
    
    # Capitalize first letter of each word
    desc = desc.title()
    
    return desc if desc else "Transaction"

def format_currency(amount: float, currency: str = "£") -> str:
    """
    Format amount as currency string.
    
    Args:
        amount: Amount to format
        currency: Currency symbol (default: £)
        
    Returns:
        Formatted currency string
        
    Examples:
        format_currency(1234.56) -> "£1,234.56"
        format_currency(-150.0) -> "-£150.00"
    """
    if amount < 0:
        return f"-{currency}{abs(amount):,.2f}"
    else:
        return f"{currency}{amount:,.2f}"

def format_currency(amount: float, currency: str = "£") -> str:
    """
    Format amount as currency string.
    
    Args:
        amount: Amount to format
        currency: Currency symbol (default: £)
        
    Returns:
        Formatted currency string
        
    Examples:
        format_currency(1234.56) -> "£1,234.56"
        format_currency(-150.0) -> "-£150.00"
    """
    if amount < 0:
        return f"-{currency}{abs(amount):,.2f}"
    else:
        return f"{currency}{amount:,.2f}"

def validate_date_string(date_str: str) -> bool:
    """
    Check if string looks like a valid date.
    
    Args:
        date_str: Date string to validate
        
    Returns:
        True if looks like a date, False otherwise
    """
    if not date_str or len(date_str) < 6:
        return False
    
    # Common date patterns
    date_patterns = [
        r'\d{1,2}[-/]\w{3}[-/]\d{4}',      # 11-Apr-2025
        r'\d{1,2}[-/]\d{1,2}[-/]\d{4}',    # 11/04/2025
        r'\d{4}[-/]\d{1,2}[-/]\d{1,2}',    # 2025-04-11
        r'\d{1,2}\s+\w{3}\s+\d{4}',        # 11 Apr 2025
    ]
    
    return any(re.match(pattern, date_str.strip()) for pattern in date_patterns)

def standardize_date_format(date_str: str) -> str:
    """
    Convert various date formats to ISO format (YYYY-MM-DD).
    
    Args:
        date_str: Date string in various formats
        
    Returns:
        Standardized date string or original if can't parse
    """
    if not date_str or not validate_date_string(date_str):
        return date_str
    
    # Try parsing with pandas
    try:
        parsed_date = pd.to_datetime(date_str, dayfirst=True)
        return parsed_date.strftime('%Y-%m-%d')
    except:
        logger.debug(f"Could not standardize date: '{date_str}'")
        return date_str

def detect_delimiter(text: str) -> str:
    """
    Detect delimiter used in CSV-like text.
    
    Args:
        text: Sample text to analyze
        
    Returns:
        Most likely delimiter (comma, semicolon, or tab)
    """
    delimiters = [',', ';', '\t', '|']
    delimiter_counts = {}
    
    for delimiter in delimiters:
        count = text.count(delimiter)
        if count > 0:
            delimiter_counts[delimiter] = count
    
    if delimiter_counts:
        return max(delimiter_counts, key=delimiter_counts.get)
    
    return ','  # Default to comma

def safe_float_convert(value) -> Optional[float]:
    """
    Safely convert value to float, returning None if impossible.
    
    Args:
        value: Value to convert
        
    Returns:
        Float value or None if conversion fails
    """
    if pd.isna(value):
        return None
    
    try:
        return float(value)
    except (ValueError, TypeError):
        return None

def truncate_string(text: str, max_length: int = 100) -> str:
    """
    Truncate string to maximum length with ellipsis.
    
    Args:
        text: String to truncate
        max_length: Maximum allowed length
        
    Returns:
        Truncated string with "..." if needed
    """
    if not text or len(text) <= max_length:
        return text
    
    return text[:max_length-3] + "..."

def remove_empty_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove completely empty rows from DataFrame.
    
    Args:
        df: DataFrame to clean
        
    Returns:
        DataFrame with empty rows removed
    """
    return df.dropna(how='all').reset_index(drop=True)

def identify_numeric_columns(df: pd.DataFrame) -> list:
    """
    Identify columns that appear to contain numeric data.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        List of column names that appear numeric
    """
    numeric_columns = []
    
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            numeric_columns.append(col)
            continue
        
        # Check if string column contains mostly numbers
        non_null_values = df[col].dropna().astype(str)
        if len(non_null_values) == 0:
            continue
            
        numeric_count = 0
        for value in non_null_values:
            cleaned_value = re.sub(r'[£$€,\s()-]', '', str(value))
            try:
                float(cleaned_value)
                numeric_count += 1
            except ValueError:
                pass
        
        # If >70% of values are numeric, consider it a numeric column
        if numeric_count / len(non_null_values) > 0.7:
            numeric_columns.append(col)
    
    return numeric_columns

def create_summary_stats(df: pd.DataFrame) -> dict:
    """
    Create summary statistics for transformed bank statement.
    
    Args:
        df: Transformed bank statement DataFrame
        
    Returns:
        Dictionary with summary statistics
    """
    if df.empty:
        return {
            'total_transactions': 0,
            'date_range': 'No data',
            'total_amount': 0.0,
            'credit_amount': 0.0,
            'debit_amount': 0.0,
            'credit_count': 0,
            'debit_count': 0
        }
    
    # Basic stats
    stats = {
        'total_transactions': len(df),
        'date_range': 'Unknown',
        'total_amount': 0.0,
        'credit_amount': 0.0,
        'debit_amount': 0.0,
        'credit_count': 0,
        'debit_count': 0
    }
    
    # Date range
    if 'Date' in df.columns:
        dates = df['Date'].dropna()
        if not dates.empty:
            try:
                date_series = pd.to_datetime(dates, errors='coerce').dropna()
                if not date_series.empty:
                    min_date = date_series.min().strftime('%Y-%m-%d')
                    max_date = date_series.max().strftime('%Y-%m-%d')
                    stats['date_range'] = f"{min_date} to {max_date}"
            except:
                pass
    
    # Amount analysis
    if 'Amount' in df.columns:
        amounts = pd.to_numeric(df['Amount'], errors='coerce').fillna(0)
        
        stats['total_amount'] = amounts.sum()
        
        # Credits (positive amounts)
        credits = amounts[amounts > 0]
        stats['credit_amount'] = credits.sum()
        stats['credit_count'] = len(credits)
        
        # Debits (negative amounts) 
        debits = amounts[amounts < 0]
        stats['debit_amount'] = abs(debits.sum())
        stats['debit_count'] = len(debits)
    
    return stats

def log_processing_stats(operation: str, input_count: int, output_count: int, 
                        duration_seconds: float = None):
    """
    Log processing statistics in a consistent format.
    
    Args:
        operation: Description of operation performed
        input_count: Number of input records
        output_count: Number of output records
        duration_seconds: Processing time in seconds (optional)
    """
    duration_str = f" in {duration_seconds:.2f}s" if duration_seconds else ""
    success_rate = (output_count / input_count * 100) if input_count > 0 else 0
    
    logger.info(
        f"📊 {operation}: {input_count} → {output_count} "
        f"({success_rate:.1f}% success){duration_str}"
    )

def validate_required_columns(df: pd.DataFrame, required_columns: list) -> dict:
    """
    Validate that DataFrame contains required columns.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        
    Returns:
        Validation result dictionary
    """
    validation = {
        'is_valid': True,
        'missing_columns': [],
        'extra_columns': [],
        'message': ''
    }
    
    df_columns = set(df.columns.tolist())
    required_set = set(required_columns)
    
    # Check for missing columns
    missing = required_set - df_columns
    if missing:
        validation['is_valid'] = False
        validation['missing_columns'] = list(missing)
    
    # Note extra columns (not an error, just info)
    extra = df_columns - required_set
    if extra:
        validation['extra_columns'] = list(extra)
    
    # Create message
    if validation['is_valid']:
        validation['message'] = "✅ All required columns present"
    else:
        validation['message'] = f"❌ Missing columns: {validation['missing_columns']}"
    
    return validation

# Test functions for development
def test_amount_parsing():
    """Test amount parsing with various formats."""
    test_amounts = [
        "£1,234.56",     # Currency with comma
        "-150.00",       # Negative
        "(75.50)",       # Accounting negative
        "1234",          # Simple integer
        "",              # Empty string
        "£0.00",         # Zero
        "invalid",       # Invalid format
        123.45           # Already a number
    ]
    
    print("🧪 Testing amount parsing:")
    print("=" * 30)
    
    for test_amount in test_amounts:
        result = parse_amount(test_amount)
        print(f"'{test_amount}' → {result}")

def test_description_cleaning():
    """Test description cleaning functionality."""
    test_descriptions = [
        "  CHAPS Payment  ",           # Extra whitespace
        "- Direct Debit -",            # Dashes
        "null",                        # Null value
        "UTILITIES     PAYMENT",       # Multiple spaces
        "",                            # Empty
        "transfer to savings account"  # Lowercase
    ]
    
    print("\n🧪 Testing description cleaning:")
    print("=" * 35)
    
    for desc in test_descriptions:
        result = clean_description(desc)
        print(f"'{desc}' → '{result}'")

if __name__ == "__main__":
    # Run tests
    test_amount_parsing()
    test_description_cleaning()
    
    # Test date validation
    print("\n🧪 Testing date validation:")
    print("=" * 30)
    
    test_dates = [
        "11-Apr-2025",
        "11 Apr 2025", 
        "2025-04-11",
        "invalid_date",
        "11/04/2025"
    ]
    
    for date in test_dates:
        is_valid = validate_date_string(date)
        standardized = standardize_date_format(date)
        print(f"'{date}' → Valid: {is_valid}, Standardized: '{standardized}'")