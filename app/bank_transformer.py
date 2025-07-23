"""
Bank Statement Transformer

Converts bank statements from various UK banks (Lloyds, NatWest, RBS) 
into standardized format for reconciliation processing.

Usage:
    from app.bank_transformer import transform_bank_statement, BankType
    
    # Transform with explicit bank type
    result = transform_bank_statement(df, BankType.LLOYDS)
    
    # Or use convenience functions
    result = transform_lloyds_statement(df)
    result = transform_natwest_statement(df)
"""

import pandas as pd
import logging
from typing import List, Dict
from .bank_templates import BankType, get_bank_template, BANK_TEMPLATES

# Configure logging
logger = logging.getLogger(__name__)

def transform_bank_statement(df: pd.DataFrame, bank_type: BankType) -> pd.DataFrame:
    """
    Transform bank statement to standardized format.
    
    Args:
        df: Raw bank statement DataFrame (read with header=None)
        bank_type: Bank type (BankType.LLOYDS, BankType.NATWEST, etc.)
        
    Returns:
        DataFrame with columns: ['Date', 'Description', 'Amount']
        
    Raises:
        ValueError: If bank_type is not supported
        Exception: If transformation fails
    """
    logger.info(f"🏦 Transforming {bank_type.value} statement: {len(df)} rows")
    
    if df.empty:
        logger.warning("Empty DataFrame provided")
        return pd.DataFrame(columns=['Date', 'Description', 'Amount'])
    
    # Get bank template
    template = get_bank_template(bank_type)
    logger.info(f"📋 Using template: {template.name}")
    
    try:
        # Step 1: Find header row
        header_row_idx = _find_header_row(df, template)
        if header_row_idx is None:
            logger.error(f"❌ Could not find header row for {template.name}")
            return pd.DataFrame(columns=['Date', 'Description', 'Amount'])
        
        # Step 2: Extract headers
        headers = _extract_headers(df, header_row_idx)
        logger.info(f"📋 Headers: {headers}")
        
        # Step 3: Find transaction rows
        transaction_indices = _find_transaction_rows(df, template, header_row_idx)
        logger.info(f"💰 Found {len(transaction_indices)} transactions")
        
        if not transaction_indices:
            logger.warning("No transaction rows found")
            return pd.DataFrame(columns=['Date', 'Description', 'Amount'])
        
        # Step 4: Transform transactions
        result = _transform_transactions(df, transaction_indices, headers, template)
        
        logger.info(f"✅ Successfully transformed {len(result)} transactions")
        return result
        
    except Exception as e:
        logger.error(f"❌ Transformation failed: {e}")
        raise Exception(f"Failed to transform {bank_type.value} statement: {str(e)}")

def transform_lloyds_statement(df: pd.DataFrame) -> pd.DataFrame:
    """Transform Lloyds bank statement."""
    return transform_bank_statement(df, BankType.LLOYDS)

def transform_natwest_statement(df: pd.DataFrame) -> pd.DataFrame:
    """Transform NatWest bank statement."""
    return transform_bank_statement(df, BankType.NATWEST)

def transform_rbs_statement(df: pd.DataFrame) -> pd.DataFrame:
    """Transform RBS bank statement."""
    return transform_bank_statement(df, BankType.RBS)

def get_supported_bank_types() -> List[str]:
    """Get list of supported bank types."""
    return [bank_type.value for bank_type in BankType]

def validate_transformed_data(df: pd.DataFrame) -> Dict[str, any]:
    """
    Validate transformed bank statement data.
    
    Returns:
        Dict with validation results and statistics
    """
    validation = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'stats': {
            'total_transactions': len(df),
            'date_issues': 0,
            'amount_issues': 0,
            'description_issues': 0
        }
    }
    
    if df.empty:
        validation['is_valid'] = False
        validation['errors'].append("No transactions found")
        return validation
    
    # Check required columns
    required_cols = ['Date', 'Description', 'Amount']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        validation['is_valid'] = False
        validation['errors'].append(f"Missing columns: {missing_cols}")
        return validation
    
    # Validate dates
    empty_dates = df['Date'].isna().sum() + (df['Date'] == '').sum()
    if empty_dates > 0:
        validation['stats']['date_issues'] = empty_dates
        validation['warnings'].append(f"{empty_dates} transactions with empty dates")
    
    # Validate amounts
    non_numeric_amounts = 0
    for amount in df['Amount']:
        try:
            float(amount)
        except (ValueError, TypeError):
            non_numeric_amounts += 1
    
    if non_numeric_amounts > 0:
        validation['stats']['amount_issues'] = non_numeric_amounts
        validation['warnings'].append(f"{non_numeric_amounts} transactions with invalid amounts")
    
    # Validate descriptions
    empty_descriptions = df['Description'].isna().sum() + (df['Description'] == '').sum()
    if empty_descriptions > 0:
        validation['stats']['description_issues'] = empty_descriptions
        validation['warnings'].append(f"{empty_descriptions} transactions with empty descriptions")
    
    return validation

# Internal helper functions
def _find_header_row(df: pd.DataFrame, template) -> int:
    """Find the header row using template rules."""
    for idx, row in df.iterrows():
        row_values = [str(val).lower().strip() if pd.notna(val) else "" for val in row]
        
        # Count matches with template header keywords
        matches = sum(1 for keyword in template.header_keywords 
                     if any(keyword in val for val in row_values))
        
        # Need at least 2 matches for header detection
        if matches >= 2:
            logger.debug(f"Header found at row {idx} with {matches} matches")
            return idx
    
    return None

def _extract_headers(df: pd.DataFrame, header_row_idx: int) -> List[str]:
    """Extract and clean header row."""
    headers = df.iloc[header_row_idx].fillna("").astype(str).str.strip().str.lower().tolist()
    return [h for h in headers if h]  # Remove empty headers

def _find_transaction_rows(df: pd.DataFrame, template, header_row_idx: int) -> List[int]:
    """Find rows containing transaction data using smart date column detection."""
    transaction_indices = []
    
    # Extract headers and find the date column
    headers = _extract_headers(df, header_row_idx)
    
    # Use template's smart date column detection
    date_col_idx = template.find_date_column_index(headers)
    
    # Validate that this column actually contains dates
    if not template.validate_date_column(df, date_col_idx, header_row_idx):
        logger.warning(f"Date column {date_col_idx} validation failed, trying alternatives...")
        
        # Try other columns if validation failed
        for alt_col in range(min(3, df.shape[1])):  # Check first 3 columns
            if alt_col != date_col_idx and template.validate_date_column(df, alt_col, header_row_idx):
                date_col_idx = alt_col
                logger.info(f"Using alternative date column: {date_col_idx}")
                break
    
    logger.info(f"Using date column index: {date_col_idx} (header: '{headers[date_col_idx] if date_col_idx < len(headers) else 'N/A'}')")
    
    for idx in range(header_row_idx + 1, len(df)):
        row = df.iloc[idx]
        
        # Skip completely empty rows
        if row.isna().all():
            continue
        
        # Get the value from the identified date column
        if date_col_idx < len(row):
            date_cell = row.iloc[date_col_idx]
        else:
            # Fallback to first column if date column index is invalid
            date_cell = row.iloc[0]
            
        # Skip rows where date cell is empty
        if pd.isna(date_cell) or str(date_cell).strip() == "":
            logger.debug(f"Skipping row {idx}: empty date cell")
            continue
        
        # Skip summary rows (check before date validation for efficiency)
        row_text = " ".join([str(val) for val in row if pd.notna(val)]).lower()
        if any(keyword in row_text for keyword in template.skip_keywords):
            logger.debug(f"Skipping row {idx}: summary row - {row_text[:50]}...")
            continue
        
        # Check if the date cell contains a valid date pattern
        date_cell_str = str(date_cell).strip()
        if template.matches_date_pattern(date_cell_str):
            transaction_indices.append(idx)
            logger.debug(f"Found transaction row {idx}: date='{date_cell_str}' in column {date_col_idx}")
        else:
            logger.debug(f"Skipping row {idx}: invalid date pattern '{date_cell_str}'")
    
    return transaction_indices

def _transform_transactions(df: pd.DataFrame, transaction_indices: List[int], 
                          headers: List[str], template) -> pd.DataFrame:
    """Transform transaction rows to standard format."""
    from .utils import parse_amount, clean_description
    
    # Map column positions
    column_map = template.map_columns(headers)
    
    transformed_data = []
    
    for row_idx in transaction_indices:
        row = df.iloc[row_idx]
        
        try:
            # Extract date
            date = str(row.iloc[column_map.get('date', 0)]).strip()
            
            # Extract description
            description = template.extract_description(row, column_map, headers)
            description = clean_description(description)
            
            # Extract amount  
            amount = template.extract_amount(row, column_map, headers)
            
            
            transformed_data.append({
                'Date': date,
                'Description': description,
                'Amount': amount
            })
            
        except Exception as e:
            logger.warning(f"Error processing row {row_idx}: {e}")
            continue
    
    return pd.DataFrame(transformed_data)

# For testing and development
if __name__ == "__main__":
    # Simple test
    print("🧪 Testing Bank Transformer...")
    
    # Test data
    sample_data = pd.DataFrame([
        ['Client ID:', 'XXXXXXXX', None, None, None],
        ['Bank Name:', 'Lloyds', None, None, None],
        ['Posting Date', 'Type', 'Details', 'Debits', 'Credits'],
        ['11-Apr-2025', 'CHAPS Payment', 'F/FLOW 1/ACROMION', None, '900.00'],
        ['12-Apr-2025', 'Direct Debit', 'UTILITIES', '150.00', None]
    ])
    
    result = transform_lloyds_statement(sample_data)
    print(f"✅ Transformed {len(result)} transactions")
    print(result)
    
    # Validate result
    validation = validate_transformed_data(result)
    print(f"Validation: {'✅ Valid' if validation['is_valid'] else '❌ Invalid'}")
    if validation['warnings']:
        print(f"Warnings: {validation['warnings']}")