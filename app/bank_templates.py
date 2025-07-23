"""
Bank-specific templates and processing logic.

Defines how to parse different bank statement formats including
header detection, date patterns, column mapping, and data extraction rules.
"""

import re
import pandas as pd
from enum import Enum
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class BankType(Enum):
    """Supported bank statement formats."""
    LLOYDS = "lloyds"
    NATWEST = "rbs/natwest"
    HSBC = "hsbc"
    BARCLAYS = "barclays"

class BankTemplate:
    """Template defining bank-specific parsing rules."""
    
    def __init__(self, name: str, header_keywords: List[str], 
                 date_patterns: List[str], skip_keywords: List[str],
                 column_mapping: Dict[str, List[str]]):
        self.name = name
        self.header_keywords = header_keywords
        self.date_patterns = [re.compile(pattern) for pattern in date_patterns]
        self.skip_keywords = skip_keywords
        self.column_mapping = column_mapping
    
    def matches_date_pattern(self, text: str) -> bool:
        """Check if text matches any of the bank's date patterns."""
        if not text or len(text) < 6:
            return False
        
        return any(pattern.match(text.strip()) for pattern in self.date_patterns)
    
    def map_columns(self, headers: List[str]) -> Dict[str, int]:
        """Map semantic column names to actual header positions."""
        column_map = {}
        
        for semantic_name, possible_names in self.column_mapping.items():
            for i, header in enumerate(headers):
                if any(name in header.lower() for name in possible_names):
                    column_map[semantic_name] = i
                    break
        
        return column_map
    
    def find_date_column_index(self, headers: List[str]) -> int:
        """
        Find the most likely date column index with multiple fallback strategies.
        
        Returns:
            Index of the date column (0-based)
        """
        # Strategy 1: Use column mapping
        column_map = self.map_columns(headers)
        if 'date' in column_map:
            logger.debug(f"Found date column via mapping: index {column_map['date']}")
            return column_map['date']
        
        # Strategy 2: Look for exact date-related headers
        date_keywords = ['date', 'posting date', 'transaction date', 'value date']
        for keyword in date_keywords:
            for i, header in enumerate(headers):
                if keyword == header.lower().strip():
                    logger.debug(f"Found date column via exact match '{keyword}': index {i}")
                    return i
        
        # Strategy 3: Look for partial matches
        for i, header in enumerate(headers):
            if any(keyword in header.lower() for keyword in ['date', 'posting', 'transaction']):
                logger.debug(f"Found date column via partial match '{header}': index {i}")
                return i
        
        # Strategy 4: Default to first column
        logger.debug("Using first column as date column (fallback)")
        return 0
    
    def validate_date_column(self, df: pd.DataFrame, date_col_idx: int, 
                           header_row_idx: int, sample_size: int = 3) -> bool:
        """
        Validate that a column actually contains dates by sampling values.
        """
        if date_col_idx >= len(df.columns) if hasattr(df, 'columns') else date_col_idx >= df.shape[1]:
            return False
        
        valid_dates = 0
        total_checked = 0
        
        # Sample a few rows after the header
        for idx in range(header_row_idx + 1, min(header_row_idx + 1 + sample_size, len(df))):
            if idx < len(df):
                cell_value = df.iloc[idx, date_col_idx] if date_col_idx < df.shape[1] else None
                
                if pd.notna(cell_value) and str(cell_value).strip():
                    total_checked += 1
                    if self.matches_date_pattern(str(cell_value).strip()):
                        valid_dates += 1
        
        # Consider valid if >50% of checked values are valid dates
        if total_checked == 0:
            return False
        
        validity_ratio = valid_dates / total_checked
        logger.debug(f"Date column validation: {valid_dates}/{total_checked} valid ({validity_ratio:.1%})")
        
        return validity_ratio > 0.5

    def extract_description(self, row: pd.Series, column_map: Dict[str, int], headers: List[str]) -> str:
        """Extract description from transaction row with multi-column support."""
        description_parts = []
        
        # Special handling for multiple narrative columns (NatWest case)
        narrative_columns = []
        for i, header in enumerate(headers):
            if 'narrative' in header.lower() and i < len(row):
                narrative_columns.append(i)
        
        if narrative_columns:
            # Extract from ALL narrative columns
            for col_idx in narrative_columns:
                if col_idx < len(row) and pd.notna(row.iloc[col_idx]):
                    part = str(row.iloc[col_idx]).strip()
                    if part and part.lower() not in ['', 'none', 'null', 'n/a']:
                        description_parts.append(part)
        else:
            # Standard single-column extraction
            for col_type in ['type', 'description', 'details', 'reference']:
                if col_type in column_map:
                    idx = column_map[col_type]
                    if idx < len(row) and pd.notna(row.iloc[idx]):
                        part = str(row.iloc[idx]).strip()
                        if part:
                            description_parts.append(part)
        
        # If no mapped description columns, use middle columns
        if not description_parts:
            # Skip first column (date) and last 2 (amounts)
            for i in range(1, max(1, len(row) - 2)):
                if pd.notna(row.iloc[i]):
                    part = str(row.iloc[i]).strip()
                    if part and part.lower() not in ['', 'none', 'null']:
                        description_parts.append(part)
        
        # Join with separator that preserves structure
        return " | ".join(description_parts) if description_parts else "Transaction"
    
    def extract_amount(self, row: pd.Series, column_map: Dict[str, int], headers: List[str]) -> float:
        """Extract amount from transaction row."""
        # Try single amount column first (e.g., NatWest 'Value' column)
        if 'amount' in column_map:
            amount_val = row.iloc[column_map['amount']]
            if pd.notna(amount_val):
                amount_str = str(amount_val).strip()
                if amount_str:
                    return self._parse_amount_string(amount_str)
        
        # Try credit column (positive amount)
        if 'credit' in column_map:
            credit_val = row.iloc[column_map['credit']]
            if pd.notna(credit_val) and str(credit_val).strip():
                amount_str = str(credit_val).strip()
                if amount_str:
                    return self._parse_amount_string(amount_str)
        
        # Try debit column (negative amount)
        if 'debit' in column_map:
            debit_val = row.iloc[column_map['debit']]
            if pd.notna(debit_val) and str(debit_val).strip():
                amount_str = str(debit_val).strip()
                if amount_str:
                    return -self._parse_amount_string(amount_str)  # Negative for debits
        
        return 0.0
    
    def _parse_amount_string(self, amount_str: str) -> float:
        """Parse amount string, handling currency symbols and formatting."""
        if not amount_str:
            return 0.0
        
        # Remove currency symbols, commas, and whitespace
        cleaned = amount_str.replace('£', '').replace(',', '').replace(' ', '')
        
        # Handle negative signs
        is_negative = cleaned.startswith('-')
        cleaned = cleaned.lstrip('-')
        
        try:
            amount = float(cleaned)
            return -amount if is_negative else amount
        except ValueError:
            logger.warning(f"Could not parse amount: '{amount_str}'")
            return 0.0

# Define bank-specific templates
BANK_TEMPLATES = {
    BankType.LLOYDS: BankTemplate(
        name="Lloyds Bank",
        header_keywords=["posting date", "date", "type", "details", "debits", "credits"],
        date_patterns=[
            r'\d{1,2}[-/]\w{3}[-/]\d{4}',      # 11-Apr-2025
            r'\d{1,2}[-/]\d{1,2}[-/]\d{4}',    # 11/04/2025
        ],
        skip_keywords=["totals", "balance", "end of report", "transaction", "closing", "opening", "brought forward", "carried forward"],
        column_mapping={
            "date": ["posting date", "date", "transaction date"],
            "type": ["type", "transaction type"],
            "description": ["details", "description", "reference"],
            "debit": ["debit", "debits", "payment", "out"],
            "credit": ["credit", "credits", "receipt", "deposit"]
        }
    ),
    
    BankType.NATWEST: BankTemplate(
        name="NatWest Bank",
        header_keywords=["sort code", "account number", "account alias", "account short name", "currency", "account type", "bic", "bank name", "branch name", "date", "narrative", "type", "debit", "credit"],
        date_patterns=[
            r'\d{1,2}\s+\w{3}\s+\d{4}',        # 11 Apr 2025
            r'\d{1,2}/\d{1,2}/\d{4}',          # 11/04/2025
            r'\d{4}-\d{1,2}-\d{1,2}',          # 2025-04-11
        ],
        skip_keywords=["balance brought forward", "balance carried forward", "total"],
        column_mapping={
            "date": ["date", "transaction date"],
            "type": ["type", "transaction type"],
            "description": ["description", "details", "reference", "memo"],
            "amount": ["value", "amount", "transaction amount"]
        }
    ),
    
    BankType.HSBC: BankTemplate(
        name="HSBC UK",
        header_keywords=["date", "description", "amount", "balance"],
        date_patterns=[
            r'\d{1,2}/\d{1,2}/\d{4}',          # 11/04/2025
            r'\d{1,2}-\d{1,2}-\d{4}',          # 11-04-2025
        ],
        skip_keywords=["balance brought forward", "balance carried forward"],
        column_mapping={
            "date": ["date", "posting date"],
            "description": ["description", "details", "payment details"],
            "amount": ["amount", "value"],
            "debit": ["debit", "paid out"],
            "credit": ["credit", "paid in"]
        }
    ),
    
    BankType.BARCLAYS: BankTemplate(
        name="Barclays Bank",
        header_keywords=["date", "reference", "type", "amount", "balance"],
        date_patterns=[
            r'\d{1,2}/\d{1,2}/\d{2}',          # 11/04/25
            r'\d{1,2}/\d{1,2}/\d{4}',          # 11/04/2025
        ],
        skip_keywords=["balance brought forward", "total"],
        column_mapping={
            "date": ["date"],
            "description": ["reference", "description", "memo"],
            "type": ["type"],
            "amount": ["amount"],
            "debit": ["debit"],
            "credit": ["credit"]
        }
    )
}

def get_bank_template(bank_type: BankType) -> BankTemplate:
    """Get template for specified bank type."""
    if bank_type not in BANK_TEMPLATES:
        available = [bt.value for bt in BANK_TEMPLATES.keys()]
        raise ValueError(f"Unsupported bank type: {bank_type.value}. Available: {available}")
    
    return BANK_TEMPLATES[bank_type]

def get_all_bank_types() -> List[BankType]:
    """Get all supported bank types."""
    return list(BANK_TEMPLATES.keys())

def get_bank_info(bank_type: BankType) -> Dict[str, any]:
    """Get information about a specific bank template."""
    template = get_bank_template(bank_type)
    
    return {
        'name': template.name,
        'header_keywords': template.header_keywords,
        'date_patterns': [p.pattern for p in template.date_patterns],
        'skip_keywords': template.skip_keywords,
        'column_mapping': template.column_mapping
    }

def test_date_patterns():
    """Test date pattern matching for all banks."""
    test_dates = [
        "11-Apr-2025",      # Lloyds format
        "11 Apr 2025",      # NatWest format  
        "11/04/2025",       # Common format
        "2025-04-11",       # ISO format
        "11-04-2025",       # Alternative format
        "11/04/25",         # Barclays short format
        "invalid_date"      # Should not match
    ]
    
    print("🧪 Testing date patterns for all banks:")
    print("=" * 50)
    
    for bank_type in BANK_TEMPLATES:
        template = BANK_TEMPLATES[bank_type]
        print(f"\n{template.name}:")
        
        for test_date in test_dates:
            matches = template.matches_date_pattern(test_date)
            status = "✅" if matches else "❌"
            print(f"  {status} '{test_date}' -> {matches}")

if __name__ == "__main__":
    # Run tests
    test_date_patterns()
    
    # Show template info
    print(f"\n📋 Available bank templates:")
    for bank_type in get_all_bank_types():
        info = get_bank_info(bank_type)
        print(f"- {info['name']} ({bank_type.value})")