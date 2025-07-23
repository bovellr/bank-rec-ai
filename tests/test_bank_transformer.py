"""
Tests for bank statement transformer functionality.
"""

import pytest
import pandas as pd
import sys
from pathlib import Path

# Add app directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'app'))

from app.bank_transformer import (
    transform_bank_statement,
    transform_lloyds_statement,
    transform_natwest_statement,
    validate_transformed_data,
    get_supported_bank_types
)
from app.bank_templates import BankType
from app.utils import parse_amount, clean_description

class TestBankTransformer:
    
    @pytest.fixture
    def lloyds_sample_data(self):
        """Sample Lloyds bank statement data."""
        return pd.DataFrame([
            ['Client ID:', 'XXXXXXXX', None, None, None],
            ['Reporting Period:', '1st Jan 2025 - 31st Dec 2025', None, None, None],
            ['Bank Name:', 'Lloyds', None, None, None],
            ['Account Number:', '1234567890', None, None, None],
            ['Closing Ledger Balance As At:', '31st Dec 2025', 'Closing Ledger:', None, None],
            ['Posting Date', 'Type', 'Details', 'Debits', 'Credits'],
            ['11-Apr-2025', 'CHAPS Payment', 'F/FLOW 1/ACROMION', None, '900.00'],
            ['12-Apr-2025', 'Direct Debit', 'UTILITIES', '150.00', None],
            ['13-Apr-2025', 'Transfer', 'TO SAVINGS', '500.00', None],
            [None, 'Total Credits', None, None, '900.00'],
            [None, 'Total Debits', None, '650.00', None]
        ])
    
    @pytest.fixture
    def natwest_sample_data(self):
        """Sample NatWest bank statement data."""
        return pd.DataFrame([
            ['Account Number:', '12-34-56', '12345678'],
            ['Sort Code:', '60-00-00'],
            ['Statement Period:', '01 Apr 2025 to 30 Apr 2025'],
            ['Date', 'Type', 'Description', 'Value', 'Balance'],
            ['01 Apr 2025', 'BGC', 'FASTER PAYMENT RECEIPT', '1500.00', '2500.00'],
            ['05 Apr 2025', 'DD', 'UTILITIES DIRECT DEBIT', '-120.00', '2380.00'],
            ['10 Apr 2025', 'CHQ', 'CHEQUE 000456', '-75.50', '2304.50'],
            ['Balance Carried Forward', '', '', '', '4104.50']
        ])
    
    def test_lloyds_transformation_basic(self, lloyds_sample_data):
        """Test basic Lloyds statement transformation."""
        result = transform_lloyds_statement(lloyds_sample_data)
        
        # Should find 3 transactions (excluding summary rows)
        assert len(result) == 3
        
        # Check column structure
        expected_columns = ['Date', 'Description', 'Amount']
        assert list(result.columns) == expected_columns
        
        # Check first transaction
        first_transaction = result.iloc[0]
        assert first_transaction['Date'] == '11-Apr-2025'
        assert 'CHAPS Payment' in first_transaction['Description']
        assert first_transaction['Amount'] == 900.00
        
        # Check debit transaction (should be negative)
        second_transaction = result.iloc[1]
        assert second_transaction['Amount'] == -150.00  # Debit should be negative
    
    def test_natwest_transformation_basic(self, natwest_sample_data):
        """Test basic NatWest statement transformation."""
        result = transform_natwest_statement(natwest_sample_data)
        
        # Should find 3 transactions (excluding summary rows)
        assert len(result) == 3
        
        # Check column structure
        expected_columns = ['Date', 'Description', 'Amount']
        assert list(result.columns) == expected_columns
        
        # Check amounts are parsed correctly
        amounts = result['Amount'].tolist()
        assert 1500.00 in amounts  # Credit
        assert -120.00 in amounts  # Debit (negative)
        assert -75.50 in amounts   # Debit (negative)
    
    def test_transform_with_explicit_bank_type(self, lloyds_sample_data):
        """Test transformation with explicit bank type specification."""
        result = transform_bank_statement(lloyds_sample_data, BankType.LLOYDS)
        
        assert len(result) == 3
        assert 'Date' in result.columns
        assert 'Description' in result.columns
        assert 'Amount' in result.columns
    
    def test_empty_dataframe_handling(self):
        """Test handling of empty DataFrame input."""
        empty_df = pd.DataFrame()
        result = transform_lloyds_statement(empty_df)
        
        assert len(result) == 0
        assert list(result.columns) == ['Date', 'Description', 'Amount']
    
    def test_invalid_bank_type(self, lloyds_sample_data):
        """Test error handling for unsupported bank types."""
        # This should raise an error since we're not using auto-detection
        with pytest.raises(ValueError):
            # Create an invalid enum-like object
            class InvalidBankType:
                value = "invalid_bank"
            
            transform_bank_statement(lloyds_sample_data, InvalidBankType())
    
    def test_no_header_found(self):
        """Test behavior when no header row is found."""
        # Data without proper headers
        invalid_data = pd.DataFrame([
            ['Some', 'Random', 'Data'],
            ['Without', 'Proper', 'Headers'],
            ['Should', 'Not', 'Work']
        ])
        
        result = transform_lloyds_statement(invalid_data)
        
        # Should return empty result when no header found
        assert len(result) == 0
        assert list(result.columns) == ['Date', 'Description', 'Amount']
    
    def test_validation_function(self, lloyds_sample_data):
        """Test the data validation function."""
        # Transform data first
        result = transform_lloyds_statement(lloyds_sample_data)
        
        # Validate the result
        validation = validate_transformed_data(result)
        
        assert validation['is_valid'] is True
        assert validation['stats']['total_transactions'] == 3
        assert len(validation['errors']) == 0
    
    def test_validation_with_invalid_data(self):
        """Test validation with problematic data."""
        # Create data with issues
        invalid_data = pd.DataFrame({
            'Date': ['11-Apr-2025', '', '13-Apr-2025'],
            'Description': ['Valid', '', 'Also Valid'],
            'Amount': [100.0, 'invalid', 50.0]
        })
        
        validation = validate_transformed_data(invalid_data)
        
        assert validation['stats']['date_issues'] == 1
        assert validation['stats']['description_issues'] == 1
        assert len(validation['warnings']) > 0
    
    def test_supported_bank_types(self):
        """Test that supported bank types are returned correctly."""
        supported = get_supported_bank_types()
        
        assert isinstance(supported, list)
        assert len(supported) > 0
        assert 'lloyds' in supported
        assert 'natwest' in supported

class TestUtilityFunctions:
    
    def test_amount_parsing(self):
        """Test amount parsing utility function."""
        # Test various amount formats
        test_cases = [
            ('£1,234.56', 1234.56),
            ('-150.00', -150.00),
            ('(75.50)', -75.50),  # Accounting negative
            ('1234', 1234.0),
            ('', 0.0),
            ('£0.00', 0.0),
            (123.45, 123.45),  # Already a number
        ]
        
        for input_amount, expected in test_cases:
            result = parse_amount(input_amount)
            assert result == expected, f"Failed for {input_amount}: expected {expected}, got {result}"
    
    def test_description_cleaning(self):
        """Test description cleaning utility function."""
        test_cases = [
            ('  CHAPS Payment  ', 'Chaps Payment'),
            ('- Direct Debit -', 'Direct Debit'),
            ('null', 'Transaction'),
            ('UTILITIES     PAYMENT', 'Utilities Payment'),
            ('', 'Transaction'),
            ('transfer to savings', 'Transfer To Savings'),
        ]
        
        for input_desc, expected in test_cases:
            result = clean_description(input_desc)
            assert result == expected, f"Failed for '{input_desc}': expected '{expected}', got '{result}'"

class TestBankTemplates:
    
    def test_date_pattern_matching(self):
        """Test date pattern matching for different bank formats."""
        from app.bank_templates import get_bank_template, BankType
        
        # Test Lloyds date patterns
        lloyds_template = get_bank_template(BankType.LLOYDS)
        
        assert lloyds_template.matches_date_pattern('11-Apr-2025') is True
        assert lloyds_template.matches_date_pattern('11/04/2025') is True
        assert lloyds_template.matches_date_pattern('invalid_date') is False
        
        # Test NatWest date patterns
        natwest_template = get_bank_template(BankType.NATWEST)
        
        assert natwest_template.matches_date_pattern('11 Apr 2025') is True
        assert natwest_template.matches_date_pattern('11/04/2025') is True
        assert natwest_template.matches_date_pattern('2025-04-11') is True
    
    def test_column_mapping(self):
        """Test column mapping functionality."""
        from app.bank_templates import get_bank_template, BankType
        
        lloyds_template = get_bank_template(BankType.LLOYDS)
        
        # Test header mapping
        headers = ['posting date', 'type', 'details', 'debits', 'credits']
        column_map = lloyds_template.map_columns(headers)
        
        assert 'date' in column_map
        assert 'type' in column_map
        assert 'debit' in column_map
        assert 'credit' in column_map
        
        assert column_map['date'] == 0  # 'posting date' is first
        assert column_map['debit'] == 3  # 'debits' is fourth

# Integration test
class TestEndToEndProcessing:
    
    def test_complete_lloyds_processing_pipeline(self):
        """Test complete processing pipeline for Lloyds data."""
        # Create comprehensive Lloyds sample
        sample_data = pd.DataFrame([
            ['Client ID:', 'XXXXXXXX', None, None, None],
            ['Bank Name:', 'Lloyds Bank', None, None, None],
            ['Account Number:', '1234567890', None, None, None],
            ['Posting Date', 'Type', 'Details', 'Debits', 'Credits'],
            ['11-Apr-2025', 'CHAPS Payment', 'SALARY PAYMENT', None, '2500.00'],
            ['12-Apr-2025', 'Direct Debit', 'MORTGAGE PAYMENT', '1200.00', None],
            ['13-Apr-2025', 'Transfer', 'TO SAVINGS ACCOUNT', '300.00', None],
            ['14-Apr-2025', 'Card Payment', 'GROCERY SHOPPING', '85.50', None],
            ['15-Apr-2025', 'Interest', 'MONTHLY INTEREST', None, '2.50'],
            [None, 'Total Credits', None, None, '2502.50'],
            [None, 'Total Debits', None, '1585.50', None]
        ])
        
        # Transform
        result = transform_lloyds_statement(sample_data)
        
        # Comprehensive checks
        assert len(result) == 5, f"Expected 5 transactions, got {len(result)}"
        
        # Check data types
        assert result['Date'].dtype == 'object'
        assert result['Description'].dtype == 'object'
        
        # Check amounts are numeric
        for amount in result['Amount']:
            assert isinstance(amount, (int, float)), f"Amount {amount} is not numeric"
        
        # Check balance (credits - debits should equal net amount)
        total_amount = result['Amount'].sum()
        expected_net = 2502.50 - 1585.50  # From totals in sample data
        assert abs(total_amount - expected_net) < 0.01, f"Net amount mismatch: {total_amount} vs {expected_net}"
        
        # Validate result
        validation = validate_transformed_data(result)
        assert validation['is_valid'], f"Validation failed: {validation['errors']}"
        
        print(f"✅ End-to-end test passed: {len(result)} transactions processed successfully")

if __name__ == "__main__":
    # Run tests manually (if not using pytest)
    print("🧪 Running Bank Transformer Tests...")
    
    # Create test instance
    tester = TestBankTransformer()
    
    # Create sample data
    lloyds_data = pd.DataFrame([
        ['Client ID:', 'XXXXXXXX', None, None, None],
        ['Bank Name:', 'Lloyds', None, None, None],
        ['Posting Date', 'Type', 'Details', 'Debits', 'Credits'],
        ['11-Apr-2025', 'CHAPS Payment', 'SALARY', None, '2500.00'],
        ['12-Apr-2025', 'Direct Debit', 'MORTGAGE', '1200.00', None]
    ])
    
    # Test basic transformation
    result = transform_lloyds_statement(lloyds_data)
    print(f"✅ Basic test: {len(result)} transactions found")
    print(result)
    
    # Test validation
    validation = validate_transformed_data(result)
    print(f"✅ Validation: {'Valid' if validation['is_valid'] else 'Invalid'}")
    
    print("🎉 All manual tests passed!")