# tests/test_reconcile.py
import pytest
import pandas as pd
import numpy as np
from app.reconcile import generate_features, run_reconciliation
from unittest.mock import patch, MagicMock
import tempfile
import os

class TestReconciliation:
    
    @pytest.fixture
    def sample_bank_data(self):
        """Create sample bank transaction data for testing."""
        return pd.DataFrame({
            'Amount': [100.00, -50.00, 200.00, 75.50],
            'Date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04'],
            'Description': ['Payment to ABC Corp', 'ATM Withdrawal', 'Salary Credit', 'Grocery Store']
        })
    
    @pytest.fixture
    def sample_erp_data(self):
        """Create sample ERP transaction data for testing."""
        return pd.DataFrame({
            'Amount': [100.00, 200.00, 80.00, -45.00],
            'Date': ['2024-01-01', '2024-01-03', '2024-01-05', '2024-01-02'],
            'Description': ['ABC Corp Payment', 'Salary Payment', 'Store Purchase', 'Cash Withdrawal']
        })
    
    def test_generate_features_basic(self, sample_bank_data, sample_erp_data):
        """Test basic feature generation functionality."""
        features = generate_features(sample_bank_data, sample_erp_data)
        
        # Check that features were generated
        assert not features.empty
        assert len(features) <= len(sample_bank_data) * len(sample_erp_data)
        
        # Check required columns exist
        required_cols = ['bank_index', 'erp_index', 'amount_difference', 
                        'date_difference', 'description_similarity', 
                        'signed_amount_match', 'same_day']
        for col in required_cols:
            assert col in features.columns
    
    def test_generate_features_empty_data(self):
        """Test feature generation with empty data."""
        empty_df = pd.DataFrame()
        bank_data = pd.DataFrame({'Amount': [100], 'Date': ['2024-01-01'], 'Description': ['Test']})
        
        # Test empty bank data
        features = generate_features(empty_df, bank_data)
        assert features.empty
        
        # Test empty ERP data
        features = generate_features(bank_data, empty_df)
        assert features.empty
    
    def test_generate_features_missing_columns(self):
        """Test feature generation with missing required columns."""
        invalid_df = pd.DataFrame({'WrongColumn': [1, 2, 3]})
        valid_df = pd.DataFrame({
            'Amount': [100], 
            'Date': ['2024-01-01'], 
            'Description': ['Test']
        })
        
        with pytest.raises(ValueError, match="missing required column"):
            generate_features(invalid_df, valid_df)
    
    def test_feature_calculations(self, sample_bank_data, sample_erp_data):
        """Test that feature calculations are correct."""
        features = generate_features(sample_bank_data, sample_erp_data)
        
        # Test first row calculations (bank[0] vs erp[0])
        first_row = features.iloc[0]
        
        # Amount difference should be 0 (100.00 - 100.00)
        assert first_row['amount_difference'] == 0.0
        
        # Same day should be 1 (both 2024-01-01)
        assert first_row['same_day'] == 1
        
        # Description similarity should be > 0 (similar descriptions)
        assert first_row['description_similarity'] > 0
        
        # Signed amount match should be 1 (both positive)
        assert first_row['signed_amount_match'] == 1
    
    @patch('app.reconcile.joblib.load')
    @patch('app.reconcile.config')
    def test_run_reconciliation_with_mock_model(self, mock_config, mock_joblib_load, 
                                               sample_bank_data, sample_erp_data):
        """Test reconciliation with mocked model."""
        # Setup mocks
        mock_config.MATCH_CONFIDENCE_THRESHOLD = 0.7
        mock_config.MATCHED_REPORT_PATH = '/tmp/test_matched.csv'
        mock_config.UNMATCHED_REPORT_PATH = '/tmp/test_unmatched.csv'
        mock_config.REPORT_FILE = '/tmp/test_report.xlsx'
        mock_config.MODEL_PATH = '/tmp/test_model.pkl'
        
        # Mock model
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7], [0.8, 0.2], [0.6, 0.4], [0.9, 0.1]])
        mock_joblib_load.return_value = mock_model
        
        # Run reconciliation
        with patch('app.reconcile.Path'), \
             patch('builtins.open', create=True), \
             patch('app.reconcile.pd.ExcelWriter'):
            
            matched, unmatched, summary = run_reconciliation(sample_bank_data, sample_erp_data)
            
            # Verify model was called
            mock_model.predict_proba.assert_called_once()
            
            # Check return types
            assert isinstance(matched, pd.DataFrame)
            assert isinstance(unmatched, pd.DataFrame)
            assert isinstance(summary, pd.DataFrame)
    
    def test_data_quality_validation(self):
        """Test handling of poor quality data."""
        # Data with NaN values
        dirty_data = pd.DataFrame({
            'Amount': [100.00, np.nan, 200.00],
            'Date': ['2024-01-01', '2024-01-02', np.nan],
            'Description': ['Good', None, 'Also Good']
        })
        
        clean_data = pd.DataFrame({
            'Amount': [100.00],
            'Date': ['2024-01-01'],
            'Description': ['Match']
        })
        
        # Should not crash with dirty data
        features = generate_features(dirty_data, clean_data)
        assert not features.empty