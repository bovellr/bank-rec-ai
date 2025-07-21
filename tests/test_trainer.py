# tests/test_trainer.py
import pytest
import pandas as pd
from app.trainer import train_model
from unittest.mock import patch, MagicMock

class TestTrainer:
    
    @pytest.fixture
    def sample_training_data(self):
        """Create sample training data."""
        return pd.DataFrame({
            'bank_amount': [100, 200, 150, 75],
            'erp_amount': [100, 190, 300, 75],
            'bank_description': ['ABC Corp', 'XYZ Ltd', 'Store', 'Gas Station'],
            'erp_description': ['ABC Corporation', 'XYZ Limited', 'Different Store', 'Gas Station'],
            'bank_date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04'],
            'erp_date': ['2024-01-01', '2024-01-02', '2024-01-05', '2024-01-04'],
            'label': [1, 1, 0, 1]
        })
    
    def test_train_model_basic(self, sample_training_data):
        """Test basic model training functionality."""
        with patch('app.trainer.joblib.dump'), \
             patch('app.trainer.config'):
            
            model = train_model(sample_training_data)
            
            # Check that a model was returned
            assert model is not None
            assert hasattr(model, 'fit')
            assert hasattr(model, 'predict')
    
    def test_train_model_invalid_data(self):
        """Test training with invalid data."""
        invalid_data = pd.DataFrame({'wrong_column': [1, 2, 3]})
        
        with pytest.raises(Exception):  # Should raise some kind of error
            train_model(invalid_data)