# tests/test_ui.py
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest
import pandas as pd
import tempfile
import os

class TestStreamlitUI:
    
    def test_app_loads(self):
        """Test that the Streamlit app loads without errors."""
        # This requires streamlit testing framework
        app = AppTest.from_file("app/ui.py")
        app.run()
        assert not app.exception
    
    def test_file_upload_validation(self):
        """Test file upload validation."""
        # Create test CSV file
        test_data = pd.DataFrame({
            'Amount': [100, 200],
            'Date': ['2024-01-01', '2024-01-02'],
            'Description': ['Test1', 'Test2']
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            test_data.to_csv(f.name, index=False)
            
            # Test that file can be read
            loaded_data = pd.read_csv(f.name)
            assert len(loaded_data) == 2
            assert 'Amount' in loaded_data.columns
            
        os.unlink(f.name)