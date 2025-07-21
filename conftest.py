# conftest.py - pytest configuration
import pytest
import pandas as pd
import tempfile
import os
import sys
from pathlib import Path

# Add app directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'app'))

@pytest.fixture(scope="session")
def temp_config():
    """Create temporary config for testing."""
    temp_dir = tempfile.mkdtemp()
    
    class TempConfig:
        MATCH_CONFIDENCE_THRESHOLD = 0.7
        MATCHED_REPORT_PATH = os.path.join(temp_dir, 'matched.csv')
        UNMATCHED_REPORT_PATH = os.path.join(temp_dir, 'unmatched.csv')
        REPORT_FILE = os.path.join(temp_dir, 'report.xlsx')
        MODEL_PATH = os.path.join(temp_dir, 'model.pkl')
    
    return TempConfig()