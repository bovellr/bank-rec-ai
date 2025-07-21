import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"
OUTPUT_DIR = BASE_DIR / "output"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Model configuration
MATCH_CONFIDENCE_THRESHOLD = 0.7
MODEL_PATH = str(MODEL_DIR / "reconciliation_model.pkl")

# Data file paths
BANK_FILE = str(DATA_DIR / "bank_statement.csv")
ERP_FILE = str(DATA_DIR / "erp_transactions.csv")
TRAINING_FILE = str(DATA_DIR / "training_labels.csv")

# Output file paths
MATCHED_REPORT_PATH = str(OUTPUT_DIR / "matched_transactions.csv")
UNMATCHED_REPORT_PATH = str(OUTPUT_DIR / "unmatched_transactions.csv")
REPORT_FILE = str(OUTPUT_DIR / "reconciliation_report.xlsx")

# Self-learning configuration
FEEDBACK_HISTORY_PATH = str(DATA_DIR / "feedback_history.json")
PERFORMANCE_HISTORY_PATH = str(DATA_DIR / "performance_history.json")
UNCERTAIN_CASES_PATH = str(DATA_DIR / "uncertain_cases.json")

# Logging configuration
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(str(DATA_DIR / 'app.log')),
        logging.StreamHandler()
    ]
)

MODEL_VERSION = "v1.0.0"