import pandas as pd
from app.self_learning import SelfLearningManager
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from typing import Optional, Tuple, Any
import joblib
from rapidfuzz.fuzz import token_sort_ratio
import logging
from pathlib import Path
import config
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def generate_features(bank_df: pd.DataFrame, erp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate features for reconciliation using token sort ratio.
    """
    start_time = time.time()
    logging.info(f"Generating features for {len(bank_df)} bank transactions and {len(erp_df)} ERP transactions")
    
    # Quick validation
    if bank_df.empty or erp_df.empty:
        logging.warning("One or both DataFrames are empty")
        return pd.DataFrame()

    # Validate required columns
    required_cols = ['Amount', 'Date', 'Description']
    for col in required_cols:
        if col not in bank_df.columns:
            raise ValueError(f"Bank DataFrame missing required column: {col}")
        if col not in erp_df.columns:
            raise ValueError(f"ERP DataFrame missing required column: {col}")

    bank_df = bank_df.copy()
    erp_df = erp_df.copy()

    # Ensure date columns are datetime
    bank_df['Date'] = pd.to_datetime(bank_df['Date'], errors='coerce')
    erp_df['Date'] = pd.to_datetime(erp_df['Date'], errors='coerce')
    
    # Check for date parsing issues
    bank_na_dates = bank_df['Date'].isna().sum()
    erp_na_dates = erp_df['Date'].isna().sum()
    if bank_na_dates > 0 or erp_na_dates > 0:
        logging.warning(f"Date parsing issues - Bank: {bank_na_dates}, ERP: {erp_na_dates}")

    features = []
    total_combinations = len(bank_df) * len(erp_df)
    
    # Performance warning for large datasets
    if total_combinations > 50000:
        logging.warning(f"Large dataset: {total_combinations} combinations. Consider filtering data first.")
    elif total_combinations > 100000:
        logging.error(f"Very large dataset: {total_combinations} combinations. Consider pre-filtering by date range or amount.")
        # Optional: Add automatic filtering here if needed
    
    # Generate features for each transaction in the bank and erp dataframes
    processed = 0
    skipped = 0
    
    for i, b in bank_df.iterrows():
        for j, e in erp_df.iterrows():
            try:
                # Handle missing values gracefully
                b_amount = float(b['Amount']) if pd.notna(b['Amount']) else 0.0
                e_amount = float(e['Amount']) if pd.notna(e['Amount']) else 0.0
                b_date = b['Date'] if pd.notna(b['Date']) else pd.Timestamp('1900-01-01')
                e_date = e['Date'] if pd.notna(e['Date']) else pd.Timestamp('1900-01-01')
                
                # Calculate date difference safely
                date_diff = abs((pd.Timedelta(b_date - e_date)).days)
                
                # Early filtering: skip combinations with very large differences
                amount_diff = abs(b_amount - e_amount)
                if amount_diff > abs(b_amount) * 3 and date_diff > 90:
                    skipped += 1
                    continue  # Skip obviously poor matches
                
                features.append({
                    'bank_index': i,
                    'erp_index': j,
                    'amount_difference': amount_diff,
                    'date_difference': date_diff,
                    'description_similarity': token_sort_ratio(str(b['Description']), str(e['Description'])),
                    'signed_amount_match': int((b_amount > 0) == (e_amount > 0)),
                    'same_day': int(b_date.date() == e_date.date()) if b_date != pd.Timestamp('1900-01-01') and e_date != pd.Timestamp('1900-01-01') else 0  # type: ignore
                })
                
                processed += 1
                # Progress logging for large datasets
                if processed % 10000 == 0:
                    elapsed = time.time() - start_time
                    logging.info(f"Processed {processed} combinations in {elapsed:.1f}s (skipped {skipped})")
                    
            except Exception as ex:
                logging.warning(f"Feature generation failed for pair ({i}, {j}): {ex}")
   
    elapsed = time.time() - start_time
    logging.info(f"Generated {len(features)} valid feature combinations in {elapsed:.1f}s (skipped {skipped} poor matches)")
    return pd.DataFrame(features)

def run_reconciliation(bank_df: pd.DataFrame, erp_df: pd.DataFrame, model: Optional[Any] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run the reconciliation process. Uses ML to match transactions, and anomaly detection for leftovers.
    """
    start_time = time.time()
    logging.info("Starting reconciliation process")
    
    # Input validation
    if bank_df.empty:
        raise ValueError("Bank DataFrame is empty")
    if erp_df.empty:
        raise ValueError("ERP DataFrame is empty")
    
    # Generate features
    features = generate_features(bank_df, erp_df)
    if features.empty:
        raise ValueError("No features generated — check data consistency.")
    
    # Drop the index columns
    X = features.drop(columns=['bank_index', 'erp_index'])
    
    # Handle any infinite or very large values
    X = X.replace([float('inf'), float('-inf')], 0)
    X = X.fillna(0)

    # Enhanced model loading with better error handling
    if model is None:
        try:
            model = joblib.load(config.MODEL_PATH)
            logging.info("Loaded existing model successfully")
        except FileNotFoundError:
            logging.info("No existing model found, creating new RandomForest model")
            # Create a simple model if no trained model exists
            model = RandomForestClassifier(
                n_estimators=100, 
                random_state=42,
                max_depth=10,  # Prevent overfitting
                min_samples_split=5
            )
            # Train with dummy data or use unsupervised approach
            dummy_y = [0] * len(X)
            model.fit(X, dummy_y)
            logging.warning("Model trained with dummy data - consider training with real labeled data")
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise

    # Enhanced prediction with better error handling
    if model is not None and hasattr(model, "predict_proba"):
        try:
            proba_results = model.predict_proba(X)
            if proba_results.shape[1] > 1:
                features["match_confidence"] = proba_results[:, 1]
            else:
                features["match_confidence"] = proba_results[:, 0]
            logging.info("Successfully generated prediction probabilities")
        except Exception as e:
            logging.error(f"predict_proba failed: {e}, falling back to predict")
            predictions = model.predict(X)
            features["match_confidence"] = predictions.astype(float)
    else:
        raise ValueError("Model is None or does not support predict_proba")

    # Apply threshold with validation
    threshold = config.MATCH_CONFIDENCE_THRESHOLD
    if not 0 <= threshold <= 1:
        logging.warning(f"Invalid threshold {threshold}, using default 0.7")
        threshold = 0.7
        
    features["is_match"] = (features["match_confidence"] >= threshold).astype(int)
    matched = features[features["is_match"] == 1]
    
    logging.info(f"Found {len(matched)} matches using threshold {threshold}")

    # Build matched report with additional context
    matched_report = pd.DataFrame()
    if not matched.empty:
        # Add quality metrics to the report
        matched_report["match_confidence"] = matched["match_confidence"].tolist()
        matched_report["description_similarity"] = matched["description_similarity"].tolist()
        matched_report["amount_difference"] = matched["amount_difference"].tolist()
        matched_report["date_difference"] = matched["date_difference"].tolist()
        
        matched_report = pd.concat([
            bank_df.loc[matched["bank_index"]].reset_index(drop=True).add_prefix("bank_"),
            erp_df.loc[matched["erp_index"]].reset_index(drop=True).add_prefix("erp_"),
            matched_report
        ], axis=1)

    # Unmatched = bank rows not found in match
    matched_indices = set(matched["bank_index"])
    unmatched_indices = set(bank_df.index) - matched_indices
    unmatched = bank_df.loc[list(unmatched_indices)].copy()
    
    logging.info(f"Found {len(unmatched)} unmatched transactions")

    # Enhanced Anomaly Detection
    if not unmatched.empty:
        try:
            # Use adaptive contamination rate based on data size
            contamination_rate = min(0.1, max(0.05, 10.0 / len(unmatched)))
            iso_model = IsolationForest(contamination=str(contamination_rate), random_state=42)
            
            # Use only Amount for anomaly detection (could be expanded)
            anomaly_features = unmatched[["Amount"]].fillna(unmatched["Amount"].median())
            unmatched["anomaly"] = iso_model.fit_predict(anomaly_features)
            
            anomaly_count = (unmatched["anomaly"] == -1).sum()
            logging.info(f"Detected {anomaly_count} anomalous transactions (rate: {contamination_rate:.1%})")
        except Exception as e:
            logging.error(f"Anomaly detection failed: {e}")
            unmatched["anomaly"] = 0

    # Enhanced file operations with directory creation and backup
    try:
        # Ensure output directories exist
        Path(config.MATCHED_REPORT_PATH).parent.mkdir(parents=True, exist_ok=True)
        Path(config.UNMATCHED_REPORT_PATH).parent.mkdir(parents=True, exist_ok=True)
        Path(config.REPORT_FILE).parent.mkdir(parents=True, exist_ok=True)
        
        # Create backup of existing files if they exist
        if Path(config.MATCHED_REPORT_PATH).exists():
            backup_path = str(config.MATCHED_REPORT_PATH) + ".backup"
            Path(config.MATCHED_REPORT_PATH).rename(backup_path)
            logging.info("Created backup of existing matched report")
        
        # Save reports
        matched_report.to_csv(config.MATCHED_REPORT_PATH, index=False)
        unmatched.to_csv(config.UNMATCHED_REPORT_PATH, index=False)
        logging.info("CSV reports saved successfully")
    except Exception as e:
        logging.error(f"Failed to save CSV reports: {e}")

    # Enhanced model saving with metadata
    try:
        # Save model with some metadata
        model_data = {
            'model': model,
            'threshold': threshold,
            'feature_columns': X.columns.tolist(),
            'save_timestamp': pd.Timestamp.now().isoformat()
        }
        joblib.dump(model_data, config.MODEL_PATH)
        logging.info("Model saved successfully with metadata")
    except Exception as e:
        logging.error(f"Failed to save model: {e}")

    
    total_elapsed = time.time() - start_time
    anomaly_count = (unmatched["anomaly"] == -1).sum() if "anomaly" in unmatched.columns else 0
    match_rate = round((len(matched_report) / len(bank_df)) * 100, 2) if len(bank_df) > 0 else 0
    
    summary_df = pd.DataFrame({
        'Metric': [
            'Bank Transactions', 
            'ERP Transactions', 
            'Matches Found', 
            'Unmatched Transactions', 
            'Anomalies Detected',
            'Match Rate (%)',
            'Processing Time (seconds)',
            'Confidence Threshold Used',
            'Average Match Confidence',
            'Features Generated'
        ],
        'Value': [
            len(bank_df), 
            len(erp_df), 
            len(matched_report), 
            len(unmatched),
            anomaly_count,
            match_rate,
            round(total_elapsed, 2),
            threshold,
            round(matched["match_confidence"].mean(), 3) if not matched.empty else 0,
            len(features)
        ]
    })
     
    
    # Enhanced Excel export with better summary
    try:
        with pd.ExcelWriter(config.REPORT_FILE, engine='openpyxl') as writer:
            # Main reports
            if not matched_report.empty:
                matched_report.to_excel(writer, sheet_name="Matched", index=False)
            unmatched.to_excel(writer, sheet_name="Unmatched", index=False)
            
            # Enhanced summary with more metrics
            
            summary_df.to_excel(writer, sheet_name="Summary", index=False)
            
        logging.info(f"Excel report saved successfully with enhanced summary (total time: {total_elapsed:.1f}s)")
    except Exception as e:
        logging.error(f"Failed to save Excel report: {e}")

    

    logging.info(f"Reconciliation completed in {total_elapsed:.1f}s")
    return matched_report, unmatched, summary_df


def run_reconciliation_with_learning(bank_df: pd.DataFrame, erp_df: pd.DataFrame, model: Optional[Any] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list]:
    """
    Enhanced reconciliation with self-learning capabilities.
    Returns: matched_report, unmatched, summary_df, uncertain_cases
    """
    # Initialize self-learning manager
    learning_manager = SelfLearningManager()
    
    # Run normal reconciliation
    start_time = time.time()
    logging.info("Starting reconciliation with self-learning...")
    
    # Generate features (same as before)
    features = generate_features(bank_df, erp_df)
    if features.empty:
        raise ValueError("No features generated — check data consistency.")
    
    X = features.drop(columns=['bank_index', 'erp_index'])
    X = X.replace([float('inf'), float('-inf')], 0).fillna(0)
    
    # Load or create model (same as before)
    if model is None:
        try:
            model_data = joblib.load(config.MODEL_PATH)
            if isinstance(model_data, dict) and 'model' in model_data:
                model = model_data['model']
            else:
                model = model_data
            logging.info("Loaded existing model successfully")
        except FileNotFoundError:
            logging.info("No existing model found, creating new RandomForest model")
            model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10, min_samples_split=5)
            dummy_y = [0] * len(X)
            model.fit(X, dummy_y)
            logging.warning("Model trained with dummy data")
    
    # Enhanced prediction with uncertainty calculation
    predictions, confidence, uncertainty = learning_manager.calculate_uncertainty(model, X)
    
    # Add confidence to features
    features["match_confidence"] = confidence
    features["uncertainty"] = uncertainty
    
    # Apply threshold
    threshold = config.MATCH_CONFIDENCE_THRESHOLD
    features["label"] = (features["match_confidence"] >= threshold).astype(int)
    matched = features[features["label"] == 1]
    
    # Identify uncertain cases for human review
    uncertain_cases = learning_manager.identify_uncertain_cases(features, predictions, confidence)
    
    logging.info(f"Found {len(matched)} matches, {len(uncertain_cases)} uncertain cases need review")
    
    # Continue with normal reconciliation process...
    # (Rest of the function remains the same as your current run_reconciliation)
    
    # Build matched report
    matched_report = pd.DataFrame()
    if not matched.empty:
        matched_report["match_confidence"] = matched["match_confidence"].tolist()
        matched_report["uncertainty"] = matched["uncertainty"].tolist()  # Add uncertainty info
        matched_report["description_similarity"] = matched["description_similarity"].tolist()
        matched_report["amount_difference"] = matched["amount_difference"].tolist()
        matched_report["date_difference"] = matched["date_difference"].tolist()
        
        matched_report = pd.concat([
            bank_df.loc[matched["bank_index"]].reset_index(drop=True).add_prefix("bank_"),
            erp_df.loc[matched["erp_index"]].reset_index(drop=True).add_prefix("erp_"),
            matched_report
        ], axis=1)
    
    # Unmatched transactions
    matched_indices = set(matched["bank_index"])
    unmatched_indices = set(bank_df.index) - matched_indices
    unmatched = bank_df.loc[list(unmatched_indices)].copy()
    
    # Anomaly detection (same as before)
    if not unmatched.empty:
        try:
            contamination_rate = min(0.1, max(0.05, 10.0 / len(unmatched)))
            iso_model = IsolationForest(contamination=str(contamination_rate), random_state=42)
            anomaly_features = unmatched[["Amount"]].fillna(unmatched["Amount"].median())
            unmatched["anomaly"] = iso_model.fit_predict(anomaly_features)
            anomaly_count = int((unmatched["anomaly"] == -1).sum())
            logging.info(f"Detected {anomaly_count} anomalous transactions")
        except Exception as e:
            logging.error(f"Anomaly detection failed: {e}")
            unmatched["anomaly"] = 0
    
    # Create enhanced summary with learning statistics
    total_elapsed = time.time() - start_time
    learning_stats = learning_manager.get_learning_statistics()
    
    summary_df = pd.DataFrame({
        'Metric': [
            'Bank Transactions', 'ERP Transactions', 'Matches Found', 'Unmatched Transactions',
            'Uncertain Cases', 'Average Confidence', 'Processing Time (seconds)',
            'Total Feedback Entries', 'Model Retrains', 'Latest Accuracy'
        ],
        'Value': [
            len(bank_df), len(erp_df), len(matched_report), len(unmatched),
            len(uncertain_cases),
            round(matched["match_confidence"].mean(), 3) if not matched.empty else 0,
            round(total_elapsed, 2),
            learning_stats['total_feedback_entries'],
            learning_stats['total_retrains'],
            round(learning_stats['latest_accuracy'], 3)
        ]
    })
    
    # Save files (same as before)
    try:
        Path(config.MATCHED_REPORT_PATH).parent.mkdir(parents=True, exist_ok=True)
        matched_report.to_csv(config.MATCHED_REPORT_PATH, index=False)
        unmatched.to_csv(config.UNMATCHED_REPORT_PATH, index=False)
        
        with pd.ExcelWriter(config.REPORT_FILE, engine='openpyxl') as writer:
            matched_report.to_excel(writer, sheet_name="Matched", index=False)
            unmatched.to_excel(writer, sheet_name="Unmatched", index=False)
            summary_df.to_excel(writer, sheet_name="Summary", index=False)
            
            # Add uncertain cases sheet
            if uncertain_cases:
                uncertain_df = pd.DataFrame(uncertain_cases)
                uncertain_df.to_excel(writer, sheet_name="Uncertain_Cases", index=False)
        
        logging.info("Reports saved successfully")
    except Exception as e:
        logging.error(f"Failed to save reports: {e}")
    
    return matched_report, unmatched, summary_df, uncertain_cases
