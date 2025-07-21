import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import config
import logging
from rapidfuzz.fuzz import token_sort_ratio
from typing import Tuple

logging.basicConfig(level=logging.INFO)


def prepare_training_features(training_df: pd.DataFrame) -> tuple:
    """
    Prepare features from training data.
    
    Expected columns in training_df:
    - bank_amount, erp_amount
    - bank_description, erp_description  
    - bank_date, erp_date
    - label(target variable)
    """
    logging.info(f"Preparing features from {len(training_df)} training samples")
    
    # Validate required columns
    required_columns = [
        'bank_amount', 'erp_amount',
        'bank_description', 'erp_description',
        'bank_date', 'erp_date',
        'label'
    ]
    
    missing_columns = [col for col in required_columns if col not in training_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Create features similar to reconcile.py
    features = []
    
    for _, row in training_df.iterrows():
        try:
            # Amount difference
            amount_diff = abs(float(row['bank_amount']) - float(row['erp_amount']))
            
            # Date difference (handle string dates)
            bank_date = pd.to_datetime(row['bank_date'])
            erp_date = pd.to_datetime(row['erp_date'])
            date_diff = abs((bank_date - erp_date).days)
            
            # Description similarity
            desc_sim = token_sort_ratio(str(row['bank_description']), str(row['erp_description']))
            
            # Signed amount match
            bank_positive = float(row['bank_amount']) > 0
            erp_positive = float(row['erp_amount']) > 0
            signed_match = 1 if bank_positive == erp_positive else 0
            
            # Same day
            same_day = 1 if bank_date.date() == erp_date.date() else 0
            
            features.append([
                amount_diff,
                date_diff,
                desc_sim,
                signed_match,
                same_day
            ])
            
        except Exception as e:
            logging.warning(f"Error processing training row: {e}")
            continue
    
    # Convert to numpy arrays (this is the key fix!)
    X = np.array(features, dtype=np.float64)
    y = np.array(training_df['label'].values, dtype=np.int32)
    
    logging.info(f"Created feature matrix: {X.shape}")
    logging.info(f"Created target vector: {y.shape}")
    
    return X, y

def train_model(training_df: pd.DataFrame):
    """
    Train a machine learning model for transaction matching.
    
    Args:
        training_df: DataFrame with labeled training data
        
    Returns:
        Trained model
    """
    logging.info("Starting model training...")
    
    try:
        # Prepare features - this converts DataFrames to numpy arrays
        X, y = prepare_training_features(training_df)
        
        if len(X) == 0:
            raise ValueError("No valid training features generated")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logging.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        
        # Create and train the model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        
        # X_train and y_train are numpy arrays, not DataFrames
        model.fit(X_train, y_train)
        
        # Evaluate the model
        train_accuracy = model.score(X_train, y_train)
        test_accuracy = model.score(X_test, y_test)
        
        logging.info(f"Training accuracy: {train_accuracy:.3f}")
        logging.info(f"Test accuracy: {test_accuracy:.3f}")
        
        # Make predictions for detailed evaluation
        y_pred = model.predict(X_test)
        
        # Log classification report
        report = classification_report(y_test, y_pred)
        logging.info(f"Classification Report:\n{report}")
        
        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5)
        logging.info(f"Cross-validation Accuracy: {cv_scores.mean():.2f}")

        # Feature importance
        feature_names = ['amount_difference', 'date_difference', 'description_similarity', 
                        'signed_amount_match', 'same_day']
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logging.info("Feature Importance:")
        for _, row in importance_df.iterrows():
            logging.info(f"  {row['feature']}: {row['importance']:.3f}")
        
        # Save the model
        try:
            joblib.dump(model, config.MODEL_PATH)
            logging.info(f"Model saved to {config.MODEL_PATH}")
        except Exception as e:
            logging.warning(f"Failed to save model: {e}")
        
        return model
        
    except Exception as e:
        logging.error(f"Model training failed: {e}")
        raise
