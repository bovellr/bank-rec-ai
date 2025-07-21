import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import logging
from datetime import datetime
from pathlib import Path
import json
import sys


sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    import config
except ImportError:
    # Fallback config
    class Config:
        MODEL_PATH = "model/reconciliation_model.pkl"
        MATCH_CONFIDENCE_THRESHOLD = 0.7
        MATCHED_REPORT_PATH = "output/matched_transactions.csv"
        UNMATCHED_REPORT_PATH = "output/unmatched_transactions.csv"
        REPORT_FILE = "output/reconciliation_report.xlsx"
    config = Config()

    
class SelfLearningManager:
    """
    Manages self-learning capabilities for the reconciliation system.
    """
    
    def __init__(self):
        self.feedback_file = Path("data/feedback_history.json")
        self.performance_file = Path("data/performance_history.json")
        self.uncertain_cases_file = Path("data/uncertain_cases.json")
        
        # Create data directory if it doesn't exist
        self.feedback_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing data
        self.feedback_history = self.load_feedback_history()
        self.performance_history = self.load_performance_history()
    
    def load_feedback_history(self):
        """Load feedback history from file."""
        if self.feedback_file.exists():
            try:
                with open(self.feedback_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logging.warning(f"Could not load feedback history: {e}")
        return []
    
    def save_feedback_history(self):
        """Save feedback history to file."""
        try:
            with open(self.feedback_file, 'w') as f:
                json.dump(self.feedback_history, f, indent=2, default=str)
        except Exception as e:
            logging.error(f"Could not save feedback history: {e}")
    
    def load_performance_history(self):
        """Load performance history from file."""
        if self.performance_file.exists():
            try:
                with open(self.performance_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logging.warning(f"Could not load performance history: {e}")
        return []
    
    def save_performance_history(self):
        """Save performance history to file."""
        try:
            with open(self.performance_file, 'w') as f:
                json.dump(self.performance_history, f, indent=2, default=str)
        except Exception as e:
            logging.error(f"Could not save performance history: {e}")
    
    def calculate_uncertainty(self, model, X):
        """
        Calculate prediction uncertainty using entropy.
        """
        if not hasattr(model, 'predict_proba'):
            # For models without predict_proba, use decision function or distance from boundary
            predictions = model.predict(X)
            # Simple uncertainty: assume 0.7 confidence for all predictions
            confidence = np.full(len(predictions), 0.7)
            uncertainty = 1 - confidence
            return predictions, confidence, uncertainty
        
        # Get prediction probabilities
        try:
            proba = model.predict_proba(X)
            predictions = model.predict(X)
            
            # Calculate entropy-based uncertainty
            # Higher entropy = more uncertainty
            epsilon = 1e-10  # Prevent log(0)
            entropy = -np.sum(proba * np.log2(proba + epsilon), axis=1)
            
            # Normalize entropy to 0-1 range
            max_entropy = np.log2(proba.shape[1])
            uncertainty = entropy / max_entropy
            
            # Confidence is inverse of uncertainty
            confidence = 1 - uncertainty
            
            return predictions, confidence, uncertainty
            
        except Exception as e:
            logging.error(f"Error calculating uncertainty: {e}")
            predictions = model.predict(X)
            confidence = np.full(len(predictions), 0.7)
            uncertainty = 1 - confidence
            return predictions, confidence, uncertainty
    
    def identify_uncertain_cases(self, features, predictions, confidence, threshold=0.8):
        """
        Identify cases that need human review.
        """
        uncertain_cases = []
        
        # Find low confidence cases
        low_confidence_mask = confidence < threshold
        
        # Find cases near decision boundary (confidence around 0.5)
        boundary_mask = np.abs(confidence - 0.5) < 0.2
        
        # Combine criteria
        review_mask = low_confidence_mask | boundary_mask
        
        for idx in np.where(review_mask)[0]:
            case = {
                'index': int(idx),
                'bank_index': int(features.iloc[idx]['bank_index']) if 'bank_index' in features.columns else idx,
                'erp_index': int(features.iloc[idx]['erp_index']) if 'erp_index' in features.columns else idx,
                'prediction': int(predictions[idx]),
                'confidence': float(confidence[idx]),
                'uncertainty': float(1 - confidence[idx]),
                'reason': 'low_confidence' if low_confidence_mask[idx] else 'boundary_case',
                'needs_review': True,
                'timestamp': datetime.now().isoformat()
            }
            uncertain_cases.append(case)
        
        # Sort by uncertainty (most uncertain first)
        uncertain_cases.sort(key=lambda x: x['uncertainty'], reverse=True)
        
        # Save uncertain cases for UI
        self.save_uncertain_cases(uncertain_cases)
        
        logging.info(f"Identified {len(uncertain_cases)} cases needing review")
        return uncertain_cases
    
    def save_uncertain_cases(self, uncertain_cases):
        """Save uncertain cases for UI display."""
        try:
            with open(self.uncertain_cases_file, 'w') as f:
                json.dump(uncertain_cases, f, indent=2, default=str)
        except Exception as e:
            logging.error(f"Could not save uncertain cases: {e}")
    
    def load_uncertain_cases(self):
        """Load uncertain cases for UI display."""
        if self.uncertain_cases_file.exists():
            try:
                with open(self.uncertain_cases_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logging.warning(f"Could not load uncertain cases: {e}")
        return []
    
    def collect_feedback(self, case_index, bank_row, erp_row, user_decision, confidence, user_comment=""):
        """
        Collect human feedback for a specific case.
        """
        feedback_entry = {
            'timestamp': datetime.now().isoformat(),
            'case_index': case_index,
            'bank_data': {
                'Amount': float(bank_row.get('Amount', 0)),
                'Date': str(bank_row.get('Date', '')),
                'Description': str(bank_row.get('Description', ''))
            },
            'erp_data': {
                'Amount': float(erp_row.get('Amount', 0)),
                'Date': str(erp_row.get('Date', '')),
                'Description': str(erp_row.get('Description', ''))
            },
            'model_prediction': int(case_index),  # You'll need to pass this
            'model_confidence': float(confidence),
            'user_decision': int(user_decision),  # 1 for match, 0 for no match
            'user_comment': str(user_comment),
            'feedback_source': 'manual_review'
        }
        
        self.feedback_history.append(feedback_entry)
        self.save_feedback_history()
        
        logging.info(f"Collected feedback for case {case_index}: user_decision={user_decision}")
        return True
    
    def prepare_retraining_data(self):
        """
        Prepare data for model retraining from feedback history.
        """
        if len(self.feedback_history) < 5:
            logging.info("Not enough feedback for retraining")
            return None, None
        
        # Convert feedback to training features
        features_list = []
        labels_list = []
        
        for feedback in self.feedback_history:
            try:
                # Calculate features similar to reconcile.py
                bank_data = feedback['bank_data']
                erp_data = feedback['erp_data']
                
                # Amount difference
                amount_diff = abs(bank_data['Amount'] - erp_data['Amount'])
                
                # Date difference
                bank_date = pd.to_datetime(bank_data['Date'])
                erp_date = pd.to_datetime(erp_data['Date'])
                date_diff = abs((bank_date - erp_date).days)
                
                # Description similarity (simplified - you can enhance this)
                from rapidfuzz.fuzz import token_sort_ratio
                desc_sim = token_sort_ratio(bank_data['Description'], erp_data['Description'])
                
                # Signed amount match
                bank_positive = bank_data['Amount'] > 0
                erp_positive = erp_data['Amount'] > 0
                signed_match = 1 if bank_positive == erp_positive else 0
                
                # Same day
                same_day = 1 if bank_date.date() == erp_date.date() else 0
                
                features_list.append([
                    amount_diff,
                    date_diff,
                    desc_sim,
                    signed_match,
                    same_day
                ])
                
                labels_list.append(feedback['user_decision'])
                
            except Exception as e:
                logging.warning(f"Error processing feedback entry: {e}")
                continue
        
        if len(features_list) == 0:
            return None, None
        
        X = np.array(features_list, dtype=np.float64)
        y = np.array(labels_list, dtype=np.int32)
        
        logging.info(f"Prepared retraining data: {X.shape[0]} samples")
        return X, y
    
    def retrain_model(self, original_model=None):
        """
        Retrain the model with feedback data.
        """
        # Prepare feedback data
        feedback_X, feedback_y = self.prepare_retraining_data()
        
        if feedback_X is None:
            logging.info("No valid feedback data for retraining")
            return False
        
        try:
            # Load existing model or create new one
            if original_model is not None:
                model = original_model
            else:
                try:
                    model_data = joblib.load(config.MODEL_PATH)
                    if isinstance(model_data, dict) and 'model' in model_data:
                        model = model_data['model']
                    else:
                        model = model_data
                except:
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
            
            # Get original training data if available
            original_X = None
            original_y = None
            
            if hasattr(model, '_training_data'):
                original_X = model._training_data.get('X')
                original_y = model._training_data.get('y')
            
            # Combine original and feedback data
            if original_X is not None and original_y is not None:
                combined_X = np.vstack([original_X, feedback_X])
                combined_y = np.hstack([original_y, feedback_y])
                logging.info(f"Combined training data: {len(original_X)} original + {len(feedback_X)} feedback")
            else:
                combined_X = feedback_X
                combined_y = feedback_y
                logging.info(f"Using only feedback data: {len(feedback_X)} samples")
            
            # Calculate old accuracy if possible
            old_accuracy = 0.0
            if len(feedback_X) > 5:
                try:
                    old_predictions = model.predict(feedback_X)
                    old_accuracy = np.mean(old_predictions == feedback_y)
                except:
                    pass
            
            # Retrain model
            new_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42
            )
            
            new_model.fit(combined_X, combined_y)
            
            # Store training data for future use
            new_model._training_data = {
                'X': combined_X,
                'y': combined_y,
                'feedback_count': len(feedback_X)
            }
            
            # Calculate new accuracy
            new_accuracy = 0.0
            if len(feedback_X) > 5:
                try:
                    new_predictions = new_model.predict(feedback_X)
                    new_accuracy = np.mean(new_predictions == feedback_y)
                except:
                    pass
            
            # Save performance history
            performance_entry = {
                'timestamp': datetime.now().isoformat(),
                'old_accuracy': float(old_accuracy),
                'new_accuracy': float(new_accuracy),
                'improvement': float(new_accuracy - old_accuracy),
                'feedback_samples': len(feedback_X),
                'total_samples': len(combined_X),
                'model_version': f"retrained_{len(self.performance_history) + 1}"
            }
            
            self.performance_history.append(performance_entry)
            self.save_performance_history()
            
            # Save the improved model
            model_data = {
                'model': new_model,
                'threshold': config.MATCH_CONFIDENCE_THRESHOLD,
                'feature_columns': ['amount_difference', 'date_difference', 'description_similarity', 'signed_amount_match', 'same_day'],
                'save_timestamp': datetime.now().isoformat(),
                'training_info': {
                    'total_samples': len(combined_X),
                    'feedback_samples': len(feedback_X),
                    'retrain_count': len(self.performance_history)
                }
            }
            
            joblib.dump(model_data, config.MODEL_PATH)
            
            logging.info(f"Model retrained successfully!")
            logging.info(f"Accuracy improvement: {old_accuracy:.3f} → {new_accuracy:.3f}")
            
            return True
            
        except Exception as e:
            logging.error(f"Model retraining failed: {e}")
            return False
    
    def get_learning_statistics(self):
        """
        Get statistics about the learning process.
        """
        stats = {
            'total_feedback_entries': len(self.feedback_history),
            'total_retrains': len(self.performance_history),
            'latest_accuracy': 0.0,
            'accuracy_trend': 'stable',
            'pending_uncertain_cases': len(self.load_uncertain_cases())
        }
        
        if self.performance_history:
            latest = self.performance_history[-1]
            stats['latest_accuracy'] = latest.get('new_accuracy', 0.0)
            
            if len(self.performance_history) > 1:
                prev = self.performance_history[-2]
                if latest.get('new_accuracy', 0) > prev.get('new_accuracy', 0):
                    stats['accuracy_trend'] = 'improving'
                elif latest.get('new_accuracy', 0) < prev.get('new_accuracy', 0):
                    stats['accuracy_trend'] = 'declining'
        
        return stats
