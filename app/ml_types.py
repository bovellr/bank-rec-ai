# typing.py or inside reconcile.py above the function

from typing import Protocol
import numpy as np
import pandas as pd

class PredictiveModel(Protocol):
    """Type protocol for ML models used in reconciliation."""
    
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> 'PredictiveModel': 
        """Train the model with features X and labels y."""
        ...
    
    def predict(self, X: pd.DataFrame) -> np.ndarray: 
        """Make predictions on features X."""
        ...
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: 
        """Return prediction probabilities for features X."""
        ...