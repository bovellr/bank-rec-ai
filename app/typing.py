# typing.py or inside reconcile.py above the function

from typing import Protocol
import numpy as np
import pandas as pd

class PredictiveModel(Protocol):
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> 'PredictiveModel': ...
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...
