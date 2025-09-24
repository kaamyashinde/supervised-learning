# stream_transformers.py
from __future__ import annotations

from decimal import Decimal, getcontext
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

__all__ = [
    "LastDigitStreamTransformer",
    "Times1000Mod2Transformer",
    "StreamFeatureTransformer",
]


class LastDigitStreamTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer for 'last digit stream' method:
    - Converts values robustly to string (via Decimal for stability)
    - Ignores characters that are not [0-9]
    - Takes last digit and returns parity (even=0, odd=1)
    """
    
    def __init__(self):
        getcontext().prec = 50  # robust precision for formatting
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X.apply(self._last_digit_parity)
        elif isinstance(X, pd.Series):
            return self._last_digit_parity(X)
        else:
            # Handle numpy arrays
            return pd.Series(X).apply(self._last_digit_parity)
    
    def _last_digit_parity(self, s):
        if isinstance(s, pd.Series):
            return s.apply(self._single_value_parity).astype(int)
        else:
            return self._single_value_parity(s)
    
    def _single_value_parity(self, x):
        if pd.isna(x):
            return 0
        try:
            d = Decimal(str(x))
            txt = format(d, "f")
            digits = "".join(ch for ch in txt if ch.isdigit())
            if not digits:
                return 0
            return int(digits[-1]) % 2
        except Exception:
            # Fallback: float-format with fixed precision
            try:
                txt = f"{float(x):.15f}"
                digits = "".join(ch for ch in txt if ch.isdigit())
                if not digits:
                    return 0
                return int(digits[-1]) % 2
            except Exception:
                return 0


class Times1000Mod2Transformer(BaseEstimator, TransformerMixin):
    """
    Transformer for 'times 1000 mod 2' method:
    - Multiply by 1000, take floor, and check parity:
      - even -> 0
      - odd -> 1
    """
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X.apply(self._times1000_mod2)
        elif isinstance(X, pd.Series):
            return self._times1000_mod2(X)
        else:
            # Handle numpy arrays
            return pd.Series(X).apply(self._times1000_mod2)
    
    def _times1000_mod2(self, s):
        if isinstance(s, pd.Series):
            x = np.floor(s * 1000.0)
            # use abs before cast to avoid negative mod edge cases
            return (np.abs(x).astype(np.int64) % 2).astype(int)
        else:
            x = np.floor(s * 1000.0)
            return (np.abs(x).astype(np.int64) % 2).astype(int)


class StreamFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that applies a specific stream decoding method to a specific column
    and combines it with other stream columns.
    """
    
    def __init__(self, target_column: str, method: str, other_columns: list = None):
        self.target_column = target_column
        self.method = method
        self.other_columns = other_columns or []
        
        # Initialize the appropriate transformer
        if method == "last_digit_stream":
            self.transformer = LastDigitStreamTransformer()
        elif method == "times1000_mod2":
            self.transformer = Times1000Mod2Transformer()
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Apply transformation to target column
        transformed_col = self.transformer.transform(X[self.target_column])
        transformed_col.name = f"{self.target_column}__{self.method}_bit"
        
        # Combine with other columns
        if self.other_columns:
            other_data = X[self.other_columns]
            result = pd.concat([transformed_col, other_data], axis=1)
        else:
            result = transformed_col.to_frame()
        
        # Ensure numeric and fill NaN
        result = result.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        return result