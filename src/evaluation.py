"""
Evaluation metrics for Sox2 expression predictions.
"""

import numpy as np
from typing import Tuple, Dict
from scipy import stats


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate MAE."""
    return np.mean(np.abs(y_true - y_pred))


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate MSE."""
    return np.mean((y_true - y_pred) ** 2)


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate RMSE."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def pearson_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """
    Calculate Pearson correlation coefficient.
    
    Returns:
        (correlation, p_value)
    """
    r, p = stats.pearsonr(y_true, y_pred)
    return r, p


def expression_level_accuracy(
    y_true: np.ndarray, y_pred: np.ndarray, tolerance: float = 0.1
) -> float:
    """
    Classify predictions as: no expression (0), WT (1), or overexpression (>1).
    Calculate accuracy within tolerance.
    """
    return np.mean(np.abs(y_true - y_pred) < tolerance)


def evaluate_predictions(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Comprehensive evaluation of predictions.
    
    Returns:
        Dictionary of evaluation metrics
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r, p_value = pearson_correlation(y_true, y_pred)
    acc = expression_level_accuracy(y_true, y_pred)

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "pearson_r": r,
        "p_value": p_value,
        "classification_accuracy": acc,
    }
