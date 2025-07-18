"""Model evaluation and analysis utilities."""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def compute_metrics(y_true, y_pred, output_names):
    """
    Compute MAE, MSE, RMSE for each output variable.
    
    Args:
        y_true (np.ndarray): shape (N, D) or (N,), true target values.
        y_pred (np.ndarray): shape (N, D) or (N,), predicted values.
        output_names (list of str): List of output variable names.
    Returns:
        dict: {output_name: {"MAE": val, "MSE": val, "RMSE": val}, ...}
    """
    metrics = {}
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    for i, name in enumerate(output_names):
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        mse = mean_squared_error(y_true[:, i], y_pred[:, i])
        rmse = np.sqrt(mse)
        metrics[name] = {"MAE": mae, "MSE": mse, "RMSE": rmse}
    return metrics

def print_metrics(metrics_dict, tag=None):
    """
    Print metrics in a clean format.
    """
    if tag:
        print(f"Metrics for {tag}:")
    for name, vals in metrics_dict.items():
        print(f"  {name}: MAE={vals['MAE']:.2e} | MSE={vals['MSE']:.2e} | RMSE={vals['RMSE']:.2e}")
