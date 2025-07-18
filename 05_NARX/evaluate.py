import os
import json
import torch
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from src.models.narx import MLPNARX
from src.analysis.evaluator import compute_metrics, print_metrics
from src.config import ALL_COLUMNS, SCRIPT_DIR,PROJECT_ROOT,MODELS_DIR,PARAMS_DIR,DATA_DIR,SCALER_PATH

import joblib

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def get_test_path(cluster_label):
    cluster_dir = os.path.join(DATA_DIR, f"cluster_{cluster_label}")
    return os.path.join(cluster_dir, "test_all.csv")

def inverse_transform_subset(y_scaled, output_columns, y_scaler, all_columns):
    n_samples = y_scaled.shape[0]
    dummy = np.zeros((n_samples, len(all_columns)))
    df_dummy = pd.DataFrame(dummy, columns=all_columns)
    for i, col in enumerate(output_columns):
        df_dummy[col] = y_scaled[:, i]
    y_full_inv = y_scaler.inverse_transform(df_dummy.values)
    df_full_inv = pd.DataFrame(y_full_inv, columns=all_columns)
    return df_full_inv[output_columns].values

def evaluate_from_config(params_path):
    # Load config JSON
    with open(params_path, "r") as f:
        params = json.load(f)
    # Extract label and model_suffix from filename
    basename = os.path.basename(params_path)
    label = basename.split("_")[2]
    model_suffix = basename[len(f"best_params_{label}"):-5]  # -5 for ".json"
    model_path = os.path.join(MODELS_DIR, f"model_{label}{model_suffix}.pt")
    test_path = get_test_path(label)

    if not (os.path.exists(model_path) and os.path.exists(test_path)):
        print(f"[SKIP] Missing model or test data for cluster {label}{model_suffix}")
        return

    # Prepare test data using saved columns
    test_df = pd.read_csv(test_path)
    used_X_columns = params["used_X_columns"]
    output_columns = params["output_columns"]

    X_test = test_df[used_X_columns].values
    y_test = test_df[[f"{y}_target" for y in output_columns]].values
    X_test_tensor = torch.from_numpy(X_test).float().to(DEVICE)

    # Reconstruct model and load weights
    model = MLPNARX(
        X_test.shape[1],
        len(output_columns),
        params["hidden_sizes"],
        params["activation"],
        dropout=params.get("dropout", 0.0),
        batchnorm=params.get("batchnorm", False),
        output_activation=None
    ).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    # Predict
    with torch.no_grad():
        y_pred_scaled = model(X_test_tensor).cpu().numpy()
    y_true_scaled = y_test

    # Load scaler
    try:
        y_scaler = joblib.load(SCALER_PATH)
    except Exception:
        print(f"[Warning] No global scaler found at {SCALER_PATH}. Using dummy (no inverse scaling).")
        class DummyScaler:
            def inverse_transform(self, x): return x
        y_scaler = DummyScaler()

    # Inverse-transform with dummy logic
    y_pred = inverse_transform_subset(y_pred_scaled, output_columns, y_scaler, ALL_COLUMNS)
    y_true = inverse_transform_subset(y_true_scaled, output_columns, y_scaler, ALL_COLUMNS)

    # Metrics
    metrics = compute_metrics(y_true, y_pred, output_columns)
    print_metrics(metrics, tag=f"Cluster {label}{model_suffix}")

if __name__ == "__main__":
    # Loop through all best_params_*.json configs
    for fname in sorted(os.listdir(PARAMS_DIR)):
        if fname.startswith("best_params_") and fname.endswith(".json"):
            params_path = os.path.join(PARAMS_DIR, fname)
            evaluate_from_config(params_path)
