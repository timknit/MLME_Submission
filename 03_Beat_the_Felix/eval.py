import os
import json
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
import joblib
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=FutureWarning)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Configurable directories ===
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PARAMS_DIR = os.path.join(PROJECT_ROOT, "NARX_Models", "params")
MODELS_DIR = os.path.join(PROJECT_ROOT, "NARX_Models", "models")
DATA_DIR   = os.path.join(PROJECT_ROOT, "narx_data")
SCALER_PATH = os.path.join(PROJECT_ROOT, "scaler", "scaler_df_all.joblib")

ALL_COLUMNS = [
    "mf_PM", "mf_TM", "Q_g", "w_crystal", "c_in", "T_PM_in", "T_TM_in",
    "T_PM", "T_TM", "c", "d10", "d50", "d90"
]
def compute_metrics(y_true, y_pred, output_names):
    metrics = {}
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    for i, name in enumerate(output_names):
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        mse = mean_squared_error(y_true[:, i], y_pred[:, i])
        rmse = np.sqrt(mse)
        metrics[name] = {"MAE": mae, "MSE": mse}
    return metrics

def print_metrics(metrics_dict, tag=None):
    if tag:
        print(f"Metrics for {tag}:")
    for name, vals in metrics_dict.items():
        print(f"  {name}: MAE={vals['MAE']:.2e} | MSE={vals['MSE']:.2e}")

class MLPNARX(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes, activation="relu",
                 dropout=0.0, batchnorm=False, output_activation=None):
        super().__init__()
        acts = {
            "relu": torch.nn.ReLU(),
            "tanh": torch.nn.Tanh(),
            "sigmoid": torch.nn.Sigmoid(),
            "gelu": torch.nn.GELU()
        }
        layers = []
        last_dim = input_size
        for h in hidden_sizes:
            layers.append(torch.nn.Linear(last_dim, h))
            if batchnorm:
                layers.append(torch.nn.BatchNorm1d(h))
            layers.append(acts[activation])
            if dropout > 0.0:
                layers.append(torch.nn.Dropout(dropout))
            last_dim = h
        layers.append(torch.nn.Linear(last_dim, output_size))
        if output_activation is not None and output_activation in acts:
            layers.append(acts[output_activation])
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def get_test_path(cluster_label):
    cluster_dir = os.path.join(DATA_DIR, f"cluster_{cluster_label}")
    return os.path.join(cluster_dir, "eval_all.csv")

def inverse_transform_subset(y_scaled, output_columns, y_scaler, all_columns):
    n_samples = y_scaled.shape[0]
    dummy = np.zeros((n_samples, len(all_columns)))
    df_dummy = pd.DataFrame(dummy, columns=all_columns)
    for i, col in enumerate(output_columns):
        df_dummy[col] = y_scaled[:, i]
    y_full_inv = y_scaler.inverse_transform(df_dummy.values)
    df_full_inv = pd.DataFrame(y_full_inv, columns=all_columns)
    return df_full_inv[output_columns].values

def plot_model_predictions(model, X_test_tensor, y_test_tensor, y_scaler, y_outputs_names, cluster_label):
    """
    Plot true vs. predicted values for all outputs in a time series style.
    """
    model.eval()
    with torch.no_grad():
        y_pred_tensor = model(X_test_tensor)
    y_pred_scaled = y_pred_tensor.cpu().numpy()
    y_test_scaled = y_test_tensor.cpu().numpy()
    # === Change here: robust inverse scaling ===
    import pandas as pd
    y_pred = inverse_transform_subset(y_pred_scaled, y_outputs_names, y_scaler, ALL_COLUMNS)
    y_true = inverse_transform_subset(y_test_scaled, y_outputs_names, y_scaler, ALL_COLUMNS)

    fig, axes = plt.subplots(y_pred.shape[1], 1, figsize=(14, 8), sharex=True)
    if y_pred.shape[1] == 1:
        axes = [axes]
    for i in range(y_pred.shape[1]):
        ax = axes[i]
        ax.plot(y_true[:, i], label="True", linewidth=1)
        ax.plot(y_pred[:, i], label="Predicted", linestyle='--', linewidth=1)
        ax.set_ylabel(y_outputs_names[i])
        ax.grid(True)
        if i == 0:
            ax.set_title("Model Prediction vs. True (Test Data)")
        if i == y_pred.shape[1] - 1:
            ax.set_xlabel("Index")
    fig.legend(loc='upper right', bbox_to_anchor=(0.98, 0.98))
    plt.tight_layout(rect=[0, 0, 0.95, 1])
    plt.close()

def evaluate_from_config(params_path):
    # Load config
    with open(params_path, "r") as f:
        params = json.load(f)
    # Infer label/model_suffix
    basename = os.path.basename(params_path)
    label = basename.split("_")[2]
    model_suffix = basename[len(f"best_params_{label}"):-5]  # -5 for ".json"
    model_path = os.path.join(MODELS_DIR, f"model_{label}{model_suffix}.pt")
    test_path = get_test_path(label)

    if not (os.path.exists(model_path) and os.path.exists(test_path)):
        print(f"[SKIP] Missing model or test data for cluster {label}{model_suffix}")
        return

    # Load test data
    test_df = pd.read_csv(test_path)
    used_X_columns = params["used_X_columns"]
    output_columns = params["output_columns"]
    X_test = test_df[used_X_columns].values
    y_test = test_df[[f"{y}_target" for y in output_columns]].values
    X_test_tensor = torch.from_numpy(X_test).float().to(DEVICE)

    # Load model
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

    y_pred = inverse_transform_subset(y_pred_scaled, output_columns, y_scaler, ALL_COLUMNS)
    y_true = inverse_transform_subset(y_true_scaled, output_columns, y_scaler, ALL_COLUMNS)

    # Metrics
    metrics = compute_metrics(y_true, y_pred, output_columns)
    print_metrics(metrics, tag=f"Cluster {label}{model_suffix}")

    y_test_tensor = torch.from_numpy(y_true_scaled).float().to(DEVICE)
    plot_model_predictions(
        model,
        X_test_tensor,
        y_test_tensor,
        y_scaler,
        output_columns,
        cluster_label=f"{label}{model_suffix}"
    )




if __name__ == "__main__":
    for fname in sorted(os.listdir(PARAMS_DIR)):
        if fname.startswith("best_params_") and fname.endswith(".json"):
            params_path = os.path.join(PARAMS_DIR, fname)
            evaluate_from_config(params_path)