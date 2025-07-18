import os
import torch
import pandas as pd
import numpy as np
import json
from torch.utils.data import TensorDataset, DataLoader
import re
from src.models.narx import MLPNARX 
from src.config import INPUT_COLUMNS
from src.utils.visualization import plot_model_predictions, parity_plot
from src.config import X_INPUTS_NAMES, ALL_COLUMNS
from evaluate import inverse_transform_subset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def load_narx_as_tensors(path, X_Inputs_names, y_outputs_names, lag):
    df = pd.read_csv(path)
    # Regular expression for allowed lag indices (e.g., lag=4 keeps _lag0, _lag1, _lag2, _lag3)
    lag_pattern = re.compile(r"_lag(\d+)$")
    X_cols = []
    for col in df.columns:
        # Check if col is an input or output lagged feature
        if any([col.startswith(x + "_lag") for x in X_Inputs_names]) or any([col.startswith(y + "_lag") for y in y_outputs_names]):
            # Extract lag number
            m = lag_pattern.search(col)
            if m and int(m.group(1)) < lag:
                X_cols.append(col)
    # Targets as before
    y_cols = [f"{y}_target" for y in y_outputs_names]
    X = df[X_cols].values
    y = df[y_cols].values
    return X, y

def train_cluster(
    train_path,
    val_path,
    test_path,
    label, lag, hidden_sizes, loss_function, activation,
    epochs=100, batch_size=128, lr=1e-3,
    l1_lambda=0.0, l2_lambda=0.0, dropout=0.0, batchnorm=False,
    output_activation=None, patience=10, verbose=True,
    output_columns=None, model_suffix="",
    models_dir="models",
    params_dir="best_params",
    return_val_loss=False
):
    
    SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
    MODELS_DIR  = os.path.join(PROJECT_ROOT, models_dir)
    PARAMS_DIR  = os.path.join(PROJECT_ROOT, params_dir)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PARAMS_DIR, exist_ok=True)
    PLOT_DIR    = os.path.join(PROJECT_ROOT, "plots")
    os.makedirs(PLOT_DIR, exist_ok=True)
    if output_columns is None:
        raise ValueError("output_columns must be specified.")

    for path in (train_path, val_path, test_path):
        if not os.path.exists(path):
            print(f"Missing file: {path}")
            return None
    #print("Aktuelles Gerät:", torch.cuda.current_device())
    # === NEW: use dynamic lagged selection ===
    X_train, y_train = load_narx_as_tensors(train_path, X_INPUTS_NAMES, output_columns, lag)
    X_val, y_val = load_narx_as_tensors(val_path, X_INPUTS_NAMES, output_columns, lag)
    X_test, y_test = load_narx_as_tensors(test_path, X_INPUTS_NAMES, output_columns, lag)
    # --- Collect actual input columns used ---
    train_df = pd.read_csv(train_path)
    lag_pattern = re.compile(r"_lag(\d+)$")
    used_X_columns = []
    for col in train_df.columns:
        if any([col.startswith(x + "_lag") for x in X_INPUTS_NAMES]) or any([col.startswith(y + "_lag") for y in output_columns]):
            m = lag_pattern.search(col)
            if m and int(m.group(1)) < lag:
                used_X_columns.append(col)
    if X_train.shape[0] == 0 or X_val.shape[0] == 0 or X_test.shape[0] == 0:
        print(f"Not enough samples for cluster {label}")
        return None

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_ds   = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    test_ds  = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader   = DataLoader(val_ds, batch_size=batch_size)
    test_loader  = DataLoader(test_ds, batch_size=batch_size)

    model = MLPNARX(
        X_train.shape[1],
        y_train.shape[1],
        hidden_sizes,
        activation,
        dropout=dropout,
        batchnorm=batchnorm,
        output_activation=output_activation
    ).to(DEVICE)
    loss_fn = {"mse": torch.nn.MSELoss(), "mae": torch.nn.L1Loss(), "huber": torch.nn.SmoothL1Loss()}[loss_function]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=l2_lambda)

    best_loss = float("inf")
    wait = 0
    best_state = None
    for epoch in range(1, epochs+1):
        model.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            output = model(Xb)
            loss = loss_fn(output, yb)
            if l1_lambda > 0.0:
                l1_penalty = sum(param.abs().sum() for param in model.parameters())
                loss = loss + l1_lambda * l1_penalty
            loss.backward()
            optimizer.step()
        # Validation loss
        model.eval()
        total_val = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                output = model(Xb)
                val_loss = loss_fn(output, yb)
                if l1_lambda > 0.0:
                    l1_penalty = sum(param.abs().sum() for param in model.parameters())
                    val_loss = val_loss + l1_lambda * l1_penalty
                total_val += val_loss.item() * Xb.size(0)
        avg_val = total_val / len(val_ds)
        if verbose and (epoch % 10 == 0 or epoch == 1 or epoch == epochs):
            print(f"[{label}{model_suffix}] [Epoch {epoch}] Train/Val Loss: {loss.item():.4e}/{avg_val:.4e}")
        if avg_val < best_loss:
            best_loss = avg_val
            wait = 0
            best_state = model.state_dict()
        else:
            wait += 1
            if wait >= patience:
                #print(f"Early stopping at epoch {epoch}")
                break

        if best_state is not None:
            model.load_state_dict(best_state)
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"model_{label}{model_suffix}.pt"))
            #print(f"Best model weights for cluster {label}{model_suffix} saved.")

            # Save config JSON for reproducibility

            # --- WICHTIG: Speichere die tatsächlich genutzten y-Output-Spalten ---
            used_y_columns = [f"{y}_target" for y in output_columns]

            params = {
                "lag": lag,
                "hidden_sizes": hidden_sizes,
                "activation": activation,
                "input_columns": X_INPUTS_NAMES,
                "output_columns": output_columns,           # Deine "logische" Output-Liste (z.B. ['T_PM', 'T_TM'])
                "used_X_columns": used_X_columns,           # Exakt die Spalten im DataFrame
                "used_y_columns": used_y_columns,           # Exakt die Output-Spalten im DataFrame
                "loss_function": loss_function,
                "batch_size": batch_size,
                "dropout": dropout,
                "batchnorm": batchnorm,
            }

        with open(os.path.join(PARAMS_DIR, f"best_params_{label}{model_suffix}.json"), "w") as f:
            json.dump(params, f)
        #print(f"Config for cluster {label}{model_suffix} saved.")

    if return_val_loss:
        return best_loss
    
    # --- Test Evaluation + Parity Plot ---
    model.eval()
    total_test = 0.0
    all_y_true = []
    all_y_pred = []
    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            output = model(Xb)
            test_loss = loss_fn(output, yb)
            total_test += test_loss.item() * Xb.size(0)
            all_y_true.append(yb.cpu().numpy())
            all_y_pred.append(output.cpu().numpy())
    avg_test = total_test / len(test_ds)
    print(f"Test Loss (cluster {label}{model_suffix}): {avg_test:.6e}")

    # --- Load output scaler if available ---
    # --- Load global output scaler if available ---
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../"))
    SCALER_PATH = os.path.join(PROJECT_ROOT, "data_preprocessing", "scaler", "scaler_df_all.joblib")
    try:
        import joblib
        y_scaler = joblib.load(SCALER_PATH)
    except Exception:
        print(f"[Warning] No global scaler found at {SCALER_PATH}. Plotting scaled predictions.")
        class DummyScaler:
            def inverse_transform(self, x): return x
        y_scaler = DummyScaler()
    X_test_tensor = torch.from_numpy(X_test).float().to(DEVICE)
    y_test_tensor = torch.from_numpy(y_test).float().to(DEVICE)
    plot_model_predictions(
    model, X_test_tensor, y_test_tensor, y_scaler, output_columns,
    cluster_label=f"{label}{model_suffix}", outdir=PLOT_DIR
    )

    y_true = inverse_transform_subset(np.concatenate(all_y_true, axis=0), output_columns, y_scaler, ALL_COLUMNS)
    y_pred = inverse_transform_subset(np.concatenate(all_y_pred, axis=0), output_columns, y_scaler, ALL_COLUMNS)

    parity_plot(y_true, y_pred, output_columns, f"{label}{model_suffix}", outdir=PLOT_DIR)

