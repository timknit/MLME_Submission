import os
import json
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from src.models.narx import MLPNARX, create_narx_lagged_data_tim
from src.models.quantile import PinballLoss, make_quantile_net, train_with_early_stopping
from src.analysis.evaluator import compute_metrics, print_metrics
from src.config import ALL_COLUMNS, SCALER_PATH, MODELS_DIR, PARAMS_DIR, PLOT_DIR, CQR_DIR, SCRIPT_DIR, PROJECT_ROOT, DATA_DIR
import joblib
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def inverse_transform_subset(y_scaled, output_columns, y_scaler, all_columns):
    n_samples = y_scaled.shape[0]
    dummy = np.zeros((n_samples, len(all_columns)))
    df_dummy = pd.DataFrame(dummy, columns=all_columns)
    for i, col in enumerate(output_columns):
        df_dummy[col] = y_scaled[:, i]
    y_full_inv = y_scaler.inverse_transform(df_dummy.values)
    df_full_inv = pd.DataFrame(y_full_inv, columns=all_columns)
    return df_full_inv[output_columns].values

def plot_cqr_results(results, plot_path=None, outlier_color="red"):
    n = len(results)
    fig, axes = plt.subplots(n, 1, figsize=(15, 3 * n), sharex=True)
    axes = np.atleast_1d(axes)
    for i, res in enumerate(results):
        ax = axes[i]
        idx = np.arange(len(res["true"]))
        outliers = (res["true"] < res["lower"]) | (res["true"] > res["upper"])
        ax.fill_between(idx, res["lower"], res["upper"], color="#4bd500", alpha=0.5, label="CQR Interval")
        ax.plot(idx, res["true"], label="True", color="black", linewidth=1)
        ax.plot(idx, res["pred"], label="Pred", color="blue", linestyle="--", linewidth=1)
        ax.scatter(idx[outliers], res["true"][outliers], color=outlier_color, label="Outlier", zorder=10, s=8)
        ax.set_ylabel(res["label"])
        ax.legend(loc="upper right")
        ax.set_title(f"CQR: {res['label']}, Outliers: {outliers.sum()} ({100*outliers.mean():.1f}%)")
        ax.grid(True, linestyle='--', alpha=0.7)
    axes[-1].set_xlabel("Sample Index")
    plt.tight_layout()
    if plot_path:
        plt.savefig(plot_path, dpi=150)
        print(f"Plot saved to {plot_path}")
    plt.close()

def evaluate_interval_coverage(y_true, lower_bounds, upper_bounds):
    n_vars = y_true.shape[1]
    coverage = {}
    for i in range(n_vars):
        in_interval = (y_true[:, i] >= lower_bounds[:, i]) & (y_true[:, i] <= upper_bounds[:, i])
        coverage[i] = np.mean(in_interval) * 100
    in_interval_all = np.all([(y_true[:, i] >= lower_bounds[:, i]) & (y_true[:, i] <= upper_bounds[:, i]) for i in range(n_vars)], axis=0)
    coverage['overall'] = np.mean(in_interval_all) * 100
    interval_width = {i: np.mean(upper_bounds[:, i] - lower_bounds[:, i]) for i in range(n_vars)}
    interval_width['overall'] = np.mean(upper_bounds - lower_bounds)
    return {'coverage': coverage, 'interval_width': interval_width}

def run_cqr_for_cluster(label, model_suffix="_main", alpha=0.1):
    params_path = os.path.join(PARAMS_DIR, f"best_params_{label}{model_suffix}.json")
    if not os.path.exists(params_path):
        print(f"No params found for cluster {label}{model_suffix}. Skipping.")
        return
    with open(params_path, "r") as f:
        params = json.load(f)
    # Load global scaler
    y_scaler = joblib.load(SCALER_PATH)
    # Data paths (match narx_data structure!)
    val_path  = os.path.join(DATA_DIR, f"cluster_{label}", "val_all.csv")
    test_path = os.path.join(DATA_DIR, f"cluster_{label}", "test_all.csv")
    val_df  = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    used_X_columns = params["used_X_columns"]
    output_columns = params["output_columns"]
    X_val = val_df.reindex(columns=used_X_columns, fill_value=0).values
    y_val = val_df[[f"{y}_target" for y in output_columns]].values
    X_test = test_df.reindex(columns=used_X_columns, fill_value=0).values
    y_test = test_df[[f"{y}_target" for y in output_columns]].values


    print(f"\n=== [CQR] {label}{model_suffix} ===")
    print("Model config:")
    print("  input_size:", X_val.shape[1])
    print("  hidden_sizes:", params["hidden_sizes"])
    print("  output_size:", len(params["output_columns"]))
    model = MLPNARX(
        input_size=X_val.shape[1],
        output_size=len(params["output_columns"]),
        hidden_sizes=params["hidden_sizes"],
        activation=params["activation"],
        dropout=params.get("dropout", 0.0),
        batchnorm=params.get("batchnorm", False),
        output_activation=None
    ).to(DEVICE)
    model_path = os.path.join(MODELS_DIR, f"model_{label}{model_suffix}.pt")
    print("Model weights from:", model_path)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    X_val_tensor = torch.from_numpy(X_val).float().to(DEVICE)
    with torch.no_grad():
        base_pred_val = model(X_val_tensor).cpu().numpy()

    results = []
    for j, out_name in enumerate(params["output_columns"]):
        print(f"  > Quantile regressors for: {out_name}")
        err_train = y_val[:, [j]] - base_pred_val[:, [j]]
        train_ds = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(err_train).float())
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        val_loader = DataLoader(train_ds, batch_size=64)
        # Lower quantile
        ql = make_quantile_net(X_val.shape[1]).to(DEVICE)
        optimizer_ql = torch.optim.Adam(ql.parameters(), lr=0.01)
        train_with_early_stopping(ql, PinballLoss(alpha / 2), optimizer_ql, train_loader, val_loader, DEVICE, n_epochs=30, patience=5, save_path=os.path.join(CQR_DIR, f"q_low_{label}_{out_name}.pth"))
        # Upper quantile
        qh = make_quantile_net(X_val.shape[1]).to(DEVICE)
        optimizer_qh = torch.optim.Adam(qh.parameters(), lr=0.01)
        train_with_early_stopping(qh, PinballLoss(1 - alpha / 2), optimizer_qh, train_loader, val_loader, DEVICE, n_epochs=30, patience=5, save_path=os.path.join(CQR_DIR, f"q_high_{label}_{out_name}.pth"))
        # Conformalize for this output
        ql, qh = conformalize_intervals(ql, qh, X_val, y_val[:, [j]], base_pred_val[:, [j]], alpha=alpha, device=DEVICE)
        # Predict on test
        X_test_tensor = torch.from_numpy(X_test).float().to(DEVICE)
        with torch.no_grad():
            pred = model(X_test_tensor).cpu().numpy()[:, j]
            low = ql(X_test_tensor).cpu().numpy().flatten()
            high = qh(X_test_tensor).cpu().numpy().flatten()
        lower = pred + low - ql.correction[0]
        upper = pred + high + qh.correction[0]
        true = y_test[:, j]
        # === INVERSE TRANSFORM all arrays! ===
        # They need to be 2D (n,1) for inverse_transform_subset
        true_inv = inverse_transform_subset(true.reshape(-1,1), [out_name], y_scaler, ALL_COLUMNS).flatten()
        pred_inv = inverse_transform_subset(pred.reshape(-1,1), [out_name], y_scaler, ALL_COLUMNS).flatten()
        lower_inv = inverse_transform_subset(lower.reshape(-1,1), [out_name], y_scaler, ALL_COLUMNS).flatten()
        upper_inv = inverse_transform_subset(upper.reshape(-1,1), [out_name], y_scaler, ALL_COLUMNS).flatten()
        results.append({
            "label": out_name,
            "true": true_inv,
            "pred": pred_inv,
            "lower": lower_inv,
            "upper": upper_inv
        })
    plot_path = os.path.join(PLOT_DIR, f"cqr_cluster_{label}{model_suffix}.png")
    plot_cqr_results(results, plot_path=plot_path)
    # Coverage evaluation
    lowers = np.stack([r["lower"] for r in results], axis=1)
    uppers = np.stack([r["upper"] for r in results], axis=1)
    yts    = np.stack([r["true"] for r in results], axis=1)
    cov_metrics = evaluate_interval_coverage(yts, lowers, uppers)
    print("\nInterval evaluation:")
    print("  Coverage (overall): {:.2f}%".format(cov_metrics['coverage']['overall']))
    for i, out_name in enumerate(params["output_columns"]):
        print("  {}: Coverage={:.2f}%, Width={:.4f}".format(
            out_name, cov_metrics['coverage'][i], cov_metrics['interval_width'][i]
        ))

def conformalize_intervals(q_low, q_high, X_cal, y_cal, base_predictions, alpha=0.1, device='cpu'):
    errors = y_cal - base_predictions
    q_low.eval()
    q_high.eval()
    X_cal_tensor = torch.FloatTensor(X_cal).to(device)
    with torch.no_grad():
        low_preds = q_low(X_cal_tensor).cpu().numpy()
        high_preds = q_high(X_cal_tensor).cpu().numpy()
    E_low = errors - low_preds
    E_high = high_preds - errors
    n_cal = len(X_cal)
    q_level = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
    correction_low = [np.quantile(E_low[:, j], q_level) for j in range(y_cal.shape[1])]
    correction_high = [np.quantile(E_high[:, j], q_level) for j in range(y_cal.shape[1])]
    q_low.correction = correction_low
    q_high.correction = correction_high
    return q_low, q_high

if __name__ == "__main__":
    for suffix in ["_main", "_dxx"]:
        for fname in sorted(os.listdir(PARAMS_DIR)):
            if fname.startswith("best_params_") and fname.endswith(f"{suffix}.json"):
                label = fname.split("_")[2]
                run_cqr_for_cluster(label, model_suffix=suffix, alpha=0.1)
