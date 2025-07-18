import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from src.config import ALL_COLUMNS  # <-- import ALL_COLUMNS
import pandas as pd
def inverse_transform_subset(y_scaled, output_columns, y_scaler, all_columns):
    n_samples = y_scaled.shape[0]
    dummy = np.zeros((n_samples, len(all_columns)))
    df_dummy = pd.DataFrame(dummy, columns=all_columns)
    for i, col in enumerate(output_columns):
        df_dummy[col] = y_scaled[:, i]
    y_full_inv = y_scaler.inverse_transform(df_dummy.values)
    df_full_inv = pd.DataFrame(y_full_inv, columns=all_columns)
    return df_full_inv[output_columns].values

def plot_model_predictions(model, X_test_tensor, y_test_tensor, y_scaler, y_outputs_names, cluster_label, outdir=None):
    """
    Plot true vs. predicted values for all outputs in a time series style.
    """
    model.eval()
    with torch.no_grad():
        y_pred_tensor = model(X_test_tensor)
    y_pred_scaled = y_pred_tensor.cpu().numpy()
    y_test_scaled = y_test_tensor.cpu().numpy()
    y_pred = inverse_transform_subset(y_pred_scaled, y_outputs_names, y_scaler, ALL_COLUMNS)
    y_true = inverse_transform_subset(y_test_scaled, y_outputs_names, y_scaler, ALL_COLUMNS)

    fig, axes = plt.subplots(y_pred.shape[1], 1, figsize=(14, 8), sharex=True)
    if y_pred.shape[1] == 1:
        axes = [axes]

    # Store the line handles for the first axis only
    handles, labels = None, None
    for i in range(y_pred.shape[1]):
        ax = axes[i]
        line_true, = ax.plot(y_true[:, i], label="True", linewidth=1)
        line_pred, = ax.plot(y_pred[:, i], label="Predicted", linestyle='--', linewidth=1)
        ax.set_ylabel(y_outputs_names[i])
        ax.grid(True)
        if i == 0:
            ax.set_title("Model Prediction vs. True (Test Data)")
            # Only grab the legend handles/labels from the first subplot
            handles, labels = ax.get_legend_handles_labels()
        if i == y_pred.shape[1] - 1:
            ax.set_xlabel("Index")

    # Only add ONE legend, outside the plot
    if handles and labels:
        fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    plt.tight_layout(rect=[0, 0, 0.95, 1])
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(os.path.join(outdir, f"Prediction_cluster_{cluster_label}.png"))
    plt.close()



def parity_plot(y_true, y_pred, out_names, cluster_label, outdir=None):
    """
    Parity (true vs. predicted scatter) plot for each output, with means highlighted.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    n_outputs = y_true.shape[1]
    fig, axs = plt.subplots(1, n_outputs, figsize=(5 * n_outputs, 5))
    if n_outputs == 1:
        axs = [axs]
    for i in range(n_outputs):
        axs[i].scatter(y_true[:, i], y_pred[:, i], alpha=0.5, label='Test Data')
        minmax = [y_true[:, i].min(), y_true[:, i].max()]
        axs[i].plot(minmax, minmax, 'r--', label='Perfect prediction')
        # Highlight mean
        #mean_true = np.mean(y_true[:, i])
        #mean_pred = np.mean(y_pred[:, i])
        #axs[i].scatter([mean_true], [mean_pred], color="orange", s=120, marker="x", label="Mean (true/pred)")
        axs[i].set_xlabel(f"True {out_names[i]}")
        axs[i].set_ylabel(f"Predicted {out_names[i]}")
        axs[i].set_title(f"Parity plot: {out_names[i]}")
        axs[i].legend()
    plt.suptitle(f"Cluster {cluster_label} Parity Plots")
    plt.tight_layout()
    if outdir:
        import os
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(os.path.join(outdir, f"parity_cluster_{cluster_label}.png"))
    plt.close()

def plot_cqr_results(results, title="CQR Prediction Intervals", color="deepskyblue", save_path=None):
    """
    Plot CQR prediction intervals, true values, and predictions for each output variable.
    results: list of dicts, one per output. Each dict: {'label', 'true', 'pred', 'lower', 'upper'}
    """
    n = len(results)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n), sharex=True)
    axes = np.atleast_1d(axes)
    for res in results:
        print(f"{res['label']}: interval mean width = {(res['upper'] - res['lower']).mean():.4f}")

    for i, res in enumerate(results):
        ax = axes[i]
        ax.plot(res["true"], label="True", linewidth=1)
        ax.plot(res["pred"], label="Prediction", linestyle='--', linewidth=1)
        ax.plot(res["lower"], "--", color="green", linewidth=0.7, alpha=0.7, label="Lower Bound")
        ax.plot(res["upper"], "--", color="red", linewidth=0.7, alpha=0.7, label="Upper Bound")
        ax.fill_between(
            np.arange(len(res["pred"])),
            res["lower"],
            res["upper"],
            color=color,
            alpha=0.5,
            label="CQR Interval"
        )
        ax.set_ylabel(res["label"], fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc="upper right", fontsize=10)
        if i == 0:
            ax.set_title(title, fontsize=14)
        if i == n - 1:
            ax.set_xlabel("Index", fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 1])
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Plot saved to: {save_path}")
    plt.show()
