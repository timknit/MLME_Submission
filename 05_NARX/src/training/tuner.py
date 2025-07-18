"""Optuna-based hyperparameter tuning."""
import os
import optuna
import json

from src.training.trainer import train_cluster

def objective(trial, label, data_dir, model_suffix, output_columns):
    # Suggest hyperparameters
    lag = trial.suggest_int("lag", 2, 5)
    num_layers = trial.suggest_int("num_layers", 1, 4)
    hidden_sizes = [trial.suggest_int(f"hidden_{i+1}", 8, 100) for i in range(num_layers)]
    activation = trial.suggest_categorical("activation", ["relu", "gelu"])
    batch_size = trial.suggest_categorical("batch_size", [64, 128])
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-3)
    l1_lambda = trial.suggest_loguniform("l1_lambda", 1e-8, 1e-5)
    l2_lambda = trial.suggest_loguniform("l2_lambda", 1e-12, 1e-5)
    dropout = trial.suggest_float("dropout", 0.0, 0.2)
    batchnorm = False
    loss_function = trial.suggest_categorical("loss_function", ["mse", "huber"])
    output_activation = None

    # Build paths for this cluster
    cluster_dir = os.path.join(data_dir, f"cluster_{label}")
    train_path = os.path.join(cluster_dir, "train_all.csv")
    val_path = os.path.join(cluster_dir, "val_all.csv")
    test_path = os.path.join(cluster_dir, "test_all.csv")

    val_loss = train_cluster(
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        label=label,
        lag=lag,
        hidden_sizes=hidden_sizes,
        loss_function=loss_function,
        activation=activation,
        epochs=150,
        batch_size=batch_size,
        lr=lr,
        l1_lambda=l1_lambda,
        l2_lambda=l2_lambda,
        dropout=dropout,
        batchnorm=batchnorm,
        output_activation=output_activation,
        patience=15,
        verbose=False,
        output_columns=output_columns,
        model_suffix=model_suffix,
        return_val_loss=True,
    )
    # In case something fails in train_cluster, return a high loss so Optuna discards it
    if val_loss is None:
        return 1e10
    return val_loss

def save_best_params(study, label, model_suffix, params_dir):
    """Save best params from Optuna study as JSON."""
    os.makedirs(params_dir, exist_ok=True)
    params = study.best_trial.params
    with open(os.path.join(params_dir, f"best_params_{label}{model_suffix}_optuna.json"), "w") as f:
        import json
        json.dump(params, f, indent=2)
