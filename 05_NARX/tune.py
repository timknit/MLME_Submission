import os
import optuna
import warnings
from src.config import MAIN_COLUMNS, DXX_COLUMNS, SCRIPT_DIR, DATA_DIR, PARAMS_DIR
from src.training.tuner import objective, save_best_params  
warnings.filterwarnings("ignore", category=FutureWarning)

#cluster_labels = [name.replace("cluster_", "") for name in os.listdir(DATA_DIR) if name.startswith("cluster_")]
cluster_labels = ["1", "0"]
N_TRIALS = 100

for label in cluster_labels:
    print(f"---------- Tuning cluster {label} (main) --------------")
    study_main = optuna.create_study(direction="minimize")
    study_main.optimize(
        lambda trial: objective(trial, label, DATA_DIR, "_main", MAIN_COLUMNS),
        n_trials=N_TRIALS,
        show_progress_bar=True,
        n_jobs=-1
    )
    save_best_params(study_main, label, "_main", PARAMS_DIR)

    print(f"---------- Tuning cluster {label} (dxx) --------------")
    study_dxx = optuna.create_study(direction="minimize")
    study_dxx.optimize(
        lambda trial: objective(trial, label, DATA_DIR, "_dxx", DXX_COLUMNS),
        n_trials=N_TRIALS,
        show_progress_bar=True,
        n_jobs=-1
    )
    save_best_params(study_dxx, label, "_dxx", PARAMS_DIR)

print("Tuning complete.")
