# src/config.py
import os
INPUT_COLUMNS = ['mf_PM', 'mf_TM', 'Q_g', 'w_crystal', 'c_in', 'T_PM_in', 'T_TM_in']
MAIN_COLUMNS = ['T_PM', 'T_TM', 'c']
DXX_COLUMNS = ['d10', 'd50', 'd90']
X_INPUTS_NAMES = ['mf_PM', 'mf_TM', 'Q_g', 'w_crystal', 'c_in', 'T_PM_in', 'T_TM_in']
Y_OUTPUTS_NAMES = ['T_PM', 'T_TM', 'c', 'd10', 'd50', 'd90']
ALL_COLUMNS = ['c', 'T_PM', 'd50', 'd90', 'd10', 'T_TM', 'mf_PM', 'mf_TM', 'Q_g', 'w_crystal', 'c_in', 'T_PM_in', 'T_TM_in']


# Pfade
Config_dir = os.path.dirname(os.path.abspath(__file__))
SCRIPT_DIR = os.path.abspath(os.path.join(Config_dir, ".."))
#print(f"Script directory: {SCRIPT_DIR}")
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "04_data_preprocessing","narx_data")
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
PARAMS_DIR = os.path.join(SCRIPT_DIR, "best_params")
PLOT_DIR = os.path.join(SCRIPT_DIR, "plots")
CQR_DIR = os.path.join(MODELS_DIR, "cqr")
SCALER_PATH = os.path.join(PROJECT_ROOT,"data_preprocessing", "scaler", "scaler_df_all.joblib")
