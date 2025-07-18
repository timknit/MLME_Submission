import os
# from data_preprocessing_BTF import DATA_PREPROCESSING
from data_preprocessing_BTF import DATA_PREPROCESSING
from data_laggen_BTF import DATA_LAGGEN
from eval import evaluate_from_config
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PARAMS_DIR = os.path.join(PROJECT_ROOT, "NARX_Models", "params")


if __name__ == "__main__":
    data_flash = DATA_PREPROCESSING(dir_name='beat-the-felix-data')   # Initialisiert die Datenpipeline
    data_flash.rough_filter(output_folder='data_trash', threshold=0.001)   # 1. Grobes Filtern der Dateien (entfernt Dateien mit zu hohen d50-Werten oder falschem Format)
    data_flash.load_all_rough_data()                                      # 2. Alle gefilterten Dateien laden und zu einem großen DataFrame zusammenfassen
    data_flash.normalize_rough_dataframe(scaler_path='scaler/scaler_df_all.joblib')   # 3. Daten skalieren (Standardisierung), Scaler speichern
    data_flash.process_files_in_dir()                                     # 8. Anwenden des Cluster-Modells auf alle Einzeldateien (Klassifizierung)
    data_flash.group_data_by_cluster(folder_path="cluster_predictions")   # 9. Gruppieren aller Datenpunkte nach Clusterzugehörigkeit
    # data_flash.remove_outliers_inplace(percentile=95)                     # 10. Entfernen von Ausreißern innerhalb der Cluster (Mahalanobis)
    data_flash.remove_outliers_inplace(which="inputs", percentile=97) # 10. Entfernen von Ausreißern innerhalb der Cluster (Mahalanobis)
    data_flash.remove_outliers_inplace(which="outputs", percentile=99) # 10. Entfernen von Ausreißern innerhalb der Cluster (Mahalanobis)
    data_flash.save_clusters_to_csv(save_dir="cluster_csv")               # 13. Speichern der bereinigten Cluster als separate CSV-Dateien
    data_flash.filter_predictions_with_clusters()                         # 14. Vergleich/Schnittmenge: Predictions vs. bereinigte Cluster, Speichern der gemeinsamen Punkte

    # Data laggen
    data_lagger = DATA_LAGGEN(data_path="common_points")
    data_lagger.create_files_per_cluster()
    data_lagger.process_all_files_by_trajectory(lag=5, out_base_dir="narx_data")
    data_lagger.cleanup_folders()


    # Model evaluation
    for fname in sorted(os.listdir(PARAMS_DIR)):
        if fname.startswith("best_params_") and fname.endswith(".json"):
            params_path = os.path.join(PARAMS_DIR, fname)
            evaluate_from_config(params_path)



    








