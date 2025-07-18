import os
import re
import pandas as pd
import numpy as np
import shutil

"""
class DATA_LAGGEN:
    def __init__(self, data_path, seed=42):
        self.data_path = data_path
        self.X_Inputs_names = ['mf_PM', 'mf_TM', 'Q_g', 'w_crystal', 'c_in', 'T_PM_in', 'T_TM_in']
        self.y_outputs_names = ['T_PM', 'T_TM', 'c', 'd10', 'd50', 'd90']
        self.files_per_cluster = None
        self.seed = seed


    def create_files_per_cluster(self):
        files_per_cluster = {}
        for file in os.listdir(self.data_path):
            if file.endswith(".csv"):
                match = re.search(r"_cluster(\d+)\.csv", file)
                if match:
                    cluster_id = int(match.group(1))
                    if cluster_id not in files_per_cluster:
                        files_per_cluster[cluster_id] = []
                    files_per_cluster[cluster_id].append(os.path.join(self.data_path, file))
        self.files_per_cluster = files_per_cluster
        return files_per_cluster


    def create_narx_lagged_df(self, df, lag=10):
        X = df[self.X_Inputs_names].values
        y = df[self.y_outputs_names].values
        X_narx, y_narx = [], []
        for t in range(lag, len(X)):
            x_lagged = X[t-lag:t].flatten()
            y_lagged = y[t-lag:t-1].flatten()
            x_full = np.concatenate([x_lagged, y_lagged])
            X_narx.append(x_full)
            y_narx.append(y[t])
        X_narx = np.array(X_narx)
        y_narx = np.array(y_narx)
        if y_narx.ndim == 2 and y_narx.shape[1] == 1:
            y_narx = y_narx.flatten()
        col_names = []
        for l in range(lag):
            for name in self.X_Inputs_names:
                col_names.append(f"{name}_lag{lag-l}")
        for l in range(lag-1):
            for name in self.y_outputs_names:
                col_names.append(f"{name}_lag{lag-1-l}")
        for name in self.y_outputs_names:
            col_names.append(f"{name}_target")
        data_combined = np.concatenate([X_narx, y_narx.reshape(-1, 1) if y_narx.ndim==1 else y_narx], axis=1)
        df_lagged = pd.DataFrame(data_combined, columns=col_names)
        return df_lagged


    def split_files_by_trajectory(self, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
        if self.files_per_cluster is None:
            raise ValueError("files_per_cluster ist noch nicht gesetzt. Bitte zuerst create_files_per_cluster() aufrufen.")
        random.seed(self.seed)
        split_dict = {}
        for cluster_id, file_list in self.files_per_cluster.items():
            file_list = file_list[:]  # copy
            random.shuffle(file_list)
            n = len(file_list)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)
            split_dict[cluster_id] = {
                "train": file_list[:n_train],
                "val": file_list[n_train:n_train+n_val],
                "test": file_list[n_train+n_val:]
            }
        return split_dict


    def process_all_files_by_trajectory(self, lag=10, out_base_dir="narx_data"):
        split_dict = self.split_files_by_trajectory()
        for cluster_id, split in split_dict.items():
            big_dfs = {"train": [], "val": [], "test": []}
            for split_name, file_list in split.items():
                print("------------------------------------------------------------------------------------------------------")
                for file_path in file_list:
                    df = pd.read_csv(file_path)
                    df_lagged = self.create_narx_lagged_df(df, lag=lag)
                    big_dfs[split_name].append(df_lagged)
                    print(f"{os.path.basename(file_path)}: Gelaggt und zu {split_name} für Cluster {cluster_id} hinzugefügt ({len(df_lagged)} Zeilen).")
            # Nach Durchlauf: ein großes DataFrame pro Split speichern
            print("\n______________________________________________________________________________________________________")
            for split_name in ["train", "val", "test"]:
                if big_dfs[split_name]:  # nur wenn Daten vorhanden
                    outdir = os.path.join(out_base_dir, f"cluster_{cluster_id}")
                    os.makedirs(outdir, exist_ok=True)
                    big_df = pd.concat(big_dfs[split_name], ignore_index=True)
                    merged_path = os.path.join(outdir, f"{split_name}_all.csv")
                    big_df.to_csv(merged_path, index=False)
                    print(f"-> Cluster {cluster_id}: {split_name}_all.csv gespeichert ({len(big_df)} Zeilen)")


    def cleanup_folders(self):
        print("\nBeginne nicht benötigte Ordner zu löschen")
        # Zu löschende Ordner
        folders_to_delete = [
            'cluster_csv',
            'cluster_data',
            'cluster_predictions',
        ]

        for folder in folders_to_delete:
            if os.path.isdir(folder):
                print(f"Lösche Ordner: {folder}")
                shutil.rmtree(folder)
        print("Nicht benötigte Ordner löschen abegschlossen")
"""



class DATA_LAGGEN:
    def __init__(self, data_path, seed=42):
        self.data_path = data_path
        self.X_Inputs_names = ['mf_PM', 'mf_TM', 'Q_g', 'w_crystal', 'c_in', 'T_PM_in', 'T_TM_in']
        self.y_outputs_names = ['T_PM', 'T_TM', 'c', 'd10', 'd50', 'd90']
        self.files_per_cluster = None
        self.seed = seed

    def create_files_per_cluster(self):
        files_per_cluster = {}
        for file in os.listdir(self.data_path):
            if file.endswith(".csv"):
                match = re.search(r"_cluster(\d+)\.csv", file)
                if match:
                    cluster_id = int(match.group(1))
                    if cluster_id not in files_per_cluster:
                        files_per_cluster[cluster_id] = []
                    files_per_cluster[cluster_id].append(os.path.join(self.data_path, file))
        self.files_per_cluster = files_per_cluster
        return files_per_cluster

    def create_narx_lagged_df(self, df, lag=10):
        X = df[self.X_Inputs_names].values
        y = df[self.y_outputs_names].values
        X_narx, y_narx = [], []
        for t in range(lag, len(X)):
            x_lagged = X[t-lag:t].flatten()
            y_lagged = y[t-lag:t-1].flatten()
            x_full = np.concatenate([x_lagged, y_lagged])
            X_narx.append(x_full)
            y_narx.append(y[t])
        X_narx = np.array(X_narx)
        y_narx = np.array(y_narx)
        if y_narx.ndim == 2 and y_narx.shape[1] == 1:
            y_narx = y_narx.flatten()
        col_names = []
        for l in range(lag):
            for name in self.X_Inputs_names:
                col_names.append(f"{name}_lag{lag-l}")
        for l in range(lag-1):
            for name in self.y_outputs_names:
                col_names.append(f"{name}_lag{lag-1-l}")
        for name in self.y_outputs_names:
            col_names.append(f"{name}_target")
        data_combined = np.concatenate([X_narx, y_narx.reshape(-1, 1) if y_narx.ndim==1 else y_narx], axis=1)
        df_lagged = pd.DataFrame(data_combined, columns=col_names)
        return df_lagged

    def process_all_files_by_trajectory(self, lag=10, out_base_dir="narx_data"):
            self.create_files_per_cluster()
            for cluster_id, file_list in self.files_per_cluster.items():
                big_dfs = []
                print(f"Cluster {cluster_id}:")
                for file_path in file_list:
                    df = pd.read_csv(file_path)
                    df_lagged = self.create_narx_lagged_df(df, lag=lag)
                    big_dfs.append(df_lagged)
                    print(f"  {os.path.basename(file_path)}: Gelaggt und hinzugefügt ({len(df_lagged)} Zeilen)")
                # Zusammenführen und speichern
                outdir = os.path.join(out_base_dir, f"cluster_{cluster_id}")
                os.makedirs(outdir, exist_ok=True)
                big_df = pd.concat(big_dfs, ignore_index=True)
                merged_path = os.path.join(outdir, "eval_all.csv")
                big_df.to_csv(merged_path, index=False)
                print(f"-> Cluster {cluster_id}: eval_all.csv gespeichert ({len(big_df)} Zeilen)")

    def cleanup_folders(self):
        print("\nBeginne nicht benötigte Ordner zu löschen")
        folders_to_delete = [
            'cluster_csv',
            'cluster_predictions',
        ]
        for folder in folders_to_delete:
            if os.path.isdir(folder):
                print(f"Lösche Ordner: {folder}")
                shutil.rmtree(folder)
        print("Nicht benötigte Ordner löschen abgeschlossen")




if __name__ == "__main__":
    data_lagger = DATA_LAGGEN(data_path="common_points")
    data_lagger.create_files_per_cluster()
    data_lagger.process_all_files_by_trajectory(lag=5, out_base_dir="narx_data")
    data_lagger.cleanup_folders()
