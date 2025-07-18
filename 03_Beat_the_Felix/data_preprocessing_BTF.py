import os
import shutil
import warnings
from collections import Counter

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from scipy.spatial.distance import mahalanobis

import hdbscan
from hdbscan import approximate_predict

import joblib

warnings.simplefilter(action='ignore', category=FutureWarning)






class DATA_PREPROCESSING:
    def __init__(self, dir_name):
        self.dir_name = dir_name
        self.dfs = pd.DataFrame()
        self.dfs_scaled = pd.DataFrame()
        self.cluster_labels = []
        self.X_PCA = []
        self.df_result_cluster = pd.DataFrame()
        self.cluster_data = {}
        self.X_Inputs_names = ['mf_PM', 'mf_TM', 'Q_g', 'w_crystal', 'c_in', 'T_PM_in', 'T_TM_in']
        self.y_outputs_names = ['T_PM', 'T_TM', 'c', 'd10', 'd50', 'd90']


    def rough_filter(self, output_folder, threshold=1.0): # get rid of 200-files
        print("Beginne grobes Filtern")
        os.makedirs(output_folder, exist_ok=True)
        for file in os.listdir(self.dir_name):
            file_name = os.path.join(self.dir_name, file)
            if not file_name.endswith('.txt'):
                shutil.move(file_name, os.path.join(output_folder, file))
                continue

            df = pd.read_csv(file_name, sep='\t')  # Assuming tab-separated values
            if 'd50' in df.columns:
                    d50_value = df['d50'].mean()  # Calculate mean d50 value
                    if d50_value > threshold:
                        print(f"Moving {file} with d50 = {d50_value:.2} to {output_folder}.")
                        shutil.move(file_name, os.path.join(output_folder, file))
            else:
                    print(f"Column 'd50' not found in {file}. Skipping this file.")

        print("Grobes Filtern abgeschlossen\n")


    def load_all_rough_data(self):
        print("Beginne alle Daten zu laden")
        dfs = [pd.read_csv(os.path.join(self.dir_name, f), sep='\t') for f in os.listdir(self.dir_name)]
        self.dfs = pd.concat(dfs, ignore_index=True)

        print("Alle Daten laden abgeschlossen\n")
    

    def normalize_rough_dataframe(self, scaler_path='scaler/scaler_df_all.joblib'):
        print("Beginne alle Daten zu skalieren")
        scaler = StandardScaler()
        scaled = scaler.fit_transform(self.dfs)
        scaled_df = pd.DataFrame(scaled, columns=self.dfs.columns, index=self.dfs.index)

        os.makedirs(os.path.dirname(scaler_path), exist_ok=True) # Sicherstellen, dass der Zielordner existiert
        joblib.dump(scaler, scaler_path)     # Scaler speichern mit joblib

        self.dfs_scaled = scaled_df
        print("Skalieren der Daten abgeschlossen\n")

    def remove_outliers_zscore(self, threshold=3): ##### NEU hinzugefügt
        print("Beginne Ausreißer mit Z-Score-Filter zu entfernen")
        # DataFrame: self.dfs_scaled
        z_scores = np.abs(self.dfs_scaled)
        # Für jede Zeile prüfen: gibt es eine Spalte mit zscore > threshold?
        mask = (z_scores < threshold).all(axis=1)
        num_outliers = (~mask).sum()
        print(f"{num_outliers} Ausreißer-Zeilen werden entfernt")
        self.dfs_scaled = self.dfs_scaled[mask].reset_index(drop=True)
        print("Ausreißer-Filter abgeschlossen\n")

    

    def hdbscan_clustering(self, cluster_path="cluster_data/cluster_df_all.joblib", min_cluster_size=500):
        print("Beginne Clustering")
        os.makedirs(os.path.dirname(cluster_path), exist_ok=True)  

        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, prediction_data=True)
        labels = clusterer.fit_predict(self.dfs_scaled.values)
        joblib.dump(clusterer, cluster_path)

        self.cluster_labels = labels
        print("Clustering abegschlossen\n")


    def compute_pca_2d(self):
        print("Beginne PCA-Berechnungen")
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(self.dfs_scaled.values)
        self.X_PCA = X_pca
        print("PCA-Berechnungen abgeschlossen")

    

    def plot_hdbscan(self, do_plot=True):
        print("Beginne Cluster-Plotting")
        if not do_plot:
            return
        
        plt.figure(figsize=(10, 7))
        
        unique_labels = sorted(set(self.cluster_labels))
        colors = cm.get_cmap('tab20', len(unique_labels))

        counts = Counter(self.cluster_labels)
        
        for idx, label in enumerate(unique_labels):
            mask = self.cluster_labels == label
            if label == -1:
                color = 'k'
                name = f'Noise ({np.sum(mask)} Punkte)'
            else:
                color = colors(idx)
                name = f'Cluster {label} ({np.sum(mask)} Punkte)'
            
            plt.scatter(
                self.X_PCA[mask, 0], self.X_PCA[mask, 1],
                c=[color],
                label=name,
                s=10,
                alpha=0.7
            )
        plt.title("HDBSCAN Clustering (PCA auf 2D)", fontsize=14)
        plt.xlabel("PCA-Komponente 1")
        plt.ylabel("PCA-Komponente 2")
        plt.legend(loc="best", fontsize=9)
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        print("Cluster-Plotting abgeschlossen")


    def cluster_summary(self):
        print("Beginne Clusterübersicht zu erstellen")
        n_clusters = len(set(self.cluster_labels)) - (1 if -1 in self.cluster_labels else 0)
        n_noise = list(self.cluster_labels).count(-1)
        print(f"Anzahl Cluster (ohne Noise): {n_clusters}")
        print(f"Anzahl Noise-Punkte: {n_noise}")
        print("Clusterübersicht erstellen abgeschlossen")


    def classify_datafiles_general(self, file_path, scaler_path="scaler/scaler_df_all.joblib", clusterer_path="cluster_data/cluster_df_all.joblib", save_output=True, output_path=None):
        # Modelle laden
        scaler = joblib.load(scaler_path)
        clusterer = joblib.load(clusterer_path)
        
        df = pd.read_csv(file_path, sep='\t') # Daten laden
        data_scaled = scaler.transform(df) # Skalieren

        # Vorhersage
        labels, strengths = approximate_predict(clusterer, data_scaled)
        # print(f"Anzahl der Cluster: {len(set(labels)) - (1 if -1 in labels else 0)}")
        print(f"\tClusterlabels: {set(labels)}")

        # Ergebnis-DataFrame erstellen
        df_result = pd.DataFrame(data_scaled, columns=df.columns)
        df_result["predicted_cluster"] = labels
        df_result["cluster_strength"] = strengths

        # Speichern nur, wenn ausschließlich Cluster 0 und/oder 1 (evtl. mit -1) vorkommen
        valid_clusters = set(labels) - {-1}  # Ignoriere Noise
        if valid_clusters.issubset({0, 1}):
            if save_output:
                if output_path is None:
                    base_name = os.path.basename(file_path).replace(".txt", "_clustered.txt")
                    output_path = os.path.join("cluster_predictions", base_name)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                df_result.to_csv(output_path, sep='\t', index=False)
                # print(f"Ergebnisse gespeichert in '{output_path}'")
        else:
            print(f"Datei '{file_path}' enthält andere Cluster ({valid_clusters}) - nicht gespeichert.")

        self.df_result_cluster = df_result


    def process_files_in_dir(self):
        print("Beginne Files in Cluster einzuordnen")
        counter = 1
        len_dir = len(os.listdir(self.dir_name))
        for filename in os.listdir(self.dir_name):
            if filename.endswith(".txt"):
                file_path = os.path.join(self.dir_name, filename)
                try:
                    print(f"Cluster: {counter:02} von {len_dir}")
                    self.classify_datafiles_general(file_path)
                except Exception as e:
                    print(f"Fehler bei Datei '{file_path}': {e}")
            counter += 1
        print("Files in Cluster einzuordnen abgeschlossen")


    def group_data_by_cluster(self, folder_path="cluster_predictions"):
        print("Beginne die Cluster zu gruppieren")
        cluster_data = {}

        # Alle .txt-Dateien im Ordner durchgehen
        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(folder_path, filename)
                df = pd.read_csv(file_path, sep='\t')

                # Sicherstellen, dass 'predicted_cluster' vorhanden ist
                if 'predicted_cluster' not in df.columns:
                    print(f"⚠️  Datei '{filename}' enthält keine 'predicted_cluster'-Spalte. Überspringe.")
                    continue

                # Nach Clustern gruppieren und sammeln
                for cluster_id, group in df.groupby('predicted_cluster'):
                    cluster_id = int(cluster_id)  # sicherstellen, dass es int ist
                    if cluster_id not in cluster_data:
                        cluster_data[cluster_id] = group.copy()
                    else:
                        cluster_data[cluster_id] = pd.concat([cluster_data[cluster_id], group.copy()], ignore_index=True)

        self.cluster_data = cluster_data
        print("Die Cluster zu gruppieren abgeschlossen")


    def filter_outliers_mahalanobis(self, X, percentile=97):
        mean = np.mean(X, axis=0)
        cov = np.cov(X, rowvar=False)
        inv_covmat = np.linalg.pinv(cov)

        distances = np.array([mahalanobis(x, mean, inv_covmat) for x in X])
        threshold = np.percentile(distances, percentile)
        mask = distances <= threshold
        return X[mask], mask


    def remove_outliers_inplace(self, which="outputs", percentile=97):
        print("Beginne die Outlier herauszufiltern")
        for cluster_id, df_cluster in self.cluster_data.items():
            if which == "outputs":
                X = df_cluster[self.y_outputs_names].values
            elif which == "inputs":
                X = df_cluster[self.X_Inputs_names].values
            else:
                raise ValueError("which muss 'inputs' oder 'outputs' sein.")
            
            _, mask = self.filter_outliers_mahalanobis(X, percentile=percentile)
            self.cluster_data[cluster_id] = df_cluster[mask].reset_index(drop=True)
        print("Die Outlier herauszufiltern abgeschlossen")

    # def save_cluster_scalers(self, save_dir="scaler"):
    #     print("Beginne die Scaler pro Cluster [Input, Output] zu erstellen")
    #     os.makedirs(save_dir, exist_ok=True)
    #     for cluster_id, df_cluster in self.cluster_data.items():
    #         if cluster_id == -1:
    #             continue  # Noise-Cluster überspringen!

    #         #################################### HIER ZWISCHEN VERÄNDERT ####################################

    #         # df_cluster = df_cluster[self.X_Inputs_names + self.y_outputs_names] # Spaltennamen predicted_cluster, cluster_strength ignorieren
    #         # # df-Zurückskalieren
    #         # scaler_df_all = joblib.load("scaler/scaler_df_all.joblib")
    #         # df_cluster = scaler_df_all.inverse_transform(df_cluster)

    #         #################################### HIER ZWISCHEN VERÄNDERT ####################################
    #         # Input-Scaler
    #         scaler_inputs = StandardScaler()
    #         scaler_inputs.fit(df_cluster[self.X_Inputs_names])
    #         joblib.dump(scaler_inputs, os.path.join(save_dir, f"scaler_inputs_cluster{cluster_id}.joblib"))

    #         # Output-Scaler
    #         scaler_outputs = StandardScaler()
    #         scaler_outputs.fit(df_cluster[self.y_outputs_names])
    #         joblib.dump(scaler_outputs, os.path.join(save_dir, f"scaler_outputs_cluster{cluster_id}.joblib"))
    #         print(f"Scaler für Cluster {cluster_id} gespeichert.")

    #     print("Die Scaler pro Cluster [Input, Output] zu erstellen abgeschlossen")


    def save_cluster_scalers(self, save_dir="scaler"):
        print("Beginne die Scaler pro Cluster [Input, Output] zu erstellen")
        os.makedirs(save_dir, exist_ok=True)
        for cluster_id, df_cluster in self.cluster_data.items():
            if cluster_id == -1:
                continue  # Noise-Cluster überspringen!
            
            # Spaltenreihenfolge merken
            cols = df_cluster.columns.tolist()
            
            # Input skalieren
            scaler_inputs = StandardScaler()
            scaler_inputs.fit(df_cluster[self.X_Inputs_names])
            joblib.dump(scaler_inputs, os.path.join(save_dir, f"scaler_X_Inputs_cluster{cluster_id}.joblib"))
            X_scaled = scaler_inputs.transform(df_cluster[self.X_Inputs_names])

            # Outputs skalieren
            scaler_outputs = StandardScaler()
            scaler_outputs.fit(df_cluster[self.y_outputs_names])
            joblib.dump(scaler_outputs, os.path.join(save_dir, f"scaler_y_outputs_cluster{cluster_id}.joblib"))
            y_scaled = scaler_outputs.transform(df_cluster[self.y_outputs_names])
            
            # Neu zusammensetzen – ACHTUNG: Index bleibt erhalten!
            X_scaled_df = pd.DataFrame(X_scaled, columns=self.X_Inputs_names, index=df_cluster.index)
            y_scaled_df = pd.DataFrame(y_scaled, columns=self.y_outputs_names, index=df_cluster.index)
            
            # Alle anderen Spalten unverändert übernehmen (falls vorhanden)
            other_cols = [col for col in df_cluster.columns if col not in self.X_Inputs_names + self.y_outputs_names]
            others_df = df_cluster[other_cols] if other_cols else None
            
            # DataFrame zusammensetzen, Reihenfolge wie vorher
            frames = []
            for col in cols:
                if col in self.X_Inputs_names:
                    frames.append(X_scaled_df[[col]])
                elif col in self.y_outputs_names:
                    frames.append(y_scaled_df[[col]])
                else:
                    if others_df is not None:
                        frames.append(others_df[[col]])
            df_scaled = pd.concat(frames, axis=1)
            
            # Überschreiben
            self.cluster_data[cluster_id] = df_scaled



    def plot_clusters_2d(self, X, labels, title="Cluster-Visualisierung (nach Größe)", max_clusters=20, max_legend_items=10):
        print("Beginne die Cluster mit PCA zu plotten")
        # PCA-Reduktion
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        df_plot = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
        df_plot["cluster"] = labels

        # Cluster nach Größe sortieren (ohne Rauschen)
        cluster_sizes = df_plot[df_plot["cluster"] != -1]["cluster"].value_counts().sort_values(ascending=False)
        cluster_order = list(cluster_sizes.index) + ([-1] if -1 in labels else [])

        # Farben zuweisen
        cmap = plt.get_cmap("tab20")
        color_map = {
            cl: cmap(i % 20) if i < max_clusters else (0.6, 0.6, 0.6)
            for i, cl in enumerate(cluster_order)
        }

        plt.figure(figsize=(10, 6))

        for i, cl in enumerate(cluster_order):
            mask = df_plot["cluster"] == cl
            color = "k" if cl == -1 else color_map[cl]
            label = None

            # Nur größte N Cluster in Legende
            if cl == -1:
                label = f"Rauschen ({mask.sum()})"
            elif i < max_legend_items:
                label = f"Cluster {cl} ({mask.sum()})"

            plt.scatter(df_plot.loc[mask, "PC1"], df_plot.loc[mask, "PC2"],
                        s=10, c=[color], label=label, alpha=0.6, edgecolors='none')

        plt.title(title)
        plt.xlabel("PCA Komponente 1")
        plt.ylabel("PCA Komponente 2")
        plt.legend(markerscale=2, loc="best", bbox_to_anchor=(1.05, 1), borderaxespad=0.)
        plt.tight_layout()
        # plt.savefig("cluster_visualization.svg", format='svg')
        plt.show()
        print("Die Cluster mit PCA zu plotten abgeschlossen")


    def plot_clusters_after_outlier_removal(self, do_plot=True):
        print("Beginne die Cluster ohne outlier zu plotten")
        if not do_plot:
            return
        
        X_all = []
        labels_all = []

        for cluster_id, df in self.cluster_data.items():
            X_all.append(df[self.X_Inputs_names].values)
            labels_all.append(np.full(len(df), cluster_id))

        X_all = np.vstack(X_all)
        labels_all = np.concatenate(labels_all)

        self.plot_clusters_2d(X=X_all, labels=labels_all)
        print("Die Cluster ohne outlier zu plotten abgeschlossen")


    def save_clusters_to_csv(self, save_dir="cluster_csv"):
        print("Beginne die Cluster als .csv-Dateien abzuspeichern")
        os.makedirs(save_dir, exist_ok=True)
        for cluster_id, df_cluster in self.cluster_data.items():
            df_to_save = df_cluster[self.X_Inputs_names + self.y_outputs_names]
            filename = f"clusterID_{cluster_id}.csv"
            df_to_save.to_csv(os.path.join(save_dir, filename), index=False)
            print(f"Cluster {cluster_id}: gespeichert als {filename}")
        print("Die Cluster als .csv-Dateien abzuspeichern abgeschlossen")


    def filter_predictions_with_clusters(self, pred_dir="cluster_predictions", cluster_dir="cluster_csv", save_dir="common_points"):
        print("Beginne Vergleich der Originalen Cluster und Outlier-freien Cluster und speicher die Schnittmenge")
        os.makedirs(save_dir, exist_ok=True)
        
        # Cluster-DataFrames laden (einmalig)
        cluster_dfs = {}
        for cid in [0, 1]:
            cluster_file = os.path.join(cluster_dir, f"clusterID_{cid}.csv")
            if os.path.exists(cluster_file):
                cluster_dfs[cid] = pd.read_csv(cluster_file)
            else:
                print(f"Clusterdatei nicht gefunden: {cluster_file}")

        # Gehe alle .txt durch
        for filename in os.listdir(pred_dir):
            if filename.endswith(".txt"):
                pred_path = os.path.join(pred_dir, filename)
                df_pred = pd.read_csv(pred_path, sep=None, engine="python")  # auto-separator
                
                # Lösche cluster_prediction und cluster_strength (falls vorhanden)
                for col in ["cluster_prediction", "cluster_strength"]:
                    if col in df_pred.columns:
                        df_pred = df_pred.drop(columns=[col])
                
                for cid, df_cluster in cluster_dfs.items():
                    # Merge/Schnittmenge der Zeilen (alle Spalten vergleichen)
                    df_common = pd.merge(df_pred, df_cluster, how="inner")
                    
                    # Nur speichern, wenn was gefunden wurde
                    if not df_common.empty:
                        save_name = f"{os.path.splitext(filename)[0]}_common_cluster{cid}.csv"
                        save_path = os.path.join(save_dir, save_name)
                        df_common.to_csv(save_path, index=False)
                        print(f"{save_name}: {len(df_common)} Zeilen gespeichert.")
        print("Vergleich der Originalen Cluster und Outlier-freien Cluster und speicher die Schnittmenge abgeschlossen")






if __name__ == "__main__":
    data_flash = DATA_PREPROCESSING(dir_name='beat-the-felix-data')   # Initialisiert die Datenpipeline
    data_flash.rough_filter(output_folder='data_trash', threshold=1.0)   # 1. Grobes Filtern der Dateien (entfernt Dateien mit zu hohen d50-Werten oder falschem Format)
    data_flash.load_all_rough_data()                                      # 2. Alle gefilterten Dateien laden und zu einem großen DataFrame zusammenfassen
    data_flash.normalize_rough_dataframe(scaler_path='scaler/scaler_df_all.joblib')   # 3. Daten skalieren (Standardisierung), Scaler speichern
    # data_flash.remove_outliers_zscore(threshold=3)
    data_flash.hdbscan_clustering(cluster_path="cluster_data/cluster_df_all.joblib", min_cluster_size=500)   # 4. Clustering mit HDBSCAN, Modell speichern
    data_flash.compute_pca_2d()                                           # 5. PCA auf 2 Dimensionen (für spätere Visualisierung)
    data_flash.plot_hdbscan(do_plot=False)                                             # 6. Cluster-Visualisierung nach PCA
    data_flash.cluster_summary()                                          # 7. Ausgabe der Clusteranzahl und Noise-Punkte
    data_flash.process_files_in_dir()                                     # 8. Anwenden des Cluster-Modells auf alle Einzeldateien (Klassifizierung)
    data_flash.group_data_by_cluster(folder_path="cluster_predictions")   # 9. Gruppieren aller Datenpunkte nach Clusterzugehörigkeit
    data_flash.remove_outliers_inplace(percentile=95)                     # 10. Entfernen von Ausreißern innerhalb der Cluster (Mahalanobis)
    # data_flash.save_cluster_scalers(save_dir="scaler")                    # 11. Speichern der Cluster-Scaler (Trennung pro Cluster und Input und Output)
    data_flash.plot_clusters_after_outlier_removal(do_plot=False)         # 12. Visualisierung der Cluster nach Outlier-Filter
    data_flash.save_clusters_to_csv(save_dir="cluster_csv")               # 13. Speichern der bereinigten Cluster als separate CSV-Dateien
    data_flash.filter_predictions_with_clusters()                         # 14. Vergleich/Schnittmenge: Predictions vs. bereinigte Cluster, Speichern der gemeinsamen Punkte



# Erweiterte Erläuterung der Methoden
"""
if __name__ == "__main__":
    dir_name = 'data'  # Ordner mit deinen Rohdaten
    
    data_flash = DATA_PREPROCESSING(dir_name=dir_name)

    # 1. Grobes Filtern der Dateien
    # -> Entfernt alle .txt-Dateien, bei denen der Mittelwert von 'd50' größer als threshold ist.
    # -> Verschiebt ungeeignete Dateien oder Nicht-Textdateien in 'data_trash'
    data_flash.rough_filter(output_folder='data_trash', threshold=1.0)

    # 2. Alle (gefilterten) Dateien laden und zu einem großen DataFrame zusammenfassen
    data_flash.load_all_rough_data()

    # 3. Skalieren aller Daten (Standardisierung, Mittelwert 0, Std 1)
    # -> Speichert den Scaler für späteres Klassifizieren
    data_flash.normalize_rough_dataframe(scaler_path='scaler/scaler_df_all.joblib')

    # 4. Clustering der normalisierten Daten mit HDBSCAN
    # -> Erkennt Cluster und speichert das Cluster-Modell für spätere Nutzung
    data_flash.hdbscan_clustering(cluster_path="cluster_data/cluster_df_all.joblib", min_cluster_size=500)

    # 5. Reduktion der Daten auf 2 Dimensionen (PCA) zur späteren Visualisierung
    data_flash.compute_pca_2d()

    # 6. Plotten der Clustersituation nach PCA-Reduktion
    data_flash.plot_hdbscan()

    # 7. Zusammenfassen & Ausgeben der Clusteranzahl und der Noise-Punkte
    data_flash.cluster_summary()

    # 8. Alle Einzeldateien erneut laden, skalieren und mit dem gespeicherten Modell klassifizieren
    # -> Die zugehörigen Clusterlabels werden gespeichert (für alle Dateien im Verzeichnis)
    data_flash.process_files_in_dir()

    # 9. Gruppieren aller Datenpunkte nach ihrem zugeordneten Cluster
    # -> Erstellt einen DataFrame pro Cluster (aus allen klassifizierten Punkten)
    data_flash.group_data_by_cluster(folder_path="cluster_predictions")

    # 10. Entfernt Ausreißer innerhalb jedes Clusters
    # -> Verwendet Mahalanobis-Distanz und filtert auf die (hier) 97% "inneren" Punkte pro Cluster
    data_flash.remove_outliers_inplace(percentile=97)

    # 11. Speichert für jeden bereinigten Cluster einen Input- und Output-Scaler
    # -> Jetzt sind die Cluster-Daten ausreißerbereinigt, Scaler werden auf diesen Daten trainiert & gespeichert
    data_flash.save_cluster_scalers(save_dir="scaler")

    # 12. Visualisiert die Cluster nach Outlier-Entfernung (erneut PCA auf Inputs)
    data_flash.plot_clusters_after_outlier_removal()

    # 13. Speichert die bereinigten Clusterdaten als separate CSV-Dateien
    # -> Für jedes Cluster gibt es eine eigene CSV mit den wichtigsten Spalten
    data_flash.save_clusters_to_csv(save_dir="cluster_csv")

    # 14. Vergleicht alle Vorhersagepunkte mit den finalen Clustern
    # -> Sucht Schnittmengen zwischen klassifizierten Einzeldateien und Clustern, speichert die gemeinsamen Punkte
    data_flash.filter_predictions_with_clusters()
"""



# Fließschemata
"""
+------------------+
| Rohdaten (.txt)  |
+------------------+
        |
        v
+----------------------------+
| Grobes Filtern (d50 etc.)  |
| --> Unerwünschte Dateien   |
|     in 'data_trash'        |
+----------------------------+
         |
         v
+-----------------------------+
| Daten einlesen & zusammen-  |
| fassen (1 großer DataFrame) |
+-----------------------------+
         |
         v
+--------------------------+
| Daten normalisieren      |
| (StandardScaler)         |
| --> Scaler speichern     |
+--------------------------+
         |
         v
+------------------------------+
| Clustering (HDBSCAN)         |
| --> Cluster-Modell speichern |
+------------------------------+
         |
         v
+---------------------+
| PCA auf 2D          |
| (für Visualisierung)|
+---------------------+
         |
         v
+-------------------+
| Plot Clusters     |
+-------------------+
         |
         v
+--------------------------+
| Cluster-Statistiken      |
+--------------------------+
         |
         v
+-----------------------------------------------------+
| Klassifizierung aller Einzeldateien:                |
| Skalieren --> Cluster vorhersagen (approx_predict)  |
+-----------------------------------------------------+
         |
         v
+-------------------------------------+
| Gruppieren nach Clusterzugehörigkeit|
+-------------------------------------+
         |
         v
+-----------------------------+
| Outlier-Filter (Mahalanobis)|
+-----------------------------+
         |
         v
+-------------------------------+
| Speichern der Cluster-Scaler  |
| (pro bereinigtem Cluster,     |
|  Inputs & Outputs getrennt)   |
+-------------------------------+
         |
         v
+----------------------+
| Plot Clusters 2D     |
| (nach Outlier-Filter)|
+----------------------+
         |
         v
+-------------------------------+
| Speichern als Cluster-CSV     |
+-------------------------------+
         |
         v
+----------------------------------------------+
| Vergleich: Schnittmenge Prediction vs Cluster|
+----------------------------------------------+
         |
         v
+------------------------+
| Fertige, gereinigte    |
| Cluster-Daten          |
+------------------------+
"""




