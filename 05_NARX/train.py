import os
import warnings
from src.config import MAIN_COLUMNS, DXX_COLUMNS, DATA_DIR, SCRIPT_DIR
from src.training.trainer import train_cluster 


warnings.filterwarnings("ignore", category=FutureWarning)

if __name__ == "__main__":
    patience = 15
    verbose = True
    epochs = 150

    # Helper to get file paths for a given cluster
    def get_paths(cluster_label):
        cluster_dir = os.path.join(DATA_DIR, f"cluster_{cluster_label}")
        return (
            os.path.join(cluster_dir, "train_all.csv"),
            os.path.join(cluster_dir, "val_all.csv"),
            os.path.join(cluster_dir, "test_all.csv"),
        )

    # Cluster 0 - main (replace params with 0_main)
    train_path, val_path, test_path = get_paths("0")
    train_cluster(
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        label="0",
        lag=4,
        hidden_sizes=[55, 81, 19],             # 3 layers
        loss_function="huber",
        activation="gelu",
        epochs=epochs,
        batch_size=128,
        lr=0.00033927047566357317,
        l1_lambda=2.357774182695726e-08,
        l2_lambda=5.153162433047658e-07,
        dropout=0.011034169871864384,
        batchnorm=False,
        output_activation=None,
        patience=patience,
        verbose=verbose,
        output_columns=MAIN_COLUMNS,
        model_suffix="_main"
    )

    # Cluster 0 - dxx (replace params with o_dxx)
    train_path, val_path, test_path = get_paths("0")
    train_cluster(
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        label="0",
        lag=5,
        hidden_sizes=[32],                     # 1 layer
        loss_function="huber",
        activation="relu",
        epochs=epochs,
        batch_size=64,
        lr=0.000138879067742347,
        l1_lambda=1.3314573336711053e-08,
        l2_lambda=1.1582437826333968e-10,
        dropout=0.06971852651796423,
        batchnorm=False,
        output_activation=None,
        patience=patience,
        verbose=verbose,
        output_columns=DXX_COLUMNS,
        model_suffix="_dxx"
    )

    # Cluster 1 - main (replace params with 1_main)
    train_path, val_path, test_path = get_paths("1")
    train_cluster(
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        label="1",
        lag=5,
        hidden_sizes=[72, 64, 39],             # 3 layers
        loss_function="huber",
        activation="gelu",
        epochs=epochs,
        batch_size=128,
        lr=0.0002012998313494599,
        l1_lambda=1.0795335391966569e-08,
        l2_lambda=7.29604113633572e-08,
        dropout=0.028346535275646886,
        batchnorm=False,
        output_activation=None,
        patience=patience,
        verbose=verbose,
        output_columns=MAIN_COLUMNS,
        model_suffix="_main"
    )

    # Cluster 1 - dxx (replace params with 1_dxx)
    train_path, val_path, test_path = get_paths("1")
    train_cluster(
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        label="1",
        lag=4,
        hidden_sizes=[97],                     # 1 layer
        loss_function="huber",
        activation="relu",
        epochs=epochs,
        batch_size=64,
        lr=0.0002522126038462181,
        l1_lambda=6.857678018642172e-08,
        l2_lambda=1.1629403323394067e-09,
        dropout=0.011902622428536354,
        batchnorm=False,
        output_activation=None,
        patience=patience,
        verbose=verbose,
        output_columns=DXX_COLUMNS,
        model_suffix="_dxx"
    )

    print("Training complete.")
