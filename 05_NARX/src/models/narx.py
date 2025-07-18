"""MLPNARX model and related neural nets."""
import torch
import numpy as np

class MLPNARX(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes, activation="relu",
                 dropout=0.0, batchnorm=False, output_activation=None):
        super().__init__()
        acts = {
            "relu": torch.nn.ReLU(),
            "tanh": torch.nn.Tanh(),
            "sigmoid": torch.nn.Sigmoid(),
            "gelu": torch.nn.GELU()
        }
        layers = []
        last_dim = input_size
        for h in hidden_sizes:
            layers.append(torch.nn.Linear(last_dim, h))
            if batchnorm:
                layers.append(torch.nn.BatchNorm1d(h))
            layers.append(acts[activation])
            if dropout > 0.0:
                layers.append(torch.nn.Dropout(dropout))
            last_dim = h
        layers.append(torch.nn.Linear(last_dim, output_size))
        if output_activation is not None and output_activation in acts:
            layers.append(acts[output_activation])
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def create_narx_lagged_data_tim(df, input_cols, output_cols, lag):
    X_narx, y_narx = [], []
    X = df[input_cols].values
    y = df[output_cols].values
    for t in range(lag, len(df)):
        x_lagged = X[t - lag:t].flatten()
        y_lagged = y[t - lag:t - 1].flatten()
        x_full = np.concatenate([x_lagged, y_lagged])
        X_narx.append(x_full)
        y_narx.append(y[t])
    return np.array(X_narx), np.array(y_narx)
