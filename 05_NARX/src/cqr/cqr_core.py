"""Conformalized quantile regression core functions."""
import torch
import torch.nn as nn

class PinballLoss(nn.Module):
    """
    Pinball loss for quantile regression.
    """
    def __init__(self, tau):
        super().__init__()
        self.tau = tau

    def forward(self, y_pred, y_true):
        delta = y_true - y_pred
        return torch.mean(torch.maximum(self.tau * delta, (self.tau - 1) * delta))

def make_quantile_net(input_dim, hidden_dim=64, num_layers=2):
    """
    Build a simple feedforward quantile regression net.
    """
    layers = []
    last_dim = input_dim
    for _ in range(num_layers):
        layers.append(nn.Linear(last_dim, hidden_dim))
        layers.append(nn.ReLU())
        last_dim = hidden_dim
    layers.append(nn.Linear(hidden_dim, 1))
    return nn.Sequential(*layers)

def train_with_early_stopping(model, criterion, optimizer, train_loader, val_loader,
                              device, n_epochs=30, patience=10, save_path="temp_model.pth"):
    """
    Train model with early stopping on validation loss.
    """
    best_val_loss = float('inf')
    counter = 0
    for epoch in range(n_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        model.eval()
        val_loss_epoch = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss_epoch += loss.item() * X_batch.size(0)
        val_loss_epoch /= len(val_loader.dataset)
        if val_loss_epoch < best_val_loss - 1e-6:
            best_val_loss = val_loss_epoch
            counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            counter += 1
            if counter >= patience:
                print(f"⏹️ Early Stopping at epoch {epoch+1} - best Val Loss: {best_val_loss:.6f}")
                break
    model.load_state_dict(torch.load(save_path))
