import json
import pickle
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader

from ml.dataset import EnergySequenceDataset

PROCESSED_DIR = Path("data/processed")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

TARGET_COL_IDX = 0


class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        output, _ = self.lstm(x)
        last_hidden = output[:, -1, :]
        return self.fc(last_hidden).squeeze(-1)


def inverse_target_scale(values: np.ndarray, scaler, target_col_idx: int = 0):
    return values * scaler.scale_[target_col_idx] + scaler.mean_[target_col_idx]


def evaluate(model, loader, criterion, device, scaler):
    model.eval()

    losses = []
    scaled_preds = []
    scaled_targets = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            loss = criterion(pred, y)

            losses.append(loss.item())
            scaled_preds.extend(pred.cpu().numpy())
            scaled_targets.extend(y.cpu().numpy())

    scaled_preds = np.array(scaled_preds)
    scaled_targets = np.array(scaled_targets)

    preds = inverse_target_scale(scaled_preds, scaler, TARGET_COL_IDX)
    targets = inverse_target_scale(scaled_targets, scaler, TARGET_COL_IDX)

    mae = float(mean_absolute_error(targets, preds))
    score_rmse = float(np.sqrt(mean_squared_error(targets, preds)))

    return float(np.mean(losses)), mae, score_rmse


def main():
    seq_len = 24
    batch_size = 64
    hidden_size = 64
    num_layers = 1
    epochs = 10
    learning_rate = 0.001

    train_data = np.load(PROCESSED_DIR / "train_lstm_scaled.npy")
    test_data = np.load(PROCESSED_DIR / "test_lstm_scaled.npy")

    with open(PROCESSED_DIR / "lstm_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open(PROCESSED_DIR / "lstm_feature_columns.json", "r", encoding="utf-8") as f:
        feature_columns = json.load(f)

    train_dataset = EnergySequenceDataset(
        train_data,
        seq_len=seq_len,
        target_col_idx=TARGET_COL_IDX,
    )
    test_dataset = EnergySequenceDataset(
        test_data,
        seq_len=seq_len,
        target_col_idx=TARGET_COL_IDX,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    input_size = train_data.shape[1]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = LSTMRegressor(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("smart-home-energy-forecast")

    with mlflow.start_run(run_name="lstm_seq24"):
        mlflow.log_param("model_type", "LSTM")
        mlflow.log_param("seq_len", seq_len)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("hidden_size", hidden_size)
        mlflow.log_param("num_layers", num_layers)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("input_size", input_size)
        mlflow.log_param("feature_count", len(feature_columns))
        mlflow.log_param("target", "Appliances")

        for epoch in range(epochs):
            model.train()
            train_losses = []

            for x, y in train_loader:
                x = x.to(device)
                y = y.to(device)

                optimizer.zero_grad()
                pred = model(x)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

            avg_train_loss = float(np.mean(train_losses))
            test_loss, test_mae, test_rmse = evaluate(
                model, test_loader, criterion, device, scaler
            )

            mlflow.log_metric("train_loss", avg_train_loss, step=epoch + 1)
            mlflow.log_metric("test_loss", test_loss, step=epoch + 1)
            mlflow.log_metric("test_mae", test_mae, step=epoch + 1)
            mlflow.log_metric("test_rmse", test_rmse, step=epoch + 1)

            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Test Loss: {test_loss:.4f} | "
                f"MAE: {test_mae:.2f} | "
                f"RMSE: {test_rmse:.2f}"
            )

        final_test_loss, final_mae, final_rmse = evaluate(
            model, test_loader, criterion, device, scaler
        )

        model_path = MODEL_DIR / "energy_lstm_model.pt"

        torch.save(
            {
                "model_type": "LSTM",
                "state_dict": model.state_dict(),
                "input_size": input_size,
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "seq_len": seq_len,
                "feature_columns": feature_columns,
                "target_col_idx": TARGET_COL_IDX,
            },
            model_path,
        )

        mlflow.log_metric("final_test_loss", final_test_loss)
        mlflow.log_metric("final_mae", final_mae)
        mlflow.log_metric("final_rmse", final_rmse)
        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(PROCESSED_DIR / "lstm_feature_columns.json"))

        print("\nLSTM 학습 완료")
        print(f"모델 저장 위치: {model_path}")
        print(f"Final MAE: {final_mae:.2f}")
        print(f"Final RMSE: {final_rmse:.2f}")


if __name__ == "__main__":
    main()