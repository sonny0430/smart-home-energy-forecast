import os
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from dotenv import load_dotenv

load_dotenv()


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


class EnergyPredictor:
    def __init__(self):
        self.model_path = Path(os.getenv("MODEL_PATH", "models/energy_lstm_model.pt"))
        self.scaler_path = Path(os.getenv("SCALER_PATH", "data/processed/lstm_scaler.pkl"))
        self.test_data_path = Path(os.getenv("TEST_DATA_PATH", "data/processed/test_lstm_scaled.npy"))

        if not self.model_path.exists():
            raise FileNotFoundError(f"모델 파일이 없습니다: {self.model_path}")

        if not self.scaler_path.exists():
            raise FileNotFoundError(f"스케일러 파일이 없습니다: {self.scaler_path}")

        if not self.test_data_path.exists():
            raise FileNotFoundError(f"테스트 데이터 파일이 없습니다: {self.test_data_path}")

        checkpoint = torch.load(self.model_path, map_location="cpu")

        self.input_size = checkpoint["input_size"]
        self.hidden_size = checkpoint["hidden_size"]
        self.num_layers = checkpoint["num_layers"]
        self.seq_len = checkpoint["seq_len"]
        self.target_col_idx = checkpoint["target_col_idx"]

        self.model = LSTMRegressor(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
        )

        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

        with open(self.scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

        self.test_data = np.load(self.test_data_path)

    def _inverse_target_scale(self, value: float) -> float:
        return float(
            value * self.scaler.scale_[self.target_col_idx]
            + self.scaler.mean_[self.target_col_idx]
        )

    def predict(self, forecast_steps: int = 6) -> list[float]:
        """
        MVP 예측 방식:
        - test 데이터 마지막 seq_len 구간을 최근 센서 시퀀스로 사용
        - 다음 시점 Appliances 예측
        - 예측값을 다음 입력 sequence의 Appliances 위치에 반영
        - 나머지 센서 feature는 마지막 시점 값을 유지
        """
        sequence = self.test_data[-self.seq_len :].copy()
        predictions = []

        for _ in range(forecast_steps):
            x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                scaled_pred = self.model(x).item()

            real_pred = self._inverse_target_scale(scaled_pred)
            predictions.append(round(real_pred, 2))

            next_row = sequence[-1].copy()
            next_row[self.target_col_idx] = scaled_pred

            sequence = np.vstack([sequence[1:], next_row])

        return predictions


predictor = EnergyPredictor()