import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

RAW_PATH = Path("data/raw/energydata_complete.csv")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COLUMN = "Appliances"


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["hour"] = df["date"].dt.hour
    df["dayofweek"] = df["date"].dt.dayofweek
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["dayofweek_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dayofweek_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

    return df


def add_lag_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    lag_steps = [1, 3, 6, 12, 24]

    for lag in lag_steps:
        df[f"appliances_lag_{lag}"] = df[TARGET_COLUMN].shift(lag)

    # 현재 시점 값을 포함하지 않기 위해 shift(1) 후 rolling
    shifted_target = df[TARGET_COLUMN].shift(1)

    df["appliances_rolling_mean_6"] = shifted_target.rolling(window=6).mean()
    df["appliances_rolling_std_6"] = shifted_target.rolling(window=6).std()

    df["appliances_rolling_mean_24"] = shifted_target.rolling(window=24).mean()
    df["appliances_rolling_std_24"] = shifted_target.rolling(window=24).std()

    return df


def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {RAW_PATH}")

    df = pd.read_csv(RAW_PATH)
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y %H:%M")
    df = df.sort_values("date").reset_index(drop=True)

    df = add_time_features(df)
    df = add_lag_rolling_features(df)

    sensor_columns = [
        "T1", "RH_1",
        "T2", "RH_2",
        "T3", "RH_3",
        "T4", "RH_4",
        "T5", "RH_5",
        "T6", "RH_6",
        "T7", "RH_7",
        "T8", "RH_8",
        "T9", "RH_9",
    ]

    weather_columns = [
        "T_out",
        "RH_out",
        "Windspeed",
        "Visibility",
        "Tdewpoint",
        "Press_mm_hg",
    ]

    time_columns = [
        "is_weekend",
        "hour_sin",
        "hour_cos",
        "dayofweek_sin",
        "dayofweek_cos",
    ]

    lag_columns = [
        "appliances_lag_1",
        "appliances_lag_3",
        "appliances_lag_6",
        "appliances_lag_12",
        "appliances_lag_24",
    ]

    rolling_columns = [
        "appliances_rolling_mean_6",
        "appliances_rolling_std_6",
        "appliances_rolling_mean_24",
        "appliances_rolling_std_24",
    ]

    tabular_feature_columns = (
        sensor_columns
        + weather_columns
        + time_columns
        + lag_columns
        + rolling_columns
    )

    # LSTM은 과거 Appliances 값 자체도 sequence 입력으로 사용
    lstm_feature_columns = [TARGET_COLUMN] + tabular_feature_columns

    required_columns = ["date", TARGET_COLUMN] + tabular_feature_columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"누락된 컬럼이 있습니다: {missing_columns}")

    df = df[["date", TARGET_COLUMN] + tabular_feature_columns].dropna().reset_index(drop=True)

    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()

    # RandomForest용 데이터 저장
    train_df.to_csv(PROCESSED_DIR / "train_processed.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test_processed.csv", index=False)

    X_train_tabular = train_df[tabular_feature_columns].values
    y_train = train_df[TARGET_COLUMN].values

    X_test_tabular = test_df[tabular_feature_columns].values
    y_test = test_df[TARGET_COLUMN].values

    np.save(PROCESSED_DIR / "X_train_tabular.npy", X_train_tabular)
    np.save(PROCESSED_DIR / "y_train.npy", y_train)
    np.save(PROCESSED_DIR / "X_test_tabular.npy", X_test_tabular)
    np.save(PROCESSED_DIR / "y_test.npy", y_test)

    # LSTM용 scaling: train에만 fit
    scaler = StandardScaler()
    train_lstm_scaled = scaler.fit_transform(train_df[lstm_feature_columns])
    test_lstm_scaled = scaler.transform(test_df[lstm_feature_columns])

    np.save(PROCESSED_DIR / "train_lstm_scaled.npy", train_lstm_scaled)
    np.save(PROCESSED_DIR / "test_lstm_scaled.npy", test_lstm_scaled)

    with open(PROCESSED_DIR / "lstm_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    with open(PROCESSED_DIR / "tabular_feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(tabular_feature_columns, f, ensure_ascii=False, indent=2)

    with open(PROCESSED_DIR / "lstm_feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(lstm_feature_columns, f, ensure_ascii=False, indent=2)

    print("전처리 완료")
    print(f"전체 데이터 수: {len(df)}")
    print(f"Train rows: {len(train_df)}")
    print(f"Test rows: {len(test_df)}")
    print(f"Tabular feature count: {len(tabular_feature_columns)}")
    print(f"LSTM feature count: {len(lstm_feature_columns)}")
    print("Target:", TARGET_COLUMN)


if __name__ == "__main__":
    main()
