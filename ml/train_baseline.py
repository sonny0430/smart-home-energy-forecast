from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROCESSED_DIR = Path("data/processed")


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main():
    test_df = pd.read_csv(PROCESSED_DIR / "test_processed.csv")

    y_true = test_df["Appliances"].values
    y_pred = test_df["appliances_lag_1"].values

    mae = float(mean_absolute_error(y_true, y_pred))
    score_rmse = rmse(y_true, y_pred)

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("smart-home-energy-forecast")

    with mlflow.start_run(run_name="naive_baseline"):
        mlflow.log_param("model_type", "NaiveBaseline")
        mlflow.log_param("prediction_rule", "next_value_equals_previous_appliances")
        mlflow.log_metric("final_mae", mae)
        mlflow.log_metric("final_rmse", score_rmse)

    print("Naive Baseline 완료")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {score_rmse:.2f}")


if __name__ == "__main__":
    main()