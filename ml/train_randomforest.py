import json
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROCESSED_DIR = Path("data/processed")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main():
    train_df = pd.read_csv(PROCESSED_DIR / "train_processed.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test_processed.csv")

    with open(PROCESSED_DIR / "tabular_feature_columns.json", "r", encoding="utf-8") as f:
        feature_columns = json.load(f)

    X_train = train_df[feature_columns]
    y_train = train_df["Appliances"]

    X_test = test_df[feature_columns]
    y_test = test_df["Appliances"]

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        n_jobs=1,
    )

    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_mae = float(mean_absolute_error(y_train, train_pred))
    train_rmse = rmse(y_train, train_pred)
    test_mae = float(mean_absolute_error(y_test, test_pred))
    test_rmse = rmse(y_test, test_pred)

    model_path = MODEL_DIR / "randomforest_model.pkl"
    joblib.dump(model, model_path)
    feature_importance_path = PROCESSED_DIR / "randomforest_feature_importance.csv"

    importances = (
        pd.DataFrame(
            {
                "feature": feature_columns,
                "importance": model.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    importances.to_csv(feature_importance_path, index=False)

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("smart-home-energy-forecast")

    with mlflow.start_run(run_name="randomforest_lag_rolling"):
        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("max_depth", 12)
        mlflow.log_param("n_jobs", 1)
        mlflow.log_param("feature_count", len(feature_columns))
        mlflow.log_metric("train_mae", train_mae)
        mlflow.log_metric("train_rmse", train_rmse)
        mlflow.log_metric("final_mae", test_mae)
        mlflow.log_metric("final_rmse", test_rmse)
        mlflow.log_metric("mae_gap", test_mae - train_mae)
        mlflow.log_metric("rmse_gap", test_rmse - train_rmse)
        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(PROCESSED_DIR / "tabular_feature_columns.json"))
        mlflow.log_artifact(str(feature_importance_path))

    print("RandomForest 완료")
    print(f"Train MAE: {train_mae:.2f}")
    print(f"Train RMSE: {train_rmse:.2f}")
    print(f"Test MAE: {test_mae:.2f}")
    print(f"Test RMSE: {test_rmse:.2f}")
    print(f"모델 저장: {model_path}")
    print("\nTop 10 Feature Importances")
    print(importances.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
