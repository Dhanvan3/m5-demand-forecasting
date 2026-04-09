import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_percentage_error
import mlflow
import mlflow.xgboost
import warnings
warnings.filterwarnings("ignore")

FEATURES = [
    "lag_7","lag_14","lag_28",
    "rolling_mean_7","rolling_mean_28",
    "rolling_std_7","rolling_std_28",
    "day_of_week","day_of_month","week_of_year",
    "month","year","is_weekend",
    "has_event","snap_flag",
    "sell_price","price_change"
]
TARGET = "sales"

def walk_forward_cv(df, n_splits=3, horizon=28):
    dates = sorted(df["date"].unique())
    split_size = len(dates) // (n_splits + 1)
    scores = []

    for i in range(1, n_splits + 1):
        train_end = dates[split_size * i]
        test_end  = dates[min(split_size * i + horizon, len(dates) - 1)]

        train = df[df["date"] <= train_end]
        test  = df[(df["date"] > train_end) & (df["date"] <= test_end)]

        model = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            early_stopping_rounds=50,
            eval_metric="rmse",
            random_state=42,
            n_jobs=-1
        )
        model.fit(
            train[FEATURES], train[TARGET],
            eval_set=[(test[FEATURES], test[TARGET])],
            verbose=False
        )

        preds = np.maximum(model.predict(test[FEATURES]), 0)

        # only score on rows where actual sales > 0
        mask = test[TARGET] > 0
        mape = mean_absolute_percentage_error(
            test[TARGET][mask], preds[mask]
        )
        scores.append(mape)
        print(f"  Fold {i} MAPE: {mape:.4f} ({mape*100:.1f}%)")

    return model, scores

def train():
    print("Loading active items dataset...")
    df = pd.read_csv("data/processed/ca1_active.csv", parse_dates=["date"])
    df[TARGET] = df[TARGET].clip(lower=0)
    df[FEATURES] = df[FEATURES].fillna(0)

    print(f"Shape: {df.shape}")
    print(f"Items: {df['item_id'].nunique()}")
    print(f"Date range: {df['date'].min()} → {df['date'].max()}\n")

    mlflow.set_experiment("m5-demand-forecasting")

    with mlflow.start_run(run_name="xgb_active_items_v2"):
        print("Running walk-forward CV on active items...")
        model, scores = walk_forward_cv(df)

        mean_mape = np.mean(scores)
        accuracy  = (1 - mean_mape) * 100
        print(f"\nMean MAPE : {mean_mape:.4f} ({mean_mape*100:.1f}%)")
        print(f"Accuracy  : {accuracy:.1f}%")

        mlflow.log_params({
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "dataset": "active_items_only",
            "zero_pct_threshold": 0.70
        })
        for i, s in enumerate(scores):
            mlflow.log_metric(f"fold_{i+1}_mape", s)
        mlflow.log_metric("mean_mape", mean_mape)
        mlflow.log_metric("approx_accuracy_pct", accuracy)

        mlflow.xgboost.log_model(
            model, "xgb_model",
            registered_model_name="m5-xgboost-forecaster"
        )
        print("\nModel v2 registered in MLflow!")

    return model

if __name__ == "__main__":
    model = train()
