import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_percentage_error
import mlflow
import mlflow.xgboost
import shap
import matplotlib.pyplot as plt
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
    models = []

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

        preds = model.predict(test[FEATURES])
        preds = np.maximum(preds, 0)  # sales can't be negative
        mape = mean_absolute_percentage_error(
            test[TARGET][test[TARGET] > 0],
            preds[test[TARGET] > 0]
        )
        scores.append(mape)
        models.append(model)
        print(f"  Fold {i} MAPE: {mape:.4f} ({mape*100:.1f}%)")

    return models[-1], scores

def train():
    print("Loading features...")
    df = pd.read_csv("data/processed/ca1_features.csv", parse_dates=["date"])
    df[TARGET] = df[TARGET].clip(lower=0)
    df[FEATURES] = df[FEATURES].fillna(0)

    print(f"Data shape: {df.shape}")
    print(f"Date range: {df['date'].min()} → {df['date'].max()}\n")

    mlflow.set_experiment("m5-demand-forecasting")

    with mlflow.start_run(run_name="xgb_walkforward_v1"):

        print("Running walk-forward cross validation...")
        model, scores = walk_forward_cv(df)

        mean_mape = np.mean(scores)
        print(f"\nMean MAPE: {mean_mape:.4f} ({mean_mape*100:.1f}%)")
        accuracy  = (1 - mean_mape) * 100
        print(f"Approx accuracy: {accuracy:.1f}%")

        # log to mlflow
        mlflow.log_params({
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "n_splits": 3,
            "horizon": 28
        })
        for i, s in enumerate(scores):
            mlflow.log_metric(f"fold_{i+1}_mape", s)
        mlflow.log_metric("mean_mape", mean_mape)
        mlflow.log_metric("approx_accuracy_pct", accuracy)

        # feature importance plot
        fig, ax = plt.subplots(figsize=(8, 6))
        xgb.plot_importance(model, ax=ax, max_num_features=15,
                            title="Top 15 Feature Importances")
        plt.tight_layout()
        plt.savefig("data/processed/feature_importance.png")
        mlflow.log_artifact("data/processed/feature_importance.png")
        plt.close()
        print("\nFeature importance plot saved.")

        # shap values on a sample
        print("Computing SHAP values (this takes ~30 seconds)...")
        sample = df[FEATURES].sample(500, random_state=42)
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample)
        shap.summary_plot(shap_values, sample, show=False)
        plt.tight_layout()
        plt.savefig("data/processed/shap_summary.png")
        mlflow.log_artifact("data/processed/shap_summary.png")
        plt.close()
        print("SHAP summary plot saved.")

        # register model
        mlflow.xgboost.log_model(
            model, "xgb_model",
            registered_model_name="m5-xgboost-forecaster"
        )
        print("\nModel registered in MLflow!")

    print("\nDone! Run 'mlflow ui' to see results.")
    return model

if __name__ == "__main__":
    train()
