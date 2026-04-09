import pandas as pd
import numpy as np
import xgboost as xgb
from prophet import Prophet
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

def train_prophet(item_df):
    prophet_df = item_df[["date","sales"]].copy()
    prophet_df.columns = ["ds","y"]
    prophet_df = prophet_df.sort_values("ds")
    m = Prophet(
        seasonality_mode="multiplicative",
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
    )
    m.add_country_holidays(country_name="US")
    m.fit(prophet_df)
    future = m.make_future_dataframe(periods=0)
    forecast = m.predict(future)[["ds","yhat"]]
    forecast.columns = ["date","prophet_pred"]
    forecast["prophet_pred"] = forecast["prophet_pred"].clip(lower=0)
    return forecast

def run_ensemble(item_id, df, w_xgb=0.7, w_prophet=0.3):
    item_df = df[df["item_id"] == item_id].copy()
    item_df = item_df.sort_values("date").reset_index(drop=True)

    # train/test split — last 28 days as test
    split_date = item_df["date"].max() - pd.Timedelta(days=28)
    train = item_df[item_df["date"] <= split_date]
    test  = item_df[item_df["date"] >  split_date]

    # --- XGBoost ---
    xgb_model = xgb.XGBRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
    )
    X_train = train[FEATURES].fillna(0)
    X_test  = test[FEATURES].fillna(0)
    xgb_model.fit(X_train, train[TARGET], verbose=False)
    xgb_preds = np.maximum(xgb_model.predict(X_test), 0)

    # --- Prophet ---
    prophet_forecast = train_prophet(train)
    test_with_prophet = test.merge(prophet_forecast, on="date", how="left")
    prophet_preds = test_with_prophet["prophet_pred"].fillna(0).values

    # --- Ensemble ---
    ensemble_preds = w_xgb * xgb_preds + w_prophet * prophet_preds

    # --- Evaluate ---
    actuals = test[TARGET].values
    mask = actuals > 0

    if mask.sum() == 0:
        return None

    mape_xgb      = mean_absolute_percentage_error(actuals[mask], xgb_preds[mask])
    mape_prophet  = mean_absolute_percentage_error(actuals[mask], prophet_preds[mask])
    mape_ensemble = mean_absolute_percentage_error(actuals[mask], ensemble_preds[mask])

    return {
        "item_id": item_id,
        "mape_xgb": mape_xgb,
        "mape_prophet": mape_prophet,
        "mape_ensemble": mape_ensemble
    }

if __name__ == "__main__":
    print("Loading data...")
    df = pd.read_csv("data/processed/ca1_active.csv", parse_dates=["date"])
    df[TARGET] = df[TARGET].clip(lower=0)

    # run ensemble on top 20 items by sales volume
    top_items = (df.groupby("item_id")["sales"]
                   .sum()
                   .nlargest(20)
                   .index.tolist())

    print(f"Running ensemble on top {len(top_items)} items...\n")
    results = []
    for i, item in enumerate(top_items):
        r = run_ensemble(item, df)
        if r:
            results.append(r)
            print(f"[{i+1:2d}] {item:25s} | "
                  f"XGB: {r['mape_xgb']:.3f} | "
                  f"Prophet: {r['mape_prophet']:.3f} | "
                  f"Ensemble: {r['mape_ensemble']:.3f}")

    results_df = pd.DataFrame(results)
    print(f"\n--- SUMMARY (mean across {len(results_df)} items) ---")
    print(f"XGBoost  MAPE : {results_df['mape_xgb'].mean():.4f} "
          f"({results_df['mape_xgb'].mean()*100:.1f}%)")
    print(f"Prophet  MAPE : {results_df['mape_prophet'].mean():.4f} "
          f"({results_df['mape_prophet'].mean()*100:.1f}%)")
    print(f"Ensemble MAPE : {results_df['mape_ensemble'].mean():.4f} "
          f"({results_df['mape_ensemble'].mean()*100:.1f}%)")

    improvement = (results_df['mape_xgb'].mean() -
                   results_df['mape_ensemble'].mean()) / results_df['mape_xgb'].mean() * 100
    print(f"Improvement   : {improvement:.1f}% better than XGBoost alone")

    results_df.to_csv("data/processed/ensemble_results.csv", index=False)
    print("\nSaved to data/processed/ensemble_results.csv")
