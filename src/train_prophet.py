import pandas as pd
import numpy as np
from prophet import Prophet
import mlflow
import warnings
warnings.filterwarnings("ignore")

def train_prophet_item(df, item_id):
    item_df = df[df["item_id"] == item_id][["date","sales"]].copy()
    item_df.columns = ["ds", "y"]
    item_df = item_df.sort_values("ds")

    m = Prophet(
        seasonality_mode="multiplicative",
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05
    )
    m.add_country_holidays(country_name="US")
    m.fit(item_df)

    future   = m.make_future_dataframe(periods=28)
    forecast = m.predict(future)
    return m, forecast[["ds","yhat","yhat_lower","yhat_upper"]]

if __name__ == "__main__":
    print("Loading data...")
    df = pd.read_csv("data/processed/ca1_active.csv", parse_dates=["date"])

    # pick the top selling item
    top_item = df.groupby("item_id")["sales"].sum().idxmax()
    print(f"Training Prophet on top item: {top_item}")

    m, forecast = train_prophet_item(df, top_item)

    # show last 10 forecast rows
    print("\nForecast (last 10 rows):")
    print(forecast.tail(10).to_string(index=False))

    forecast.to_csv("data/processed/prophet_forecast.csv", index=False)
    print(f"\nSaved forecast to data/processed/prophet_forecast.csv")

    # plot
    fig = m.plot(m.predict(m.make_future_dataframe(periods=28)))
    fig.savefig("data/processed/prophet_plot.png")
    print("Saved prophet plot to data/processed/prophet_plot.png")
