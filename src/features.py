import pandas as pd
import numpy as np

def add_features(df):
    df = df.copy()
    df.sort_values(["item_id", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    g = df.groupby("item_id")["sales"]

    # --- LAG FEATURES ---
    # "what did this item sell 7/14/28 days ago"
    for lag in [7, 14, 28]:
        df[f"lag_{lag}"] = g.shift(lag)

    # --- ROLLING STATS ---
    # shift(1) first to avoid leaking today's sales into the window
    for window in [7, 28]:
        df[f"rolling_mean_{window}"] = (
            g.shift(1).rolling(window).mean().reset_index(0, drop=True)
        )
        df[f"rolling_std_{window}"] = (
            g.shift(1).rolling(window).std().reset_index(0, drop=True)
        )

    # --- CALENDAR FEATURES ---
    df["day_of_week"]  = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"]        = df["date"].dt.month
    df["year"]         = df["date"].dt.year
    df["is_weekend"]   = (df["date"].dt.dayofweek >= 5).astype(int)

    # --- EVENT & PROMO FLAGS ---
    df["has_event"] = df["event_name_1"].notna().astype(int)
    df["snap_flag"] = df["snap_CA"].fillna(0).astype(int)

    # --- PRICE FEATURES ---
    df["sell_price"]   = df["sell_price"].fillna(method="ffill")
    df["price_change"] = df.groupby("item_id")["sell_price"].diff()

    # --- DROP ROWS WITH NaN from lags ---
    before = len(df)
    df.dropna(subset=[f"lag_{l}" for l in [7,14,28]] +
                     [f"rolling_mean_{w}" for w in [7,28]], inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"Dropped {before - len(df)} rows due to lag NaNs")
    print(f"Final shape after features: {df.shape}")
    print(f"Features added: {[c for c in df.columns if c not in ['id','item_id','dept_id','cat_id','store_id','state_id','d','date','wm_yr_wk','event_name_1','event_type_1','snap_CA','sales']]}")
    return df

if __name__ == "__main__":
    df = pd.read_csv("data/processed/ca1_long.csv", parse_dates=["date"])
    df = add_features(df)
    df.to_csv("data/processed/ca1_features.csv", index=False)
    print("\nSaved to data/processed/ca1_features.csv")
