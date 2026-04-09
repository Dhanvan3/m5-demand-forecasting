import pandas as pd

def load_data():
    sales = pd.read_csv("data/raw/sales_train_validation.csv")
    calendar = pd.read_csv("data/raw/calendar.csv")
    prices = pd.read_csv("data/raw/sell_prices.csv")
    return sales, calendar, prices

def prepare(store_id="CA_1"):
    sales, calendar, prices = load_data()

    # Step 1: filter to one store only
    sales = sales[sales["store_id"] == store_id]
    print(f"Items in {store_id}: {len(sales)}")

    # Step 2: melt wide -> long
    id_cols = ["id","item_id","dept_id","cat_id","store_id","state_id"]
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    df = sales.melt(id_vars=id_cols, value_vars=day_cols,
                    var_name="d", value_name="sales")
    print(f"After melt: {df.shape}")

    # Step 3: merge calendar
    cal_cols = ["d","date","wm_yr_wk","event_name_1","event_type_1","snap_CA"]
    df = df.merge(calendar[cal_cols], on="d", how="left")
    df["date"] = pd.to_datetime(df["date"])

    # Step 4: merge prices
    df = df.merge(
        prices[["store_id","item_id","wm_yr_wk","sell_price"]],
        on=["store_id","item_id","wm_yr_wk"],
        how="left"
    )

    # Step 5: sort by item and date (critical for lag features later)
    df.sort_values(["item_id","date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"Final shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Null prices: {df['sell_price'].isna().sum()}")
    print(df.head(3))
    return df

if __name__ == "__main__":
    df = prepare("CA_1")
    df.to_csv("data/processed/ca1_long.csv", index=False)
    print("\nSaved to data/processed/ca1_long.csv")
