import pandas as pd

df = pd.read_csv("data/processed/ca1_features.csv", parse_dates=["date"])

# check how many items have mostly zero sales
item_stats = df.groupby("item_id")["sales"].agg(["mean","sum","count"])
item_stats["zero_pct"] = (df.groupby("item_id")["sales"]
                          .apply(lambda x: (x==0).mean()).values)

print("Sales distribution across items:")
print(item_stats.describe().round(2))

# keep only items where less than 70% of days are zero
active_items = item_stats[item_stats["zero_pct"] < 0.70].index
print(f"\nTotal items: {item_stats.shape[0]}")
print(f"Active items (< 70% zero days): {len(active_items)}")

df_active = df[df["item_id"].isin(active_items)]
df_active.to_csv("data/processed/ca1_active.csv", index=False)
print(f"Saved active items: {df_active.shape}")
