import pandas as pd
import torch
from chronos import BaseChronosPipeline

CSV_PATH = "data/Fashion_Retail_Sales.csv"
MODEL_NAME = "amazon/chronos-t5-small"
PREDICTION_LENGTH = 7

def load_data():
    df = pd.read_csv(CSV_PATH)
    df["Date Purchase"] = pd.to_datetime(df["Date Purchase"])
    return df

def build_series(df: pd.DataFrame):
    # Total units per day (all regions, all items)
    daily_total = (
        df.groupby("Date Purchase")["Purchased Quantity"]
          .sum()
          .sort_index()
    )

    # Per-item units per day (all regions)
    per_item = {}
    for item_name, g in df.groupby("Item Purchased"):
        s = (
            g.groupby("Date Purchase")["Purchased Quantity"]
             .sum()
             .sort_index()
        )
        per_item[item_name] = s

    # Per-region units per day (all items)
    per_region = {}
    for region_name, g in df.groupby("Region"):
        s = (
            g.groupby("Date Purchase")["Purchased Quantity"]
             .sum()
             .sort_index()
        )
        per_region[region_name] = s

    # Per-item per-region units per day
    per_item_region = {}
    for (item_name, region_name), g in df.groupby(["Item Purchased", "Region"]):
        s = (
            g.groupby("Date Purchase")["Purchased Quantity"]
             .sum()
             .sort_index()
        )
        per_item_region[(item_name, region_name)] = s

    return daily_total, per_item, per_region, per_item_region

def load_pipeline():
    pipeline = BaseChronosPipeline.from_pretrained(
        MODEL_NAME,
        device_map="cpu",
        dtype=torch.bfloat16,
    )
    return pipeline

def forecast_series(name: str, series: pd.Series, pipeline: BaseChronosPipeline):
    print(f"\n=== Forecast for {name} ===")
    print(f"History points: {len(series)}")
    print(series.tail())

    context = torch.tensor(series.values, dtype=torch.float32)

    quantiles, _ = pipeline.predict_quantiles(
        context,
        prediction_length=PREDICTION_LENGTH,
        quantile_levels=[0.1, 0.5, 0.9],
    )

    low = quantiles[0, :, 0].numpy()
    median = quantiles[0, :, 1].numpy()
    high = quantiles[0, :, 2].numpy()

    print("\nNext", PREDICTION_LENGTH, "days forecast (units):")
    for i, (l, m, h) in enumerate(zip(low, median, high), start=1):
        print(f"Day +{i}: P10={l:.1f}, P50={m:.1f}, P90={h:.1f}")

def main():
    df = load_data()
    daily_total, per_item, per_region, per_item_region = build_series(df)
    pipeline = load_pipeline()

    # Forecast for all items combined (all regions)
    forecast_series("All Items (All Regions)", daily_total, pipeline)

    # Forecast per region (all items)
    print("\n" + "="*60)
    print("REGION-BASED FORECASTS")
    print("="*60)
    for region_name, s in per_region.items():
        forecast_series(f"All Items - {region_name}", s, pipeline)

    # Forecast per item (all regions)
    print("\n" + "="*60)
    print("ITEM-BASED FORECASTS (ALL REGIONS)")
    print("="*60)
    for item_name, s in per_item.items():
        forecast_series(item_name, s, pipeline)

    # Forecast per item per region
    print("\n" + "="*60)
    print("ITEM + REGION COMBINED FORECASTS")
    print("="*60)
    for (item_name, region_name), s in sorted(per_item_region.items()):
        forecast_series(f"{item_name} - {region_name}", s, pipeline)

if __name__ == "__main__":
    main()
