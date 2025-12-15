import pandas as pd
import torch
from chronos import BaseChronosPipeline   # <-- use BaseChronosPipeline

# 1) Load CSV
df = pd.read_csv("data/Fashion_Retail_Sales.csv")

# 2) Parse date column
df["Date Purchase"] = pd.to_datetime(df["Date Purchase"])

# 3) Aggregate to daily total purchase amount (proxy for demand)
daily = (
    df.groupby("Date Purchase")["Purchase Amount (USD)"]
      .sum()
      .sort_index()
)

print("Daily series head:")
print(daily.head())

# 4) Convert to tensor
context = torch.tensor(daily.values, dtype=torch.float32)

# 5) Load pre-trained Chronos model
pipeline = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",
    device_map="cpu",
    torch_dtype=torch.bfloat16,   # safe default; you can drop this if bfloat16 not supported
)

# 6) Forecast next 14 days using quantiles API
prediction_length = 14
quantiles, mean = pipeline.predict_quantiles(
    context,                       # positional, not context=...
    prediction_length=prediction_length,
    quantile_levels=[0.1, 0.5, 0.9],
)

# Take the median path (0.5 quantile)
median_forecast = quantiles[0, :, 1]  # shape: (prediction_length,)

# 7) Attach future dates
last_date = daily.index[-1]
future_dates = pd.date_range(
    last_date + pd.Timedelta(days=1),
    periods=prediction_length,
    freq="D",
)

forecast_df = pd.DataFrame({
    "date": future_dates,
    "forecast_sales_usd": median_forecast.numpy(),
})

print("\nForecast:")
print(forecast_df)