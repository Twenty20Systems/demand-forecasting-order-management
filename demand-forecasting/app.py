import pandas as pd
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from chronos import BaseChronosPipeline
from typing import Optional

app = FastAPI()

# Load data and calculate units
df = pd.read_csv("data/Fashion_Retail_Sales.csv")
df["Date Purchase"] = pd.to_datetime(df["Date Purchase"])

# Estimate units: assume average item price across all transactions
avg_price_per_unit = df["Purchase Amount (USD)"].mean()
print(f"Using avg price per unit: ${avg_price_per_unit:.2f}")

# Convert USD to estimated units per day
df["Estimated Units"] = df["Purchase Amount (USD)"] / avg_price_per_unit
daily_units = (
    df.groupby("Date Purchase")["Estimated Units"]
      .sum()
      .sort_index()
)

print("Daily units head:")
print(daily_units.head())

# Load model (forecast on units now)
pipeline = BaseChronosPipeline.from_pretrained("amazon/chronos-t5-small", device_map="cpu", dtype=torch.bfloat16)

class InventoryRequest(BaseModel):
    lead_time_days: int = 7
    current_inventory_units: int = 0      # <-- Now in UNITS
    safety_factor: float = 1.2

class InventoryRecommendation(BaseModel):
    date: str
    forecast_units: int
    reorder_point_units: int
    suggested_order_units: int
    risk_of_stockout: str

class InventoryResponse(BaseModel):
    item: str
    avg_price_per_unit: float
    lead_time_days: int
    recommendations: list[InventoryRecommendation]

@app.post("/inventory-plan", response_model=InventoryResponse)
def inventory_plan(req: InventoryRequest):
    context = torch.tensor(daily_units.values, dtype=torch.float32)
    
    # Forecast UNITS (not dollars)
    quantiles, _ = pipeline.predict_quantiles(
        context, 
        prediction_length=req.lead_time_days, 
        quantile_levels=[0.1, 0.5, 0.9]
    )
    
    median_forecast_units = quantiles[0, :, 1].numpy()     # P50 units
    high_forecast_units = quantiles[0, :, 2].numpy()       # P90 units
    
    # Lead time total demand in UNITS
    lead_time_median_demand = median_forecast_units.sum()
    lead_time_safety_demand = (high_forecast_units.sum() - lead_time_median_demand) * req.safety_factor
    
    reorder_point_units = int(lead_time_median_demand + lead_time_safety_demand)
    suggested_order_units = max(0, reorder_point_units - req.current_inventory_units)
    
    last_date = daily_units.index[-1]
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=req.lead_time_days, freq="D")
    
    points = []
    for i, (date, med_units, high_units) in enumerate(zip(future_dates, median_forecast_units, high_forecast_units)):
        days_to_stockout = req.current_inventory_units / med_units if med_units > 0 else 999
        stockout_risk = "HIGH" if days_to_stockout < req.lead_time_days * 0.8 else "MEDIUM" if days_to_stockout < req.lead_time_days else "LOW"
        points.append(InventoryRecommendation(
            date=str(date.date()),
            forecast_units=int(round(med_units)),
            reorder_point_units=reorder_point_units,
            suggested_order_units=suggested_order_units,
            risk_of_stockout=stockout_risk,
        ))
    
    return InventoryResponse(
        item="Brand Total",
        avg_price_per_unit=float(avg_price_per_unit),
        lead_time_days=req.lead_time_days,
        recommendations=points,
    )
