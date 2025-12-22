import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from chronos import BaseChronosPipeline
from typing import Optional, List


app = FastAPI(
    title="Fashion Inventory Planner",
    description="Chronos-based demand forecasting for shirts and jeans",
    version="1.0.0",
)

# -----------------------------
# Load data and build time series
# -----------------------------

df = pd.read_csv("data/Fashion_Retail_Sales.csv")
df["Date Purchase"] = pd.to_datetime(df["Date Purchase"])

# Daily total units across all items
daily_units = (
    df.groupby("Date Purchase")["Purchased Quantity"]
      .sum()
      .sort_index()
)

# Daily units per item (e.g. "shirt", "jeans")
item_series: dict[str, pd.Series] = {}
for item_name, g in df.groupby("Item Purchased"):
    s = (
        g.groupby("Date Purchase")["Purchased Quantity"]
         .sum()
         .sort_index()
    )
    item_series[item_name] = s

# -----------------------------
# Load Chronos model
# -----------------------------

pipeline = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",
    device_map="cpu",
    dtype=torch.bfloat16,
)

# -----------------------------
# Pydantic models
# -----------------------------

class InventoryRequest(BaseModel):
    lead_time_days: int = 7
    current_inventory_units: int = 0
    safety_factor: float = 1.2
    item: Optional[str] = None  # "shirt", "jeans", or None for all


class DailyForecast(BaseModel):
    date: str
    forecast_units: int
    risk_of_stockout: str


class InventoryResponse(BaseModel):
    item: str
    lead_time_days: int
    reorder_point_units: int
    suggested_order_units: int
    recommendations: List[DailyForecast]

# -----------------------------
# Main API endpoint
# -----------------------------

@app.post("/inventory-plan", response_model=InventoryResponse)
def inventory_plan(req: InventoryRequest):
    # Choose which time series to forecast
    if req.item:
        if req.item not in item_series:
            raise HTTPException(status_code=400, detail=f"Unknown item: {req.item}")
        series = item_series[req.item]
        item_name = req.item
    else:
        series = daily_units
        item_name = "All Items"

    if len(series) < 3:
        raise HTTPException(status_code=400, detail="Not enough history to forecast")

    # Chronos expects a tensor context of the time series
    context = torch.tensor(series.values, dtype=torch.float32)

    quantiles, _ = pipeline.predict_quantiles(
        context,
        prediction_length=req.lead_time_days,
        quantile_levels=[0.1, 0.5, 0.9],
    )

    median_forecast_units = quantiles[0, :, 1].numpy()  # P50
    high_forecast_units = quantiles[0, :, 2].numpy()    # P90

    # Totals over the lead time
    lead_time_median_demand = float(median_forecast_units.sum())
    lead_time_safety_demand = float(
        (high_forecast_units.sum() - lead_time_median_demand) * req.safety_factor
    )

    reorder_point_units = int(round(lead_time_median_demand + lead_time_safety_demand))
    suggested_order_units = max(0, reorder_point_units - req.current_inventory_units)

    # Build daily recommendations
    last_date = series.index[-1]
    future_dates = pd.date_range(
        last_date + pd.Timedelta(days=1),
        periods=req.lead_time_days,
        freq="D",
    )

    recommendations: List[DailyForecast] = []
    for date, med_units in zip(future_dates, median_forecast_units):
        med_units = float(med_units)
        days_to_stockout = (
            req.current_inventory_units / med_units if med_units > 0 else 999.0
        )
        if days_to_stockout < req.lead_time_days * 0.8:
            stockout_risk = "HIGH"
        elif days_to_stockout < req.lead_time_days:
            stockout_risk = "MEDIUM"
        else:
            stockout_risk = "LOW"

        recommendations.append(
            DailyForecast(
                date=str(date.date()),
                forecast_units=int(round(med_units)),
                risk_of_stockout=stockout_risk,
            )
        )

    return InventoryResponse(
        item=item_name,
        lead_time_days=req.lead_time_days,
        reorder_point_units=reorder_point_units,
        suggested_order_units=suggested_order_units,
        recommendations=recommendations,
    )

# -----------------------------
# Root proxy for API Gateway
# -----------------------------

@app.post("/", response_model=InventoryResponse)
async def inventory_plan_root_proxy(request: Request):
    body = await request.json()
    req = InventoryRequest(**body)
    return inventory_plan(req)
