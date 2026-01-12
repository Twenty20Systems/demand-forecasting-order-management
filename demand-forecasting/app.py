import pandas as pd
import torch
import json
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

# Daily total units across all items (all regions)
daily_units = (
    df.groupby("Date Purchase")["Purchased Quantity"]
     .sum()
     .sort_index()
)

# Daily units per item (e.g. "shirt", "jeans") - all regions
item_series: dict[str, pd.Series] = {}
for item_name, g in df.groupby("Item Purchased"):
    s = (
        g.groupby("Date Purchase")["Purchased Quantity"]
         .sum()
         .sort_index()
    )
    item_series[item_name] = s

# Daily units per region (all items)
region_series: dict[str, pd.Series] = {}
for region_name, g in df.groupby("Region"):
    s = (
        g.groupby("Date Purchase")["Purchased Quantity"]
         .sum()
         .sort_index()
    )
    region_series[region_name] = s

# Daily units per item per region
item_region_series: dict[tuple[str, str], pd.Series] = {}
for (item_name, region_name), g in df.groupby(["Item Purchased", "Region"]):
    s = (
        g.groupby("Date Purchase")["Purchased Quantity"]
         .sum()
         .sort_index()
    )
    item_region_series[(item_name, region_name)] = s

# -----------------------------
# Load Chronos model
# -----------------------------

pipeline = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",
    device_map="cpu",
    dtype=torch.bfloat16,
)

# -----------------------------
# Pydantic models - NO MISLEADING DEFAULTS
# -----------------------------

class InventoryRequest(BaseModel):
    lead_time_days: int           # REQUIRED
    current_inventory_units: int  # REQUIRED
    safety_factor: float = 1.2
    item: Optional[str] = None
    region: Optional[str] = None  # "East Coast" or "West Coast"

class DailyForecast(BaseModel):
    date: str
    forecast_units: int
    risk_of_stockout: str

class InventoryResponse(BaseModel):
    item: str
    region: Optional[str] = None
    lead_time_days: int
    reorder_point_units: int
    suggested_order_units: int
    recommendations: List[DailyForecast]

# -----------------------------
# Main API endpoint
# -----------------------------

@app.post("/inventory-plan", response_model=InventoryResponse)
def inventory_plan(req: InventoryRequest):
    # Validate region if provided
    if req.region:
        valid_regions = ["East Coast", "West Coast"]
        if req.region not in valid_regions:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid region: {req.region}. Must be one of: {', '.join(valid_regions)}"
            )

    # Choose which time series to forecast based on item and region
    if req.item and req.region:
        # Specific item in specific region
        key = (req.item, req.region)
        if key not in item_region_series:
            raise HTTPException(
                status_code=400,
                detail=f"No data found for item '{req.item}' in region '{req.region}'"
            )
        series = item_region_series[key]
        item_name = req.item
        region_name = req.region
    elif req.item:
        # Specific item across all regions
        if req.item not in item_series:
            raise HTTPException(status_code=400, detail=f"Unknown item: {req.item}")
        series = item_series[req.item]
        item_name = req.item
        region_name = None
    elif req.region:
        # All items in specific region
        if req.region not in region_series:
            raise HTTPException(status_code=400, detail=f"Unknown region: {req.region}")
        series = region_series[req.region]
        item_name = "All Items"
        region_name = req.region
    else:
        # All items across all regions
        series = daily_units
        item_name = "All Items"
        region_name = None

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
        region=region_name,
        lead_time_days=req.lead_time_days,
        reorder_point_units=reorder_point_units,
        suggested_order_units=suggested_order_units,
        recommendations=recommendations,
    )

# -----------------------------
# Root proxy for API Gateway - FIXED FOR WORKATO
# -----------------------------

@app.post("/", response_model=InventoryResponse)
async def inventory_plan_root_proxy(request: Request):
    raw_body = await request.body()
    text = raw_body.decode("utf-8")
    print("=== RAW BODY FROM GATEWAY ===")
    print(repr(text))
    print("=== END RAW BODY ===")

    # **FIX: Repair common JSON errors (unquoted strings)**
    text = text.replace('"item": jeans', '"item": "jeans"')
    text = text.replace('"item": shirt', '"item": "shirt"')
    text = text.replace('"region": East Coast', '"region": "East Coast"')
    text = text.replace('"region": West Coast', '"region": "West Coast"')
    print("=== REPAIRED JSON ===")
    print(repr(text))

    try:
        body = json.loads(text)
        print("PARSED BODY:", body)

        # Handle direct JSON payload (most common now)
        if isinstance(body, dict) and "lead_time_days" in body:
            print("Direct JSON payload detected")
            final_body = body
        else:
            final_body = body

        print("FINAL BODY FOR Pydantic:", final_body)

    except json.JSONDecodeError as e:
        print("JSON PARSE ERROR:", str(e))
        raise HTTPException(status_code=400, detail="Invalid JSON body - check quotes around item name")

    # Validate required fields
    missing = []
    if "lead_time_days" not in final_body:
        missing.append("lead_time_days")
    if "current_inventory_units" not in final_body:
        missing.append("current_inventory_units")

    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required fields: {', '.join(missing)}")

    try:
        req = InventoryRequest(**final_body)
        print("SUCCESSFUL Pydantic parse:", req.dict())
    except Exception as e:
        print("PYDANTIC ERROR:", str(e))
        raise HTTPException(status_code=400, detail=f"Validation error: {str(e)}")

    return inventory_plan(req)
