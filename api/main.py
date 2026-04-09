from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import xgboost as xgb
import pandas as pd
import numpy as np
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

class ForecastRequest(BaseModel):
    item_id: str
    lag_7: float
    lag_14: float
    lag_28: float
    rolling_mean_7: float
    rolling_mean_28: float
    rolling_std_7: float
    rolling_std_28: float
    day_of_week: int
    day_of_month: int
    week_of_year: int
    month: int
    year: int
    is_weekend: int
    has_event: int
    snap_flag: int
    sell_price: float
    price_change: float

class ForecastResponse(BaseModel):
    item_id: str
    predicted_sales: float
    model: str = "xgboost-v2"

models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading model from file...")
    model = xgb.XGBRegressor()
    model.load_model("api/model/xgb_model.json")
    models["xgb"] = model
    print("Model loaded!")
    yield
    models.clear()

app = FastAPI(
    title="M5 Demand Forecaster",
    description="XGBoost demand forecasting API for Walmart M5 dataset",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": "xgb" in models
    }

@app.post("/predict", response_model=ForecastResponse)
def predict(req: ForecastRequest):
    if "xgb" not in models:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        X = pd.DataFrame([req.dict()])[FEATURES]
        X = X.fillna(0)
        pred = float(models["xgb"].predict(X)[0])
        pred = max(0.0, round(pred, 2))
        return ForecastResponse(
            item_id=req.item_id,
            predicted_sales=pred
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
def predict_batch(requests: list[ForecastRequest]):
    if "xgb" not in models:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        rows = [r.dict() for r in requests]
        X = pd.DataFrame(rows)[FEATURES].fillna(0)
        preds = models["xgb"].predict(X)
        preds = np.maximum(preds, 0)
        return [
            {"item_id": r.item_id, "predicted_sales": round(float(p), 2)}
            for r, p in zip(requests, preds)
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
