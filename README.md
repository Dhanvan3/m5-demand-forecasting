# M5 Demand Forecasting

End-to-end demand forecasting system built on the Walmart M5 dataset.

## Stack
- XGBoost + Prophet ensemble (22.5% improvement over baseline)
- Walk-forward cross validation (time-series safe)
- MLflow experiment tracking + model registry
- FastAPI REST API with batch prediction support

## Results
| Model    | MAPE  | Accuracy |
|----------|-------|----------|
| XGBoost  | 80.8% | ~44%     |
| Prophet  | 100%  | —        |
| Ensemble | 62.6% | ~37%     |

## Run locally
```bash
python3 -m venv m5-env && source m5-env/bin/activate
pip install -r requirements.txt
python3 src/prepare_data.py
python3 src/features.py
python3 src/train_xgboost.py
uvicorn api.main:app --port 8080
```

## API
- GET  /health       — model status
- POST /predict      — single item forecast
- POST /predict/batch — bulk forecast
# m5-demand-forecasting
