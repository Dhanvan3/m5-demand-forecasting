import mlflow.xgboost
import os

print("Loading model from MLflow registry...")
model = mlflow.xgboost.load_model("models:/m5-xgboost-forecaster/2")

os.makedirs("api/model", exist_ok=True)
model.save_model("api/model/xgb_model.json")
print("Model saved to api/model/xgb_model.json")
