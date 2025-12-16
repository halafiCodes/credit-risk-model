import os
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, conlist
import numpy as np

# -----------------------------
# Pydantic models
# -----------------------------
class CustomerData(BaseModel):
    Amount: float
    Value: float
    hour: int
    day: int
    month: int
    year: int
    total_amount: float
    avg_amount: float
    transaction_count: int
    std_amount: float
    Amount_log: float
    total_amount_log: float
    avg_amount_log: float
    std_amount_log: float

class PredictionResponse(BaseModel):
    risk_probability: float

# -----------------------------
# App initialization
# -----------------------------
app = FastAPI(title="Credit Risk Prediction API")

# -----------------------------
# Load model and pipeline
# -----------------------------
BASE_DIR = os.path.dirname(__file__)  # src/api/
MODEL_PATH = os.path.join(BASE_DIR, "../../models/best_model.pkl")
PIPELINE_PATH = os.path.join(BASE_DIR, "../../models/feature_pipeline.pkl")

model = joblib.load(MODEL_PATH)
feature_pipeline = joblib.load(PIPELINE_PATH)

# -----------------------------
# Endpoints
# -----------------------------
@app.get("/")
def home():
    return {"message": "Credit Risk Prediction API is running."}

@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerData):
    # Convert to DataFrame
    df = pd.DataFrame([customer.dict()])

    # Transform features using the pipeline
    X_transformed = feature_pipeline.transform(df)

    # Predict probability
    risk_prob = model.predict_proba(X_transformed)[:, 1][0]  # probability of class 1

    return {"risk_probability": risk_prob}
