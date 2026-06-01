"""FastAPI service that serves churn predictions from the trained pipeline."""

from __future__ import annotations

import os

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .data import FEATURES

MODEL_PATH = os.environ.get("MODEL_PATH", "models/churn_pipeline.joblib")

app = FastAPI(title="Churn Prediction API", version="0.1.0")
_model = {"pipe": None}


def _load():
    if _model["pipe"] is None:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(503, f"Model not found at {MODEL_PATH}. Train it first.")
        _model["pipe"] = joblib.load(MODEL_PATH)
    return _model["pipe"]


class Customer(BaseModel):
    """One customer record (Telco Customer Churn schema)."""
    gender: str = "Female"
    SeniorCitizen: int = 0
    Partner: str = "Yes"
    Dependents: str = "No"
    tenure: int = Field(1, ge=0, le=100)
    PhoneService: str = "Yes"
    MultipleLines: str = "No"
    InternetService: str = "Fiber optic"
    OnlineSecurity: str = "No"
    OnlineBackup: str = "No"
    DeviceProtection: str = "No"
    TechSupport: str = "No"
    StreamingTV: str = "No"
    StreamingMovies: str = "No"
    Contract: str = "Month-to-month"
    PaperlessBilling: str = "Yes"
    PaymentMethod: str = "Electronic check"
    MonthlyCharges: float = 70.0
    TotalCharges: float = 70.0


class Prediction(BaseModel):
    churn_probability: float
    churn: bool


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": os.path.exists(MODEL_PATH)}


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer):
    pipe = _load()
    row = pd.DataFrame([customer.model_dump()])[FEATURES]
    proba = float(pipe.predict_proba(row)[0, 1])
    return Prediction(churn_probability=round(proba, 4), churn=proba >= 0.5)
