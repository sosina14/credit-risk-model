import pickle
import os
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from src.api.pydantic_models import CustomerFeatures, PredictionResponse

app = FastAPI(
    title="Bati Bank Credit Risk API",
    description="Predicts credit risk probability for BNPL applicants",
    version="1.0.0"
)

MODEL_PATH = os.getenv("MODEL_PATH", "models/best_model.pkl")
model = None

@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
    else:
        raise RuntimeError(f"Model not found: {MODEL_PATH}")

@app.get("/")
def root():
    return {"message": "Bati Bank Credit Risk API", "status": "running"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict(features: CustomerFeatures, customer_id: str = "unknown"):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        data = pd.DataFrame([features.model_dump()])
        prob = model.predict_proba(data)[0][1]
        label = int(prob >= 0.5)
        risk_label = "HIGH RISK" if label == 1 else "LOW RISK"
        return PredictionResponse(
            customer_id=customer_id,
            risk_probability=round(float(prob), 4),
            is_high_risk=label,
            risk_label=risk_label
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))