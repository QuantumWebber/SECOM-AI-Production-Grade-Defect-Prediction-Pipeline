from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="SemiGuard - Defect Detection API")

# Model load karo
model = joblib.load('../models/rf_model.pkl')

class SensorData(BaseModel):
    features: list[float]

@app.get("/")
def home():
    return {"message": "SemiGuard API is running!"}

@app.post("/predict")
def predict(data: SensorData):
    X = np.array(data.features).reshape(1, -1)
    
    proba = model.predict_proba(X)[0][1]
    prediction = "FAIL" if proba >= 0.343 else "PASS"
    
    return {
        "prediction": prediction,
        "fail_probability": round(float(proba), 4)
    }