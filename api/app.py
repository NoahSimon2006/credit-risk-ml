from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

app = FastAPI()

#load model and scaler (only once on startup)
model = joblib.load("models/xgboost.pkl")
scaler = joblib.load("models/scaler.pkl")

#define the input data shape using Pydantic
class LoanApplication(BaseModel):
    features: list  # Expecting list of numerical values

@app.get("/")
def home():
    return {"message": "Credit Risk API is Live"}

@app.post("/predict")
def predict(data: LoanApplication):
    #convert input list to DataFrame/Array
    features_arr = np.array(data.features).reshape(1, -1)
    
    #scale the data (using the same scaler from training)
    features_scaled = scaler.transform(features_arr)
    
    #predict Probability
    probability = model.predict_proba(features_scaled)[0][1]
    prediction = int(probability > 0.5)
    
    return {
        "default_probability": float(probability),
        "risk_class": prediction
    }