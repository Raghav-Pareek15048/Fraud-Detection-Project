import io
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any

from src.predict import load_artifacts, predict_single, predict_batch

app = FastAPI(
    title="Financial Transaction Fraud Detection API",
    description="API for detecting fraudulent financial transactions in real-time.",
    version="1.0.0"
)

# Enable CORS for frontend UI connections
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and scalers
model = None
scalers = None

# Attempt to load model artifacts on startup
@app.on_event("startup")
def startup_event():
    global model, scalers
    try:
        model, scalers = load_artifacts()
        print("Model and scalers loaded successfully on startup.")
    except Exception as e:
        print(f"Warning: Failed to load model artifacts on startup: {e}")

# Pydantic schema for individual transaction details
class Transaction(BaseModel):
    Time: float = Field(0.0, description="Seconds elapsed since the first transaction in the dataset")
    Amount: float = Field(0.0, description="Transaction amount")
    V1: float = Field(0.0, description="PCA transformed feature V1")
    V2: float = Field(0.0, description="PCA transformed feature V2")
    V3: float = Field(0.0, description="PCA transformed feature V3")
    V4: float = Field(0.0, description="PCA transformed feature V4")
    V5: float = Field(0.0, description="PCA transformed feature V5")
    V6: float = Field(0.0, description="PCA transformed feature V6")
    V7: float = Field(0.0, description="PCA transformed feature V7")
    V8: float = Field(0.0, description="PCA transformed feature V8")
    V9: float = Field(0.0, description="PCA transformed feature V9")
    V10: float = Field(0.0, description="PCA transformed feature V10")
    V11: float = Field(0.0, description="PCA transformed feature V11")
    V12: float = Field(0.0, description="PCA transformed feature V12")
    V13: float = Field(0.0, description="PCA transformed feature V13")
    V14: float = Field(0.0, description="PCA transformed feature V14")
    V15: float = Field(0.0, description="PCA transformed feature V15")
    V16: float = Field(0.0, description="PCA transformed feature V16")
    V17: float = Field(0.0, description="PCA transformed feature V17")
    V18: float = Field(0.0, description="PCA transformed feature V18")
    V19: float = Field(0.0, description="PCA transformed feature V19")
    V20: float = Field(0.0, description="PCA transformed feature V20")
    V21: float = Field(0.0, description="PCA transformed feature V21")
    V22: float = Field(0.0, description="PCA transformed feature V22")
    V23: float = Field(0.0, description="PCA transformed feature V23")
    V24: float = Field(0.0, description="PCA transformed feature V24")
    V25: float = Field(0.0, description="PCA transformed feature V25")
    V26: float = Field(0.0, description="PCA transformed feature V26")
    V27: float = Field(0.0, description="PCA transformed feature V27")
    V28: float = Field(0.0, description="PCA transformed feature V28")

    class Config:
        schema_extra = {
            "example": {
                "Time": 4092.0,
                "Amount": 99.99,
                "V1": -1.359807,
                "V2": -0.072781,
                "V3": 2.536347,
                "V4": 1.378155,
                "V5": -0.338321,
                "V6": 0.462388,
                "V7": 0.239599,
                "V8": 0.098698,
                "V9": 0.363787,
                "V10": 0.090794,
                "V11": -0.551600,
                "V12": -0.617801,
                "V13": -0.991390,
                "V14": -0.311169,
                "V15": 1.468177,
                "V16": -0.470401,
                "V17": 0.207971,
                "V18": 0.025791,
                "V19": 0.403993,
                "V20": 0.251412,
                "V21": -0.018307,
                "V22": 0.277838,
                "V23": -0.110474,
                "V24": 0.066928,
                "V25": 0.128539,
                "V26": -0.189115,
                "V27": 0.133558,
                "V28": -0.021053
            }
        }

@app.get("/health", summary="Health check endpoint")
def health_check():
    """Checks the health of the API and model readiness."""
    global model, scalers
    if model is None or scalers is None:
        try:
            model, scalers = load_artifacts()
        except Exception as e:
            return {
                "status": "warning",
                "message": f"API is active but model artifacts could not be loaded: {e}",
                "model_loaded": False
            }
            
    return {
        "status": "healthy",
        "message": "API is active and model is loaded.",
        "model_loaded": True
    }

@app.get("/metrics", summary="Model evaluation metrics")
def get_metrics():
    """Returns baseline performance metrics for training models (XGBoost, Random Forest, Logistic Regression)."""
    return {
        "best_model": "XGBoost",
        "models": [
            {
                "name": "XGBoost",
                "f1_score": 0.810256,
                "recall": 0.840426,
                "precision": 0.782178,
                "pr_auc": 0.857254,
                "is_best": True
            },
            {
                "name": "Random Forest",
                "f1_score": 0.597015,
                "recall": 0.851064,
                "precision": 0.459770,
                "pr_auc": 0.836475,
                "is_best": False
            },
            {
                "name": "Logistic Regression",
                "f1_score": 0.105072,
                "recall": 0.925532,
                "precision": 0.055698,
                "pr_auc": 0.755585,
                "is_best": False
            }
        ]
    }

@app.post("/predict", summary="Predict single transaction classification")
def predict(transaction: Transaction):
    """Predicts legitimacy for a single transaction input."""
    global model, scalers
    if model is None or scalers is None:
        try:
            model, scalers = load_artifacts()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Model loading error: {e}")
            
    try:
        tx_dict = transaction.dict()
        result = predict_single(tx_dict, model, scalers)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

@app.post("/predict/batch", summary="Predict batch of transactions via JSON")
def predict_batch_json(transactions: List[Transaction]):
    """Predicts legitimacy for a list of JSON-formatted transaction records."""
    global model, scalers
    if model is None or scalers is None:
        try:
            model, scalers = load_artifacts()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Model loading error: {e}")
            
    try:
        tx_list = [t.dict() for t in transactions]
        df = pd.DataFrame(tx_list)
        results_df = predict_batch(df, model, scalers)
        return results_df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {e}")

@app.post("/predict/batch/csv", summary="Predict batch of transactions via CSV file upload")
def predict_batch_csv(file: UploadFile = File(...)):
    """Predicts legitimacy for a batch of transactions uploaded as a CSV file."""
    global model, scalers
    if model is None or scalers is None:
        try:
            model, scalers = load_artifacts()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Model loading error: {e}")
            
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
        
    try:
        contents = file.file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Verify required columns (specifically Amount & Time)
        # Even if some PCA columns V1-V28 are missing, preprocessing.py will fill them with 0.0
        results_df = predict_batch(df, model, scalers)
        
        # Return records
        return results_df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing CSV upload: {e}")
    finally:
        file.file.close()
