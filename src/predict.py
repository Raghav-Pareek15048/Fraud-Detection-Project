import os
import joblib
import pandas as pd
from typing import Dict, Any, Tuple
from src.preprocessing import preprocess_inference_data

# Default paths relative to project root
DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
MODEL_NAMES = ["best_fraud_model.pkl", "fraud_detection_model.pkl"]
SCALER_NAMES = ["scalers.pkl", "scaler.pkl"]

def load_artifacts(model_dir: str = DEFAULT_MODEL_DIR) -> Tuple[Any, Dict[str, Any]]:
    """
    Loads model and scaler artifacts from the specified directory.
    Attempts to search for multiple naming conventions to ensure robustness.
    
    Parameters:
    - model_dir: Path to directory containing models and scalers.
    
    Returns:
    - Tuple containing (loaded_model, scalers_dict)
    """
    model = None
    scalers = None
    
    # Try to load model
    model_loaded = False
    for name in MODEL_NAMES:
        path = os.path.join(model_dir, name)
        if os.path.exists(path):
            try:
                model = joblib.load(path)
                model_loaded = True
                print(f"Loaded model from: {path}")
                break
            except Exception as e:
                print(f"Failed to load model from {path}: {e}")
                
    if not model_loaded:
        # Check root folder just in case
        for name in MODEL_NAMES:
            if os.path.exists(name):
                try:
                    model = joblib.load(name)
                    model_loaded = True
                    print(f"Loaded model from root: {name}")
                    break
                except Exception as e:
                    pass
                    
    if not model_loaded:
        raise FileNotFoundError("Could not find or load the fraud detection model pkl file.")

    # Patch the loaded model dynamically if it's an XGBoost model
    # (Fixes AttributeError: 'XGBClassifier' object has no attribute 'use_label_encoder' / 'gpu_id' / 'predictor')
    if model is not None:
        # 1. Sync all constructor parameters with the active version's defaults
        try:
            import xgboost as xgb
            if isinstance(model, xgb.XGBModel):
                default_model = xgb.XGBClassifier()
                for param, val in default_model.get_params().items():
                    if getattr(model, param, None) is None:
                        setattr(model, param, val)
        except Exception as e:
            print(f"Warning: Failed to sync parameters with default XGBClassifier: {e}")
                
        # 2. Set key operational attributes expected by predict()
        if getattr(model, "best_iteration", None) is None:
            try:
                model.best_iteration = 0
            except Exception:
                pass

        # 3. Dynamic subclass fallback for any other lookup
        try:
            class PatchedModel(model.__class__):
                def __getattr__(self, name):
                    if name.startswith("_"):
                        raise AttributeError(name)
                    if name == "best_iteration":
                        return 0
                    return None
            model.__class__ = PatchedModel
            print("Successfully patched model class for backward compatibility.")
        except Exception as e:
            print(f"Dynamic class patch failed: {e}")

    # Try to load scalers
    scalers_loaded = False
    for name in SCALER_NAMES:
        path = os.path.join(model_dir, name)
        if os.path.exists(path):
            try:
                scalers = joblib.load(path)
                scalers_loaded = True
                print(f"Loaded scalers from: {path}")
                break
            except Exception as e:
                print(f"Failed to load scalers from {path}: {e}")
                
    if not scalers_loaded:
        for name in SCALER_NAMES:
            if os.path.exists(name):
                try:
                    scalers = joblib.load(name)
                    scalers_loaded = True
                    print(f"Loaded scalers from root: {name}")
                    break
                except Exception as e:
                    pass

    if not scalers_loaded:
        raise FileNotFoundError("Could not find or load the scalers pkl file.")
        
    return model, scalers

def predict_single(transaction: Dict[str, float], model: Any, scalers: Dict[str, Any]) -> Dict[str, Any]:
    """
    Runs prediction for a single transaction dictionary.
    
    Parameters:
    - transaction: Dict containing keys Time, Amount, V1-V28.
    - model: Loaded scikit-learn or XGBoost model.
    - scalers: Dict containing loaded StandardScaler instances.
    
    Returns:
    - Dict with prediction labels and probability.
    """
    df = pd.DataFrame([transaction])
    processed_df = preprocess_inference_data(df, scalers)
    
    # Run prediction
    prediction = int(model.predict(processed_df)[0])
    probability = float(model.predict_proba(processed_df)[0, 1])
    
    label = "Fraudulent" if prediction == 1 else "Legitimate"
    confidence = probability if prediction == 1 else (1.0 - probability)
    
    return {
        "is_fraud": prediction == 1,
        "label": label,
        "confidence": round(confidence, 4),
        "probability": round(probability, 4)
    }

def predict_batch(transactions: pd.DataFrame, model: Any, scalers: Dict[str, Any]) -> pd.DataFrame:
    """
    Runs prediction on a batch of transactions.
    
    Parameters:
    - transactions: pd.DataFrame with transaction records.
    - model: Loaded model.
    - scalers: Loaded scalers.
    
    Returns:
    - pd.DataFrame containing the inputs alongside prediction labels and probabilities.
    """
    processed_df = preprocess_inference_data(transactions, scalers)
    
    # Generate predictions
    predictions = model.predict(processed_df)
    probabilities = model.predict_proba(processed_df)[:, 1]
    
    results = transactions.copy()
    results["is_fraud"] = predictions == 1
    results["label"] = ["Fraudulent" if p == 1 else "Legitimate" for p in predictions]
    results["probability"] = probabilities.round(4)
    results["confidence"] = [
        prob if pred == 1 else (1.0 - prob)
        for pred, prob in zip(predictions, probabilities)
    ]
    
    return results
