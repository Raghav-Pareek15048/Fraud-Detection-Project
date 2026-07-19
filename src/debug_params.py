import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import load_artifacts
import joblib

def debug_params():
    try:
        model, scalers = load_artifacts()
        print("\nModel class:", model.__class__)
        print("Model class bases:", model.__class__.__bases__)
        
        # Check _get_param_names
        if hasattr(model, "_get_param_names"):
            names = model._get_param_names()
            print("\n_get_param_names contains 'predictor':", "predictor" in names)
            print("All params:", names)
        
        # Check setting attribute
        print("\nSetting model.predictor = None manually...")
        try:
            model.predictor = None
            print("Successfully set model.predictor = None via direct assignment.")
            print("hasattr(model, 'predictor'):", hasattr(model, 'predictor'))
            print("getattr(model, 'predictor'):", getattr(model, 'predictor', 'Attribute is missing'))
        except Exception as e:
            print("FAILED to set model.predictor directly:", e)
            
        print("\nChecking get_params()...")
        try:
            params = model.get_params()
            print("get_params() succeeded! Keys:", list(params.keys()))
        except Exception as e:
            print("get_params() FAILED:", e)
            
    except Exception as e:
        print("Error during loading:", e)

if __name__ == "__main__":
    debug_params()
