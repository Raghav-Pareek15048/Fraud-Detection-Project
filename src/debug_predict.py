import sys
import os
# Add root path to PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import load_artifacts, predict_single
import traceback

def debug_prediction():
    try:
        model, scalers = load_artifacts()
        print("✅ SUCCESS: Model and scalers loaded.")
        
        sample = {
            "Time": 3600.0,
            "Amount": 45.50,
            **{f"V{i}": 0.01 for i in range(1, 29)}
        }
        
        print("\nAttempting single prediction...")
        res = predict_single(sample, model, scalers)
        print("✅ SUCCESS: Prediction completed successfully!")
        print("Result:", res)
        
    except Exception as e:
        print("❌ FAILED: Error during prediction pipeline:")
        traceback.print_exc()

if __name__ == "__main__":
    debug_prediction()
