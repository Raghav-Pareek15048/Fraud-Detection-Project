import sys
import os
# Add root path to PYTHONPATH to resolve src imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import load_artifacts, predict_single

def test_pipeline():
    print("=== Testing Fraud Detection Pipeline locally ===")
    
    # 1. Test model artifact loading
    try:
        model, scalers = load_artifacts()
        print("✅ SUCCESS: Model and scalers loaded successfully.")
    except Exception as e:
        print(f"❌ ERROR: Failed to load model artifacts: {e}")
        print("Ensure 'best_fraud_model.pkl' and 'scalers.pkl' are placed inside the 'models/' folder.")
        return
        
    # 2. Test single transaction inference
    try:
        # Create a typical non-suspicious transaction (V features close to 0)
        legit_tx = {
            "Time": 3600.0,
            "Amount": 45.50,
            **{f"V{i}": 0.01 for i in range(1, 29)}
        }
        
        # Create a typical suspicious transaction (large negative V features)
        fraud_tx = {
            "Time": 3600.0,
            "Amount": 999.00,
            **{f"V{i}": -2.5 for i in range(1, 29)}
        }
        
        result_legit = predict_single(legit_tx, model, scalers)
        result_fraud = predict_single(fraud_tx, model, scalers)
        
        print("\n✅ SUCCESS: Inferences executed.")
        print(f"Legit Sample Prediction: {result_legit['label']} (Confidence: {result_legit['confidence'] * 100}%)")
        print(f"Fraud Sample Prediction: {result_fraud['label']} (Confidence: {result_fraud['confidence'] * 100}%)")
        
    except Exception as e:
        print(f"❌ ERROR: Inference pipeline failed: {e}")

if __name__ == "__main__":
    test_pipeline()
