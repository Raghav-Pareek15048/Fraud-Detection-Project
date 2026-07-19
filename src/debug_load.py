import joblib
import traceback
import sys

print("Python Version:", sys.version)

try:
    print("\nAttempting to load models/best_fraud_model.pkl...")
    model = joblib.load("models/best_fraud_model.pkl")
    print("✅ SUCCESS: loaded best_fraud_model.pkl successfully.")
    print("Model Type:", type(model))
except Exception as e:
    print("❌ FAILED: Error loading best_fraud_model.pkl:")
    traceback.print_exc()

try:
    print("\nAttempting to load models/scalers.pkl...")
    scalers = joblib.load("models/scalers.pkl")
    print("✅ SUCCESS: loaded scalers.pkl successfully.")
    print("Scalers:", scalers)
except Exception as e:
    print("❌ FAILED: Error loading scalers.pkl:")
    traceback.print_exc()
