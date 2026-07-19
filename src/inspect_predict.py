import inspect
import xgboost as xgb
import sys

print("Python Version:", sys.version)

try:
    cls = xgb.XGBClassifier
    if hasattr(cls, "predict"):
        print("XGBClassifier.predict source code:")
        print(inspect.getsource(cls.predict))
    else:
        print("XGBClassifier does not have 'predict' method.")
except Exception as e:
    import traceback
    traceback.print_exc()
