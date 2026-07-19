import inspect
import xgboost as xgb
import sys

print("Python Version:", sys.version)

try:
    cls = xgb.XGBClassifier
    if hasattr(cls, "predictor"):
        prop = getattr(cls, "predictor")
        if isinstance(prop, property):
            print("Property source code:")
            if prop.fget:
                print(inspect.getsource(prop.fget))
            else:
                print("No getter found.")
        else:
            print("predictor is not a property, it is of type:", type(prop))
    else:
        print("XGBClassifier does not have 'predictor' attribute.")
except Exception as e:
    import traceback
    traceback.print_exc()
