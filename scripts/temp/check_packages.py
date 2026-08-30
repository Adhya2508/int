import sys

packages = ['sklearn', 'lightgbm', 'xgboost', 'catboost', 'shap', 'matplotlib', 'joblib']
for p in packages:
    try:
        __import__(p)
        print(f"  {p}: INSTALLED")
    except ImportError:
        print(f"  {p}: NOT INSTALLED")
