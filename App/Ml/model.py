import joblib
import xgboost as xgb
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "modelo_xgboost.json"


def load_model():
    model = xgb.XGBRegressor()
    model.load_model(str(MODEL_PATH))
    return model