
import xgboost as xgb
from pathlib import Path

MODELS_DIR = Path("Models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH= MODELS_DIR / "Modelo_XGboost.json"


def load_model():
      model = xgb.XGBRegressor()
      model.load_model(str(MODEL_PATH))
      return model