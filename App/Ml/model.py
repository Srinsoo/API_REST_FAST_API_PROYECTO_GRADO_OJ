import joblib

MODEL_PATH = "models/model.pkl"


def load_model():

    return joblib.load(MODEL_PATH)