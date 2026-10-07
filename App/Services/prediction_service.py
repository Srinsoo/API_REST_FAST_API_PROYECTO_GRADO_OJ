import pandas as pd
from App.Ml.preprocessing import preprocess_data
from App.Ml.model import load_model


async def predict_from_excel(file):

    df = pd.read_excel(file.file)

    df = preprocess_data(df)

    # model = load_model()

    # predictions = model.predict(df)

    return {
        # "predictions": predictions.tolist()
        "Excel preprocesado"
    }