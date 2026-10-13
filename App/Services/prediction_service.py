import pandas as pd
from app.Ml.preprocessing import preprocess_data
from app.Ml.model import load_model

model = load_model()

async def predict_from_excel(file, lat, long):

    df = pd.read_excel(file.file)

    df = preprocess_data(df,lat,long)



    predictions = model.predict(df)

    return {
        "predictions": predictions.tolist()
    }