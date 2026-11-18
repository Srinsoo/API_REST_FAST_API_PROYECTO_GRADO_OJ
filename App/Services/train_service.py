import pandas as pd
from App.Ml.preprocessing import preprocess_data
from App.Ml.trainNew_ModelXGBoost import train_modelXGBoost



async def train_from_excel(file):
    df = pd.read_excel(file.file)
    train = train_modelXGBoost(df)
    print(train)
    
    return {
        "train"
    }