from fastapi import APIRouter, UploadFile, File, Form
from App.Services.prediction_service import predict_from_excel
from App.Services.train_service import train_from_excel



router = APIRouter(prefix="/prediction", tags=["Prediction"])


@router.post("/")
async def prediction_predict(file: UploadFile = File(...),
    latitude: str = Form(...),
    longitude: str = Form(...),
):
    
    result = await predict_from_excel(file, latitude, longitude)

    return result



@router.post("/Train")
async def prediction_train(file: UploadFile = File(...),
    latitude: str = Form(...),
    longitude: str = Form(...),
):
    print("entro a la ruta")
    result = await train_from_excel(file)

    return result