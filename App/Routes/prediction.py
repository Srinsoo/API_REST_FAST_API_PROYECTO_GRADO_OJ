from fastapi import APIRouter, UploadFile, File, Form
from App.Services.prediction_service import predict_from_excel

router = APIRouter(prefix="/prediction", tags=["Prediction"])


@router.post("/")
async def prediction(file: UploadFile = File(...),
    latitude: str = Form(...),
    longitude: str = Form(...),
):
    
    result = await predict_from_excel(file, latitude, longitude)

    return result