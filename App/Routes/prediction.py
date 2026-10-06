from fastapi import APIRouter, UploadFile, File
from app.services.prediction_service import predict_from_excel

router = APIRouter(prefix="/prediction", tags=["Prediction"])


@router.post("/")
async def prediction(file: UploadFile = File(...)):

    result = await predict_from_excel(file)

    return result