from fastapi import FastAPI
from App.Routes.prediction import router as prediction_router
from fastapi.middleware.cors import CORSMiddleware



app = FastAPI()

@app.get("/")
def read_root():
    return {"status": "online", "message": "API de Predicción activa"}


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(prediction_router)