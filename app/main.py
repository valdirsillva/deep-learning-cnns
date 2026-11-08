from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.endpoints import predict_image, router as predict_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Na produção, coloque a URL do seu frontend
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router)