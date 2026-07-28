from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from api.model_manager import load_models


@asynccontextmanager
async def lifespan(app: FastAPI):

    load_models()

    print("All the Models have been Loaded")

    yield

    print("Shutting Down")


app = FastAPI(
    title="Pneumonia Classifier API",
    description="API for prediction of Xrays for Pneumonia using CNN and Transfer Learning",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://pneumonia-classifier.vercel.app/"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
