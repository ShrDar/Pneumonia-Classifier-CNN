from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.routes import router
from api.model_manager import load_models
from src.config import OUTPUT_DIR


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading CNN Models")

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
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount(
    "/outputs",
    StaticFiles(directory=OUTPUT_DIR),
    name="outputs",
)

app.include_router(router)
