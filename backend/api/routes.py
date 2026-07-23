from fastapi import APIRouter

router = APIRouter()


@router.get("/")
def root():
    return {"message": "Pneumonia Classifier API"}


@router.get("/health")
def health():
    return {"status": "Healthy 😊"}
