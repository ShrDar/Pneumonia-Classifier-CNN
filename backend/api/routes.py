from fastapi import (
    APIRouter,
    Depends,
    File,
    UploadFile,
    HTTPException,
)

from api.dependencies import get_model, get_target_layers

from api.schemas import PredictionResponse

from api.utils import save_uploaded_file, delete_file

from src.config import DEVICE, MODEL_TYPE, OUTPUT_DIR

from src.inference import predict_xray

from src.visualize import generate_gradcam


router = APIRouter()


@router.post(
    "/predict",
    response_model=PredictionResponse,
)
async def predict(
    file: UploadFile = File(...),
    model=Depends(get_model),
    target_layers=Depends(get_target_layers),
):

    if file.content_type not in [
        "image/jpeg",
        "image/png",
    ]:
        raise HTTPException(
            status_code=400,
            detail="Only PNG and JPEG images are allowed.",
        )

    image_path = save_uploaded_file(file)

    try:
        imagenet = MODEL_TYPE == "transfer"

        result = predict_xray(
            image_path=image_path,
            model=model,
            device=DEVICE,
            imagenet=imagenet,
        )

        gradcam_path = OUTPUT_DIR / f"{image_path.stem}_gradcam.png"

        generate_gradcam(
            model=model,
            input_tensor=result["input_tensor"],
            prediction=result["prediction_id"],
            probability=result["probability"],
            target_layers=target_layers,
            save_path=gradcam_path,
            imagenet=imagenet,
        )

        return PredictionResponse(
            prediction=result["prediction"],
            probability=result["probability"],
            confidence=result["confidence"],
            gradcam_url=f"/outputs/{gradcam_path.name}",
        )

    finally:
        delete_file(image_path)
