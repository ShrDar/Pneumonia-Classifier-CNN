import base64

from fastapi import (
    APIRouter,
    File,
    UploadFile,
    HTTPException,
    Form,
)

from api.schemas import PredictionResponse

from api.utils import save_uploaded_file, delete_file

from api.model_manager import get_model, get_target_layers, is_imagenet

from src.config import DEVICE

from src.inference import predict_xray
from src.visualize import generate_gradcam

router = APIRouter()


@router.post(
    "/predict",
    response_model=PredictionResponse,
)
async def predict(
    file: UploadFile = File(...),
    model: str = Form(...),
    type: str = Form(...),
):

    if file.content_type not in [
        "image/jpeg",
        "image/png",
    ]:
        raise HTTPException(
            status_code=400,
            detail="Only PNG and JPEG images are allowed.",
        )

    if model == "baseline":
        if type == "model1":
            model_key = "baseline:model1"

        elif type == "model2":
            model_key = "baseline:model2"

        else:
            raise HTTPException(
                status_code=400,
                detail="Invalid baseline model.",
            )

    elif model == "transfer":
        if type == "frozen":
            model_key = "transfer:frozen"

        else:
            raise HTTPException(
                status_code=400,
                detail="Invalid transfer model.",
            )

    else:
        raise HTTPException(
            status_code=400,
            detail="Invalid model.",
        )

    model_instance = get_model(model_key)

    target_layers = get_target_layers(model_key)

    imagenet = is_imagenet(model_key)

    image_path = save_uploaded_file(file)

    try:
        result = predict_xray(
            image_path=image_path,
            model=model_instance,
            device=DEVICE,
            imagenet=imagenet,
        )

        gradcam_buffer = generate_gradcam(
            model=model_instance,
            input_tensor=result["input_tensor"],
            prediction=result["prediction_id"],
            probability=result["probability"],
            target_layers=target_layers,
            imagenet=imagenet,
        )

        gradcam_base64 = base64.b64encode(gradcam_buffer.getvalue()).decode("utf-8")

        return PredictionResponse(
            prediction=result["prediction"],
            probability=result["probability"],
            confidence=result["confidence"],
            gradcam=gradcam_base64,
        )

    finally:
        delete_file(image_path)
