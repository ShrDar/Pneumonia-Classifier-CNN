from PIL import Image

import torch

from src.transforms import get_other_transform


def load_image(image_path, transform, device, imagenet=False):
    if imagenet:
        image = Image.open(image_path).convert("RGB")
    else:
        image = Image.open(image_path).convert("L")

    transform = get_other_transform(imagenet)
    input_tensor = transform(image).unsqueeze(0).to(device)
    return input_tensor


def predict(model, input_tensor, threshold=0.5):
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)

        probability = torch.sigmoid(output).item()

        prediction = int(probability >= threshold)

        confidence = probability if prediction == 1 else 1 - probability

    return prediction, probability, confidence


def predict_xray(
    image_path,
    model,
    device,
    imagenet=False,
    threshold=0.6,
):

    transform = get_other_transform(imagenet)
    input_tensor = load_image(
        image_path,
        transform,
        device,
        imagenet=imagenet,
    )

    prediction, probability, confidence = predict(
        model,
        input_tensor,
        threshold,
    )

    return {
        "prediction": "PNEUMONIA" if prediction else "NORMAL",
        "prediction_id": prediction,
        "confidence": confidence,
        "probability": probability,
        "input_tensor": input_tensor,
    }
