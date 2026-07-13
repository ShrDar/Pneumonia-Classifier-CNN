import argparse
import torch

from config import DEVICE
from inference import predict_xray
from transforms import get_other_transform
from transfer_model import get_transfer_model


def parse_arguments():

    parser = argparse.ArgumentParser(description="Pneumonia Detection Using ResNet18")

    parser.add_argument(
        "--image", type=str, required=True, help="Path to the chest X-Ray Image"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="../models/transfer_resnet18_frozen.pth",
        help="Path to the trained model checkpoint",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="Classification threshold",
    )

    return parser.parse_args()


def main():

    args = parse_arguments()

    print("Loading Model ")

    model = get_transfer_model(mode="finetune")

    checkpoint = torch.load(
        args.checkpoint,
        map_location=DEVICE,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(DEVICE)
    model.eval()

    print("Model Loaded Successfully\n")

    transform = get_other_transform(imagenet=True)

    target_layers = [model.layer4[-1]]

    result = predict_xray(
        image_path=args.image,
        model=model,
        transform=transform,
        target_layers=target_layers,
        device=DEVICE,
        threshold=args.threshold,
    )

    print("\n")
    print("Prediction Results")

    print(f"Prediction : {result['prediction']}")
    print(f"Confidence : {result['confidence'] * 100:.2f}%")
    print(f"Probability: {result['probability']:.4f}")


if __name__ == "__main__":
    main()
