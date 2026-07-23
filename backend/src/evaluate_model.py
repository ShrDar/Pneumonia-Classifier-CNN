import torch.nn as nn

from config import (
    DEVICE,
    MODEL_SAVE_PATH,
)

from dataset import (
    create_datasets,
    create_dataloaders,
)

from models import PneumoniaCNN
from transfer_model import get_transfer_model

from evaluate import (
    load_checkpoint,
    evaluate_model,
)

from visualize import (
    plot_all_metrics,
)

from config import MODEL_TYPE, TRANSFER_MODE


def main():

    if MODEL_TYPE == "cnn":
        CHECKPOINT_PATH = MODEL_SAVE_PATH / "pneumonia_model.pth"

        print("Evaluating CNN Model")
        train_dataset, val_dataset, test_dataset = create_datasets()

        model = PneumoniaCNN().to(DEVICE)

    elif MODEL_TYPE == "transfer":
        CHECKPOINT_PATH = MODEL_SAVE_PATH / f"transfer_resnet18_{TRANSFER_MODE}.pth"

        print("Evaluating CNN Transfer Model")
        train_dataset, val_dataset, test_dataset = create_datasets(imagenet=True)

        model = get_transfer_model(mode=TRANSFER_MODE).to(DEVICE)

    else:
        raise ValueError("MODEL_TYPE must be 'cnn' or 'transfer'")

    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
    )

    criterion = nn.BCEWithLogitsLoss()

    print("\nLoading Model")

    checkpoint = load_checkpoint(
        model,
        CHECKPOINT_PATH,
        DEVICE,
    )

    print("Model Loaded Successfully!")

    print("\nRunning Evaluation...\n")

    results = evaluate_model(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=DEVICE,
    )

    metrics = results["metrics"]
    labels = results["labels"]
    predictions = results["predictions"]
    probabilities = results["probabilities"]

    print("Test Metrics\n")

    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

    plot_all_metrics(
        labels,
        predictions,
        probabilities,
    )


if __name__ == "__main__":
    main()
