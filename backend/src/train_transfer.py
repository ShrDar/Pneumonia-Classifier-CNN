import torch.nn as nn
import torch.optim as optim

from src.config import (
    DEVICE,
    MODEL_SAVE_PATH,
)

from src.dataset import (
    create_datasets,
    create_dataloaders,
)

from src.transfer_model import get_transfer_model

from src.trainer import fit

from src.visualize import (
    plot_training_history,
    plot_learning_rate,
)


MODE = "frozen"  # "frozen" or "finetune"

NUM_EPOCHS = 15

LEARNING_RATE = 1e-4 if MODE == "frozen" else 1e-5

EARLY_STOPPING_PATIENCE = 5

LR_FACTOR = 0.5

LR_PATIENCE = 2

CHECKPOINT_PATH = MODEL_SAVE_PATH / f"transfer_resnet18_{MODE}.pth"


def main():

    print(f"Training ResNet18 ({MODE.upper()})")

    train_dataset, val_dataset, test_dataset = create_datasets(imagenet=True)

    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
    )

    model = get_transfer_model(mode=MODE).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]

    optimizer = optim.Adam(
        trainable_parameters,
        lr=LEARNING_RATE,
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=LR_FACTOR,
        patience=LR_PATIENCE,
    )

    history = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=DEVICE,
        num_epochs=NUM_EPOCHS,
        checkpoint_path=CHECKPOINT_PATH,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
    )

    plot_training_history(history)

    plot_learning_rate(history)

    print("\nTraining Completed Successfully!")

    print(f"Model saved to:\n{CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
