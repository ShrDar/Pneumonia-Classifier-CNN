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

from src.models import PneumoniaCNN
from src.trainer import fit

from src.visualize import plot_training_history

NUM_EPOCHS = 20

LEARNING_RATE = 1e-4

EARLY_STOPPING_PATIENCE = 5

LR_FACTOR = 0.5

LR_PATIENCE = 2


def main():

    print("Training Baseline CNN")

    train_dataset, val_dataset, test_dataset = create_datasets()

    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
    )

    model = PneumoniaCNN().to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(
        model.parameters(),
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
        checkpoint_path=MODEL_SAVE_PATH / "pneumonia_model.pth",
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
    )

    plot_training_history(history)

    print("\nTraining Completed Successfully")
    print(f"Model saved at: {MODEL_SAVE_PATH / 'pneumonia_model.pth'}")


if __name__ == "__main__":
    main()
