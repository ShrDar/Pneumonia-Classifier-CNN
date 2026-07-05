import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)


def run_epoch(
    model, dataloader, criterion, device, optimizer=None, return_predictions=False
):

    training = optimizer is not None

    if training:
        model.train()
        grad_context = torch.enable_grad()
    else:
        model.eval()
        grad_context = torch.no_grad()

    running_loss = 0

    all_labels = []
    all_probs = []
    all_predictions = []

    with grad_context:
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * images.size(0)

            probabilities = torch.sigmoid(outputs)

            predictions = (probabilities >= 0.5).float()

            all_labels.extend(labels.cpu().numpy())

            all_probs.extend(probabilities.detach().cpu().numpy())

            all_predictions.extend(predictions.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)

    metrics = {
        "loss": epoch_loss,
        "accuracy": accuracy_score(all_labels, all_predictions),
        "precision": precision_score(all_labels, all_predictions),
        "recall": recall_score(all_labels, all_predictions),
        "f1": f1_score(all_labels, all_predictions),
        "roc_auc": roc_auc_score(all_labels, all_probs),
        "pr_auc": average_precision_score(all_labels, all_probs),
    }

    if return_predictions:
        return (
            metrics,
            all_labels,
            all_predictions,
            all_probs,
        )

    return metrics


def fit(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    num_epochs,
    checkpoint_path,
    early_stopping_patience=5,
):
    """
    Train the model and save the best checkpoint based on validation PR-AUC.
    """

    history = {
        "train_acc": [],
        "val_acc": [],
        "train_loss": [],
        "val_loss": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
        "val_roc_auc": [],
        "val_pr_auc": [],
        "lr": [],
    }

    best_pr_auc = 0.0
    patience_counter = 0

    for epoch in range(num_epochs):
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
        )

        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            device,
        )

        old_lr = optimizer.param_groups[0]["lr"]

        scheduler.step(val_metrics["loss"])

        new_lr = optimizer.param_groups[0]["lr"]

        if new_lr < old_lr:
            print(f"LR Reduced: {old_lr:.6f} -> {new_lr:.6f}")

        history["train_acc"].append(train_metrics["accuracy"])
        history["val_acc"].append(val_metrics["accuracy"])

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])

        history["val_precision"].append(val_metrics["precision"])
        history["val_recall"].append(val_metrics["recall"])
        history["val_f1"].append(val_metrics["f1"])
        history["val_roc_auc"].append(val_metrics["roc_auc"])
        history["val_pr_auc"].append(val_metrics["pr_auc"])

        history["lr"].append(new_lr)

        if val_metrics["pr_auc"] > best_pr_auc:
            best_pr_auc = val_metrics["pr_auc"]

            patience_counter = 0

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_pr_auc": best_pr_auc,
                },
                checkpoint_path,
            )

            print(f"Best Model Saved -> PR-AUC: {best_pr_auc:.4f}")

        else:
            patience_counter += 1

        print(
            f"Epoch: {epoch + 1} / {num_epochs} | lr: {new_lr:.6f} | train_acc: {train_metrics['accuracy']} | val_acc: {val_metrics['accuracy']} | train_loss: {train_metrics['loss']:.4f} | val_loss: {val_metrics['loss']:.4f} | val_recall: {val_metrics['recall']:.4f} | val_f1: {val_metrics['f1']:.4f} | val_roc_auc: {val_metrics['roc_auc']:.4f} | val_pr_auc: {val_metrics['pr_auc']:.4f}"
        )

        if patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping after {epoch + 1} epochs.")

            break

    return history
