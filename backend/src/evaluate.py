import torch

from src.trainer import run_epoch


def load_checkpoint(model, checkpoint_path, device):

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])

    return checkpoint


def evaluate_model(model, dataloader, criterion, device):

    metrics, labels, predictions, probabilities = run_epoch(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        device=device,
        return_predictions=True,
    )

    results = {
        "metrics": metrics,
        "labels": labels,
        "predictions": predictions,
        "probabilities": probabilities,
    }

    return results
