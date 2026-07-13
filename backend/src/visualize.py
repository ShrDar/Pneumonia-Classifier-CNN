import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)

from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


def plot_confusion_matrix(labels, predictions, class_names=("Normal", "Pneumonia")):

    cm = confusion_matrix(labels, predictions)

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")

    plt.show()


def plot_roc_curve(labels, probabilities):
    fpr, tpr, _ = roc_curve(labels, probabilities)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 8))

    plt.plot(fpr, tpr, linewidth=2, label=f"AUC = {roc_auc:.4f}")

    plt.plot([0, 1], [0, 1], "--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")

    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_pr_curve(labels, probabilities):

    precision, recall, _ = precision_recall_curve(labels, probabilities)

    pr_auc = average_precision_score(labels, probabilities)

    plt.figure(figsize=(6, 5))

    plt.plot(recall, precision, linewidth=2, label=f"AP = {pr_auc:.4f}")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")

    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_prediction_distribution(labels, probabilities):

    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)

    plt.figure(figsize=(7, 5))

    plt.axvline(
        x=0.5,
        color="black",
        linestyle="--",
        linewidth=2,
        label="Threshold = 0.5",
    )

    sns.histplot(
        probabilities[labels == 0],
        bins=25,
        color="royalblue",
        label="Normal",
        stat="count",
        alpha=0.6,
    )

    sns.histplot(
        probabilities[labels == 1],
        bins=25,
        color="crimson",
        label="Pneumonia",
        stat="count",
        alpha=0.6,
    )

    plt.xlabel("Predicted Probability")
    plt.ylabel("Count")
    plt.title("Prediction Probability Distribution")

    plt.legend()

    plt.tight_layout()
    plt.show()


def visualize_gradcam(
    model,
    input_tensor,
    prediction,
    probability,
    target_layers,
):
    """
    Generate and display a Grad-CAM++ visualization.
    """

    cam = GradCAMPlusPlus(
        model=model,
        target_layers=target_layers,
    )

    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=[BinaryClassifierOutputTarget(prediction)],
    )[0]

    image = input_tensor.squeeze(0).cpu()

    image = image * 0.5 + 0.5
    image = image.permute(1, 2, 0).numpy()
    image = np.clip(image, 0, 1)

    overlay = show_cam_on_image(
        image,
        grayscale_cam,
        use_rgb=True,
        image_weight=0.6,
    )

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(
        grayscale_cam,
        cmap="jet",
    )
    axes[1].set_title("Grad-CAM++")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    confidence = probability if prediction == 1 else 1 - probability

    plt.suptitle(
        f"Prediction: {'PNEUMONIA' if prediction else 'NORMAL'} | "
        f"Confidence: {confidence * 100:.2f}%"
    )

    plt.tight_layout()
    plt.show()


def plot_training_history(history):

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    axes[0, 0].plot(history["train_loss"], label="Train")
    axes[0, 0].plot(history["val_loss"], label="Validation")
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(history["train_acc"], label="Train")
    axes[0, 1].plot(history["val_acc"], label="Validation")
    axes[0, 1].set_title("Accuracy")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    axes[1, 0].plot(history["val_precision"], label="Precision")
    axes[1, 0].plot(history["val_recall"], label="Recall")
    axes[1, 0].plot(history["val_f1"], label="F1")
    axes[1, 0].set_title("Validation Metrics")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Score")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    axes[1, 1].plot(history["val_roc_auc"], label="ROC-AUC")
    axes[1, 1].plot(history["val_pr_auc"], label="PR-AUC")
    axes[1, 1].set_title("Validation AUC")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Score")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show()


def plot_learning_rate(history):

    plt.figure(figsize=(8, 4))

    plt.plot(history["lr"], marker="o")

    plt.title("Learning Rate Schedule")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")

    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_all_metrics(labels, predictions, probabilites):

    plot_confusion_matrix(labels, predictions)

    plot_roc_curve(labels, probabilites)

    plot_pr_curve(labels, probabilites)

    plot_prediction_distribution(labels, probabilites)
