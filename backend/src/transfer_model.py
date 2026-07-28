import torch.nn as nn
from torchvision.models import resnet18


def get_transfer_model(mode="frozen"):
    # weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=None)

    if mode == "frozen":
        for param in model.parameters():
            param.requires_grad = False

    # elif mode == "finetuned":
    #     for param in model.parameters():
    #         param.requires_grad = True

    else:
        raise ValueError("Mode Must be 'frozen'")

    num_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Linear(num_features, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, 1)
    )

    return model
