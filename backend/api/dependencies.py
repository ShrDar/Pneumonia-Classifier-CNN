from src.config import DEVICE, MODEL_TYPE, TRANSFER_MODE, MODEL_SAVE_PATH

from src.models import PneumoniaCNN
from src.transfer_model import get_transfer_model
from src.evaluate import load_checkpoint


def load_model():
    """Loading the Model with the trained checkpoint parameters"""

    if MODEL_TYPE == "cnn":
        checkpoint_path = MODEL_SAVE_PATH / "pneumonia_model.pth"
        model = PneumoniaCNN().to(DEVICE)

    elif MODEL_TYPE == "transfer":
        checkpoint_path = MODEL_SAVE_PATH / f"transfer_resnet18_{TRANSFER_MODE}.pth"

        model = get_transfer_model(mode=TRANSFER_MODE).to(DEVICE)

    else:
        raise ValueError("MODEL_TYPE must be either 'cnn' or 'transfer'")

    print("Loading Model")

    load_checkpoint(model=model, checkpoint_path=checkpoint_path, device=DEVICE)

    model.eval()

    print("Model Loaded Successfully")

    return model


model = load_model()


def get_model():
    return model
