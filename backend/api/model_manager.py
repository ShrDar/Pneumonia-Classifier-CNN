from src.config import (
    DEVICE,
    MODEL_SAVE_PATH,
    HF_MODEL_REPO,
    MODEL_FILES,
)

from src.models import PneumoniaCNN
from src.transfer_model import get_transfer_model
from src.evaluate import load_checkpoint

from huggingface_hub import hf_hub_download

MODELS = {}
TARGET_LAYERS = {}
MODEL_INFO = {}


def load_models():

    print("Preparing Models")

    download_models()

    print("Loading Models")

    baseline1 = PneumoniaCNN().to(DEVICE)

    load_checkpoint(
        model=baseline1,
        checkpoint_path=MODEL_SAVE_PATH / "pneumonia_model.pth",
        device=DEVICE,
    )

    baseline1.eval()

    MODELS["baseline:model1"] = baseline1
    TARGET_LAYERS["baseline:model1"] = [baseline1.features[-4]]
    MODEL_INFO["baseline:model1"] = False

    print("Baseline Model 1 Loaded")

    baseline2 = PneumoniaCNN().to(DEVICE)

    load_checkpoint(
        model=baseline2,
        checkpoint_path=MODEL_SAVE_PATH / "pneumonia_model.pth",
        device=DEVICE,
    )

    baseline2.eval()

    MODELS["baseline:model2"] = baseline2
    TARGET_LAYERS["baseline:model2"] = [baseline2.features[-4]]
    MODEL_INFO["baseline:model2"] = False

    print("Baseline Model 2 Loaded")

    frozen = get_transfer_model("frozen").to(DEVICE)

    load_checkpoint(
        model=frozen,
        checkpoint_path=MODEL_SAVE_PATH / "transfer_resnet18_frozen.pth",
        device=DEVICE,
    )

    frozen.eval()

    MODELS["transfer:frozen"] = frozen
    TARGET_LAYERS["transfer:frozen"] = [frozen.layer4[-1].conv2]
    MODEL_INFO["transfer:frozen"] = True

    print("Transfer Frozen Loaded")

    # finetuned = get_transfer_model("finetune").to(DEVICE)

    # load_checkpoint(
    #     model=finetuned,
    #     checkpoint_path=MODEL_SAVE_PATH / "transfer_resnet18_finetuned.pth",
    #     device=DEVICE,
    # )

    # finetuned.eval()

    # MODELS["transfer:finetuned"] = finetuned
    # TARGET_LAYERS["transfer:finetuned"] = [finetuned.layer4[-1].conv2]
    # MODEL_INFO["transfer:finetuned"] = True

    # print("Transfer Finetuned Loaded")

    print("All Models Loaded Successfully")


def load_selected_model(model_key: str):

    download_models()

    print(f"Loading {model_key}")

    if model_key == "baseline:model1":
        model = PneumoniaCNN().to(DEVICE)

        load_checkpoint(
            model=model,
            checkpoint_path=MODEL_SAVE_PATH / "pneumonia_model.pth",
            device=DEVICE,
        )

        model.eval()

        target_layers = [model.features[-4]]
        imagenet = False

    elif model_key == "baseline:model2":
        model = PneumoniaCNN().to(DEVICE)

        load_checkpoint(
            model=model,
            checkpoint_path=MODEL_SAVE_PATH / "pneumonia_model2.pth",
            device=DEVICE,
        )

        model.eval()

        target_layers = [model.features[-4]]
        imagenet = False

    elif model_key == "transfer:frozen":
        model = get_transfer_model("frozen").to(DEVICE)

        load_checkpoint(
            model=model,
            checkpoint_path=MODEL_SAVE_PATH / "transfer_resnet18_frozen.pth",
            device=DEVICE,
        )

        model.eval()

        target_layers = [model.layer4[-1].conv2]
        imagenet = True

    else:
        raise ValueError(f"Unknown model key: {model_key}")

    print(f"{model_key} loaded successfully")

    return model, target_layers, imagenet


def get_model(model_key: str):

    return MODELS[model_key]


def get_target_layers(model_key: str):

    return TARGET_LAYERS[model_key]


def is_imagenet(model_key: str):

    return MODEL_INFO[model_key]


def download_models():

    MODEL_SAVE_PATH.mkdir(exist_ok=True)

    for filename in MODEL_FILES:
        model_path = MODEL_SAVE_PATH / filename

        if model_path.exists():
            print(f"{filename} model already exists")
            continue

        print(f"Downloading Model {filename}...")

        hf_hub_download(
            repo_id=HF_MODEL_REPO,
            filename=filename,
            local_dir=MODEL_SAVE_PATH,
        )

        print(f"{filename} model downloaded")
