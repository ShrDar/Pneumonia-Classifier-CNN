from pathlib import Path
import torch

SEED = 67

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 2
VALIDATION_SPLIT = 0.2

ROOT_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT_DIR / "data" / "train"
# VAL_DIR = "..."
TEST_DIR = ROOT_DIR / "data" / "test"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_SAVE_PATH = ROOT_DIR / "models"

MODEL_TYPE = "transfer"
TRANSFER_MODE = "frozen"

# For API

UPLOAD_DIR = ROOT_DIR / "uploads"
