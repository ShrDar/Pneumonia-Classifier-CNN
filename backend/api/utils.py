from pathlib import Path
from uuid import uuid4
import shutil

from fastapi import UploadFile
from src.config import UPLOAD_DIR

UPLOAD_DIR.mkdir(exist_ok=True)


def save_uploaded_file(file: UploadFile) -> Path:
    file_extension = Path(file.filename).suffix
    file_name = f"{uuid4()}{file_extension}"

    file_path = UPLOAD_DIR / file_name

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return file_path


def delete_file(file_path: Path):

    if file_path.exists():
        file_path.unlink()
