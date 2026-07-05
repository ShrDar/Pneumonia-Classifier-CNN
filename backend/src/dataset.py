from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image


# Custom Dataset
class PneumoniaDataset(Dataset):
    VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}

    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []

        self.class_map = {"NORMAL": 0, "PNEUMONIA": 1}

        for class_name, label in self.class_map.items():
            class_path = self.root_dir / class_name

            if (
                not class_path.exists()
            ):  # Checking If the required directory exists or not
                raise FileNotFoundError(f"Missing Directory {class_path}")

            for img_path in sorted(class_path.iterdir()):
                if (
                    img_path.suffix.lower() not in self.VALID_EXTENSIONS
                ):  # verifying if images are with valid extensions
                    continue

                try:
                    with Image.open(img_path) as img:  # for skipping corruped image
                        img.verify()

                    self.samples.append((img_path, label))

                except Exception:
                    print(f"Skipping Corrupted Image: {img_path}")

        if not self.samples:
            raise RuntimeError(f"No valid images found in {self.root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        if index >= len(self.samples):
            raise IndexError("Index outta Bound")
        image_path, label = self.samples[index]
        with Image.open(image_path) as img:
            image = img.convert("L")

        if self.transform:
            image = self.transform(image)

        return image, label
