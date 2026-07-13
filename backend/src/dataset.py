from pathlib import Path

from PIL import Image

from sklearn.model_selection import train_test_split

from torch.utils.data import (
    Dataset,
    DataLoader,
    Subset,
    WeightedRandomSampler,
)

from config import (
    TRAIN_DIR,
    TEST_DIR,
    VALIDATION_SPLIT,
    SEED,
    BATCH_SIZE,
    NUM_WORKERS,
)

from transforms import (
    get_train_transform,
    get_other_transform,
)

from config import DEVICE


# Custom Dataset
class PneumoniaDataset(Dataset):
    VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}

    def __init__(self, root_dir, transform=None, rgb=False):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.rgb = rgb
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
            image = img.convert("RGB" if self.rgb else "L")

        if self.transform:
            image = self.transform(image)

        return image, label


def create_datasets(imagenet=False):

    full_train_dataset = PneumoniaDataset(
        TRAIN_DIR,
        transform=None,
        rgb=imagenet,
    )

    full_train_labels = [label for _, label in full_train_dataset.samples]

    indices = list(range(len(full_train_dataset)))

    train_idx, val_idx = train_test_split(
        indices,
        test_size=VALIDATION_SPLIT,
        random_state=SEED,
        stratify=full_train_labels,
    )

    train_dataset = Subset(
        PneumoniaDataset(
            TRAIN_DIR,
            transform=get_train_transform(imagenet),
            rgb=imagenet,
        ),
        train_idx,
    )

    val_dataset = Subset(
        PneumoniaDataset(
            TRAIN_DIR,
            transform=get_other_transform(imagenet),
            rgb=imagenet,
        ),
        val_idx,
    )

    test_dataset = PneumoniaDataset(
        TEST_DIR,
        transform=get_other_transform(imagenet),
        rgb=imagenet,
    )

    return train_dataset, val_dataset, test_dataset


def create_sampler(train_dataset):

    labels = [train_dataset.dataset.samples[idx][1] for idx in train_dataset.indices]

    class_counts = [
        labels.count(0),
        labels.count(1),
    ]

    class_weights = [len(labels) / count for count in class_counts]

    sample_weights = [class_weights[label] for label in labels]

    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    return sampler


def create_dataloaders(
    train_dataset,
    val_dataset,
    test_dataset,
):

    sampler = create_sampler(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=DEVICE == "cuda",
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=DEVICE == "cuda",
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=DEVICE == "cuda",
    )

    return (
        train_loader,
        val_loader,
        test_loader,
    )
