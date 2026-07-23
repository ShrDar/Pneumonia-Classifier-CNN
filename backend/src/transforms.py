from src.config import IMG_SIZE
from torchvision import transforms


def get_normalization(imagenet=False):
    if imagenet:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        return mean, std

    return [0.5], [0.5]


def get_train_transform(imagenet=False):

    mean, std = get_normalization(imagenet)

    train_transform = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.RandomAffine(
                degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    return train_transform


def get_other_transform(imagenet=False):

    mean, std = get_normalization(imagenet)

    if imagenet:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        mean = [0.5]
        std = [0.5]

    other_transform = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    return other_transform
