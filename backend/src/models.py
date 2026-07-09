import torch.nn as nn


class PneumoniaCNN(nn.Module):
    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(  # 1
                in_channels=1,
                out_channels=32,  # number of filters 32
                kernel_size=3,  # filter size 3 * 3
                padding=1,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(  # 2
                in_channels=32, out_channels=64, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(  # 3
                in_channels=64, out_channels=128, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(  # 4
                in_channels=128, out_channels=256, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
        )

    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)

        return x


class PneumoniaCNN2(nn.Module):
    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(  # 1
                in_channels=1,
                out_channels=32,  # number of filters 32
                kernel_size=3,  # filter size 3 * 3
                padding=1,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(  # 2
                in_channels=32, out_channels=64, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(  # 3
                in_channels=64, out_channels=128, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(  # 4
                in_channels=128, out_channels=256, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(  # 5
                in_channels=256, out_channels=512, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
        )

    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)

        return x
