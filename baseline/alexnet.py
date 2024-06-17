import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            # -------- Conv1 -------- #
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=11,
                stride=4,
                padding=2,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(
                kernel_size=3,
                stride=2,
            ),
            # -------- Conv2 -------- #
            nn.Conv2d(
                in_channels=64,
                out_channels=192,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(
                kernel_size=3,
                stride=2,
            ),
            # -------- Conv3 -------- #
            nn.Conv2d(
                in_channels=192,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            # -------- Conv4 -------- #
            nn.Conv2d(
                in_channels=256,
                out_channels=384,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            # -------- Conv5 -------- #
            nn.Conv2d(
                in_channels=384,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(
                kernel_size=3,
                stride=2,
            ),
        )

        self.classifier = nn.Sequential(
            # ---------- FC1 ---------- #
            nn.Linear(256 * 30 * 18, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(
                p=0.5,
                inplace=True,
            ),
            # ---------- FC2 ---------- #
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(
                p=0.5,
                inplace=True,
            ),
            # ---------- FC3 ---------- #
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# Example usage
# model = AlexNet(num_classes=21)
# print(model)
