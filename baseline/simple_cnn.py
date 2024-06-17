import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv_layer1 = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            stride=1,
        )
        self.conv_layer2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
        )
        self.max_pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )
        self.fc1 = nn.Linear(248 * 148 * 64, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.relu(out)
        out = self.max_pool(out)

        out = self.conv_layer2(out)
        out = self.relu(out)
        out = self.max_pool(out)

        out = torch.flatten(out, 1)

        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)

        return out
