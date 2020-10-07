
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.fc1 = nn.Linear(576, 512)

        self.bn2 = nn.BatchNorm1d(num_features=self.fc1.out_features)
        self.fc2 = nn.Linear(512, 512)

        self.fc3 = nn.Linear(512, 1)

    def forward(self, x):

        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.bn1(x)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = F.sigmoid(self.fc3(x))

        return out
