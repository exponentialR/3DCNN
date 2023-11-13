import torch
import torch.nn as nn
import torch.nn.functional as F

class Example3DCNN(nn.Module):
    def __init__(self):
        super(Example3DCNN, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.conv2 = nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(32)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm3d(64)
        self.conv4 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm3d(64)
        self.conv5 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm3d(128)
        self.conv6 = nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm3d(128)

        # Max Pooling Layer
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Fully Connected/Dense Layer
        self.fc1 = nn.Linear(128*8*8*8, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward (self, input):
        x = self.pool(F.relu(self.bn1(self.conv1(input))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool(F.relu(self.bn6(self.conv6(x))))

        # Flatten the output for the fully connected layer
        x = x.view(-1, 128*8*8*8)
        x = self.fc2(x)
        return x



