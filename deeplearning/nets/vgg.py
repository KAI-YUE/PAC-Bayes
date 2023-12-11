import numpy as np

# Pytorch libraries
import torch.nn as nn
import torch.nn.functional as F

class VGG_7(nn.Module):
    def __init__(self, channel, num_classes, im_size, **kwargs):
        super(VGG_7, self).__init__()
        self.in_channels = channel
        self.input_size = im_size[0]
        self.fc_input_size = int(self.input_size/8)**2 * 512
        
        self.conv1_1 = nn.Conv2d(self.in_channels, 128, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(128, track_running_stats=False, affine=False)
        self.conv1_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(128, track_running_stats=False, affine=False)
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(256, track_running_stats=False, affine=False)
        self.conv2_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(256, track_running_stats=False, affine=False)
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(512, track_running_stats=False, affine=False)
        self.conv3_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(512, track_running_stats=False, affine=False)
        self.mp3 = nn.MaxPool2d(kernel_size=2, stride=2)        

        self.fc1 = nn.Linear(self.fc_input_size, 1024)
        self.bn4 = nn.BatchNorm1d(1024, track_running_stats=False, affine=False)
        self.fc2 = nn.Linear(1024, num_classes)

    # 32C3 - MP2 - 64C3 - Mp2 - 512FC - SM10c
    def forward(self, x):
        x = x.view(x.shape[0], self.in_channels, self.input_size, self.input_size)
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        x = self.mp1(x)

        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x)))
        x = self.mp2(x)

        x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = F.relu(self.bn3_2(self.conv3_2(x)))
        x = self.mp3(x)
        
        x = x.view(x.shape[0], -1)
        x = F.relu(self.bn4(self.fc1(x)))
        # x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x