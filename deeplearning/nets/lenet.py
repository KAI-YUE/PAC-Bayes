import torch.nn as nn
import torch.nn.functional as F
from torch import set_grad_enabled


# Acknowledgement to
# https://github.com/kuangliu/pytorch-cifar,
# https://github.com/BIGBALLON/CIFAR-ZOO,

class LeNet(nn.Module):
    def __init__(self, channel, num_classes, im_size,  no_grad: bool = False,
                 pretrained: bool = False):
        if pretrained:
            raise NotImplementedError("torchvison pretrained models not available.")
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channel, 6, kernel_size=5, padding=2 if channel == 1 else 0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_1 = nn.Linear(16 * 53 * 53 if im_size[0] == im_size[1] == 224 else 16 * 5 * 5, 120)
        self.fc_2 = nn.Linear(120, 84)
        self.fc_3 = nn.Linear(84, num_classes)

        self.no_grad = no_grad

    def get_last_layer(self):
        return self.fc_3

    def forward(self, x):
        with set_grad_enabled(not self.no_grad):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc_1(x))
            x = F.relu(self.fc_2(x))
            x = self.fc_3(x)
        return x

class LeNet_5(nn.Module):
    def __init__(self, channel, num_classes, im_size, pretrained: bool = False):
        super(LeNet_5, self).__init__()
        self.in_channels = channel
        self.input_size = im_size[0]
        self.fc_input_size = int(self.input_size/4)**2 * 64
        
        self.conv1 = nn.Conv2d(self.in_channels, 32, kernel_size=5, padding=2)
        self.mp1= nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.mp2= nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(self.fc_input_size, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.mp1(x)

        x = F.relu(self.conv2(x))
        x = self.mp2(x)
        
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x