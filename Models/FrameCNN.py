import torch.nn as nn

from utils.globalConst import *

class FrameCNN(nn.Module):
    def __init__(self, device, in_channels=3):
        super(FrameCNN, self).__init__()
        self.device = device
        self.in_channels = in_channels
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )            
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        
        self.dropout = nn.Dropout(DROPOUT_RATE)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.dropout(x)
        
        x = self.conv3(x)
        x = self.dropout(x)
        
        return x
