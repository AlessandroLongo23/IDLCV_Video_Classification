from utils.globalConst import *
# import torch
import torch.nn as nn
import torch.functional as F
import torchsummary as summary

class CNN3D(nn.Module):
    def __init__(self, device):
        super(CNN3D, self).__init__()
        
        self.device = device

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.MaxPool2d(2, 2),
            nn.Dropout(0.5),
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.5),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.MaxPool2d(2, 2),
            nn.Dropout(0.5),
        )

        self.fc = nn.Sequential(
            nn.Linear(4 * 4 * 128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(256, 1),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def print(self):
        model = self().to(self.device)
        summary(model, (3, IMAGE_SIZE, IMAGE_SIZE))
    
