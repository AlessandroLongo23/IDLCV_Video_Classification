import torch
import torch.nn as nn
import numpy as np

from utils.globalConst import *  
from utils.plotResutlts import plot_results

class DualStream(nn.Module):
    def __init__(self, device):
        super(DualStream, self).__init__()
        self.device = device
        
        self.spatial_stream = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(96, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            
            nn.Linear(512, 4096),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            
            nn.Linear(2048, 10),
            nn.Softmax(dim=1)
        )
        
        self.temporal_stream = nn.Sequential(
            nn.Conv2d(18, 96, kernel_size=7, stride=2, padding=3),  # conv1: 7x7x96
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(96, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            
            nn.Linear(512, 4096),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            
            nn.Linear(2048, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, rgb_input, optical_flow_input):
        spatial_output = self.spatial_stream(rgb_input)
        temporal_output = self.temporal_stream(optical_flow_input)

        fused_output = spatial_output + temporal_output
        return fused_output

    def train_(self, num_epochs, optimizer, scheduler, criterion, train_loader, val_loader, plot=False):
        results = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }

        for epoch in range(num_epochs):
            self.train()
            train_loss = 0
            train_correct = 0

            for flows, labels in train_loader:
                flows = flows.to(self.device)
                labels = labels.to(self.device)

                rgb_frames = flows[:, :3, :, :]

                optimizer.zero_grad()
                outputs = self(rgb_frames, flows)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_correct += (predicted == labels).sum().item()

            results['train_loss'].append(train_loss / len(train_loader))
            results['train_acc'].append(train_correct / len(train_loader.dataset) * 100)

            self.eval()
            val_loss = 0.0
            val_correct = 0

            with torch.no_grad():
                for flows, labels in val_loader:
                    flows = flows.to(self.device)
                    labels = labels.to(self.device)

                    rgb_frames = flows[:, :3, :, :]

                    outputs = self(rgb_frames, flows)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    val_correct += (predicted == labels).sum().item()

            results['val_loss'].append(val_loss / len(val_loader))
            results['val_acc'].append(val_correct / len(val_loader.dataset) * 100)
            
            if scheduler:
                scheduler.step(val_loss / len(val_loader))

            if plot:
                plot_results(results, epoch, num_epochs)
                
    def eval_(self, criterion, test_loader):
        self.eval()
        test_loss = 0.0
        test_correct = 0

        with torch.no_grad():
            for flows, labels in test_loader:
                flows = flows.to(self.device)
                labels = labels.to(self.device)

                rgb_frames = flows[:, :3, :, :]

                outputs = self(rgb_frames, flows)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                test_correct += (predicted == labels).sum().item()

        return {
            'loss': test_loss / len(test_loader), 
            'accuracy': test_correct / len(test_loader.dataset) * 100
        }
                
                
def smooth(data, window_size=5):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')