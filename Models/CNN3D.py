import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from IPython.display import clear_output

from utils.globalConst import *
from utils.plotResutlts import plot_results

class CNN3D(nn.Module):
    def __init__(self, device):
        super(CNN3D, self).__init__()
        self.device = device
        
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))  # Downsample spatial dimensions only
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)  # Downsample both temporal and spatial dimensions
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)  # Downsample both temporal and spatial dimensions
        )
        
        self.fc = nn.Sequential(
            nn.Linear(128 * (FRAMES_PER_VIDEO // 4) * (IMAGE_SIZE // 8) * (IMAGE_SIZE // 8), 512),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(512, NUM_CLASSES)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x
    
    def train_(self, num_epochs, optimizer, criterion, scheduler, train_loader, val_loader, plot=False):
        results = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        for epoch in range(num_epochs):
            self.train()
            train_loss = 0.0
            train_correct = 0

            for video_frames, labels in train_loader:
                video_frames = torch.stack(video_frames, dim=0).to(self.device)
                video_frames = video_frames.permute(1, 2, 0, 3, 4)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = self(video_frames)
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
                for video_frames, labels in val_loader:
                    video_frames = torch.stack(video_frames, dim=0).to(self.device)
                    video_frames = video_frames.permute(1, 2, 0, 3, 4)
                    labels = labels.to(self.device)
                    
                    outputs = self(video_frames)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
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
            for video_frames, labels in test_loader:
                video_frames = torch.stack(video_frames, dim=0).to(self.device)
                video_frames = video_frames.permute(1, 2, 0, 3, 4)
                labels = labels.to(self.device)

                outputs = self(video_frames)
                loss = criterion(outputs, labels)

                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                test_correct += (predicted == labels).sum().item()

        return {
            'loss': test_loss / len(test_loader), 
            'accuracy': test_correct / len(test_loader.dataset) * 100
        }