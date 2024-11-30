import torch
import torch.nn as nn

from utils.globalConst import *
from utils.plotResutlts import plot_results
from Models.FrameCNN import FrameCNN

class EarlyFusion(nn.Module):
    def __init__(self, device):
        super(EarlyFusion, self).__init__()
        self.device = device

        self.frameCNN = FrameCNN(device, in_channels=FRAMES_PER_VIDEO * 3)

        self.fc = nn.Sequential(
            nn.Linear(3 * 3 * 512, 256),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(256, NUM_CLASSES)
        )

    def forward(self, video_frames):
        T, batch_size, C, H, W = video_frames.shape

        x = video_frames.permute(1, 2, 0, 3, 4).reshape(batch_size, T * C, H, W)

        x = self.frameCNN(x)
        x = x.view(batch_size, -1)
        output = self.fc(x)
        return output
    
    def train_(self, num_epochs, optimizer, criterion, scheduler, train_loader, val_loader, plot=False):
        results = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        for epoch in range(num_epochs):
            self.train()
            self.frameCNN.train()
            train_loss = 0.0
            train_correct = 0

            for video_frames, labels in train_loader:
                video_frames = torch.stack(video_frames, dim=0).to(self.device)
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
            self.frameCNN.eval()
            val_loss = 0.0
            val_correct = 0
            
            with torch.no_grad():
                for video_frames, labels in val_loader:
                    video_frames = torch.stack(video_frames, dim=0).to(self.device)
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
                plot_results(results, epoch)
