import torch
import torch.nn as nn

from utils.globalConst import *
from utils.plotResutlts import plot_results
from Models.FrameCNN import FrameCNN

class LateFusionMLP(nn.Module):
    def __init__(self, device):
        super(LateFusionMLP, self).__init__()
        self.device = device
        self.frameCNN = FrameCNN(device)
            
        self.mlp = nn.Sequential(
            nn.Linear(FRAMES_PER_VIDEO * 3 * 3 * 512, 512),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            
            nn.Linear(512, NUM_CLASSES)
        )

    def forward(self, video_frames):
        frame_outputs = []
        
        for frame in video_frames:
            frame = frame.to(self.device)
            frame_output = self.frameCNN(frame)
            frame_outputs.append(frame_output)

        frame_outputs = torch.stack(frame_outputs, dim=1)
        frame_outputs = frame_outputs.view(frame_outputs.size(0), -1)
        
        output = self.mlp(frame_outputs)
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
            train_correct = 0.0

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
        self.frameCNN.eval()
        test_loss = 0.0
        test_correct = 0

        with torch.no_grad():
            for video_frames, labels in test_loader:
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