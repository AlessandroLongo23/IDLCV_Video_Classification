import torch
import torch.nn as nn
from utils.globalConst import *
from Models.FrameCNN import FrameCNN

class LateFusionMLP(nn.Module):
    def __init__(self, device):
        super(LateFusionMLP, self).__init__()
        self.device = device
        self.frameCNN = FrameCNN(device)
            
        self.mlp = nn.Sequential(
            nn.Linear(FRAMES_PER_VIDEO * 128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, NUM_CLASSES)
        )

    def forward(self, video_frames):
        frame_outputs = []
        
        # Process each frame through the CNN
        for frame in video_frames:
            frame = frame.to(self.device)
            frame_output = self.frameCNN(frame)
            frame_outputs.append(frame_output)

        # Stack frame outputs and flatten them
        frame_outputs = torch.stack(frame_outputs, dim=1)
        frame_outputs = frame_outputs.view(frame_outputs.size(0), -1)
        
        # Pass through the MLP
        output = self.mlp(frame_outputs)
        return output
    
    def train_(self, num_epochs, optimizer, criterion, train_loader, val_loader, plot=False):
        for epoch in range(num_epochs):
            self.train()
            running_loss = 0.0
            train_correct = 0.0

            for video_frames, labels in train_loader:
                video_frames = torch.stack(video_frames, dim=0).to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = self(video_frames)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_correct += (predicted == labels).sum().item()
                
            self.eval()
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
                    
            print(
                f'Epoch {epoch+1}/{num_epochs}\t',
                f'Train loss: {running_loss / len(train_loader):.3f}, Val loss: {val_loss / len(val_loader):.3f}\t',
                f'Train accuracy: {train_correct / len(train_loader.dataset) * 100:.2f}%, Val accuracy: {val_correct / len(val_loader.dataset) * 100:.2f}%'
            )
            
            if plot:
                pass