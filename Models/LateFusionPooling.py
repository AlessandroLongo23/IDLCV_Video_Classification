import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from IPython.display import clear_output

from utils.globalConst import *
from Models.FrameCNN import FrameCNN

class LateFusionPooling(nn.Module):
    def __init__(self, device):
        super(LateFusionPooling, self).__init__()
        self.device = device
        self.frameCNN = FrameCNN(device)
        
        self.mlp = nn.Linear(128, NUM_CLASSES)

    def forward(self, video_frames):
        frame_outputs = []
        
        for frame in video_frames:
            frame_output = self.frameCNN(frame)
            frame_outputs.append(frame_output)
            
        # Stack frame outputs (batch_size x D x H' x W') -> (batch_size x T x D x H' x W')
        frame_outputs = torch.stack(frame_outputs, dim=1)
        
        # Perform average pooling over the time dimension (T) and space (H' x W')
        # The resulting shape will be (batch_size, D, 1, 1) for each feature map
        pooled_output = frame_outputs.mean(dim=[1, 3, 4])  # Pooling over T, H', W'

        # Flatten the pooled output and pass it through the MLP
        output = self.mlp(pooled_output)
        return output

    def train_(self, num_epochs, optimizer, criterion, train_loader, val_loader, plot=False):
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
                    video_frames = torch.stack(video_frames, dim=0).to(self.device)
                    labels = labels.to(self.device)
                    outputs = self(video_frames)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_correct += (predicted == labels).sum().item()

            results['val_loss'].append(val_loss / len(val_loader))
            results['val_acc'].append(val_correct / len(val_loader.dataset) * 100)

            if plot:
                clear_output(wait=True)
                fig, axs = plt.subplots(1, 2, figsize=(20, 8))

                axs[0].plot(range(epoch + 1), results['train_loss'], label="Train Loss", color="blue")
                axs[0].plot(range(epoch + 1), results['val_loss'], label="Validation Loss", color="orange")
                axs[0].set_title("Loss over Epochs", fontsize=16)
                axs[0].set_xlabel("Epochs", fontsize=14)
                axs[0].set_ylabel("Loss", fontsize=14)
                axs[0].set_ylim(0, 2.5)
                axs[0].legend(fontsize=12)
                axs[0].grid()

                axs[1].plot(range(epoch + 1), results['train_acc'], label="Train Accuracy", color="blue")
                axs[1].plot(range(epoch + 1), results['val_acc'], label="Validation Accuracy", color="orange")
                axs[1].set_title("Accuracy over Epochs", fontsize=16)
                axs[1].set_xlabel("Epochs", fontsize=14)
                axs[1].set_ylabel("Accuracy (%)", fontsize=14)
                axs[1].legend(fontsize=12)
                axs[1].grid()

                plt.tight_layout()
                plt.show()