import torch
import torch.nn as nn
import torchsummary as summary
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

from utils.globalConst import *
from Models.FrameCNN import FrameCNN

class OutputAggregation(nn.Module):
    def __init__(self, device):
        super(OutputAggregation, self).__init__()
        self.device = device
        
        self.frameCNN = FrameCNN(device)
            
        self.fc = nn.Sequential(
            nn.Linear(8 * 8 * 128, 256),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(256, NUM_CLASSES),
        )

    def forward(self, x):
        x = self.frameCNN(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def print(self):
        model = self().to(self.device)
        summary(model, (3, IMAGE_SIZE, IMAGE_SIZE))
        
    def train_(self, num_epochs, optimizer, criterion, train_loader, val_loader, plot=False):
        results = {
            'train_acc': [],
            'val_acc': [],
            'train_loss': [],
            'val_loss': []
        }

        for epoch in range(num_epochs):
            self.train()
            self.frameCNN.train()
            train_loss = 0.0
            train_correct = 0

            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                output = torch.softmax(output, dim=1)
                predicted = torch.argmax(output, dim=1)
                train_correct += (target == predicted).sum().cpu().item()

            results['train_loss'].append(train_loss / len(train_loader))
            results['train_acc'].append(train_correct / len(train_loader.dataset) * 100)

            self.eval()
            self.frameCNN.eval()
            val_loss = 0.0
            val_correct = 0
            
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)

                with torch.no_grad():
                    output = self(data)
                
                loss = criterion(output, target)
                val_loss += loss.item()
                predicted = torch.argmax(output, dim=1)
                val_correct += (target == predicted).sum().cpu().item()

            
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
        
    def eval_(self, test_loader):
        self.eval()
        test_correct = 0
        
        for video_frames, labels in test_loader:
            target = labels.to(self.device)
            outputs = []
            
            for frame in video_frames:
                frame = frame.to(self.device)

                with torch.no_grad():
                    output = self(frame)
                
                output = torch.softmax(output, dim=1)
                output = torch.argmax(output, dim=1)
                outputs.append(output)
            
            outputs = torch.stack(outputs, dim=0)
            output, _ = torch.mode(outputs, dim=0)
            
            test_correct += (target == output).sum().cpu().item()

        accuracy = test_correct / len(test_loader.dataset)
        print(f"Test accuracy: {accuracy * 100:.1f}%")
        
    # def plot_training_history(self):
        