from utils.globalConst import *
import torch
import torch.nn as nn
import torchsummary as summary
from tqdm import tqdm
import numpy as np

class OutputAggregation(nn.Module):
    def __init__(self, device):
        super(OutputAggregation, self).__init__()
        
        self.device = device
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.5),
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.5),
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.5),
        )
            
        self.fc = nn.Sequential(
            nn.Linear(8 * 8 * 128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, NUM_CLASSES),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
            
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def print(self):
        model = self().to(self.device)
        summary(model, (3, IMAGE_SIZE, IMAGE_SIZE))
        
    def train_(self, num_epochs, optimizer, criterion, train_loader, val_loader, plot=False):
        out_dict = {
            'train_acc': [],
            'val_acc': [],
            'train_loss': [],
            'val_loss': []
        }

        for _ in tqdm(range(num_epochs), unit='epoch'):
            self.train()
            train_correct = 0
            train_loss = []

            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                train_loss.append(loss.item())

                output = torch.softmax(output, dim=1)
                predicted = torch.argmax(output, dim=1)
                train_correct += (target == predicted).sum().cpu().item() / len(target)

            self.eval()
            val_loss = []
            val_correct = 0
            
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)

                with torch.no_grad():
                    output = self(data)
                
                loss = criterion(output, target)
                val_loss.append(loss.cpu().item())
                predicted = torch.argmax(output, dim=1)
                val_correct += (target == predicted).sum().cpu().item() / len(target)

            out_dict['train_acc'].append(train_correct / len(train_loader))
            out_dict['val_acc'].append(val_correct / len(val_loader))
            out_dict['train_loss'].append(np.mean(train_loss))
            out_dict['val_loss'].append(np.mean(val_loss))
            
            if plot:
                # self.plot_training_history()
                print(
                    f"Loss train: {np.mean(train_loss):.3f}\t val: {np.mean(val_loss):.3f}\t",
                    f"Accuracy train: {out_dict['train_acc'][-1] * 100:.1f}%\t val: {out_dict['val_acc'][-1] * 100:.1f}%"
                )
        
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
        