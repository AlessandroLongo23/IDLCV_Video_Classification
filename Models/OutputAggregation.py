import torch
import torch.nn as nn
import torchsummary as summary

from utils.plotResutlts import plot_results
from utils.globalConst import *
from Models.FrameCNN import FrameCNN

class OutputAggregation(nn.Module):
    def __init__(self, device):
        super(OutputAggregation, self).__init__()
        self.device = device
        
        self.frameCNN = FrameCNN(device)
            
        self.fc = nn.Sequential(
            nn.Linear(3 * 3 * 512, 256),
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
        
    def train_(self, num_epochs, optimizer, criterion, train_loader, val_loader, scheduler=None, plot=False):
        results = {
            'train_acc': [],
            'val_acc': [],
            'train_loss': [],
            'val_loss': []
        }

        for epoch in range(num_epochs):
            self.train()
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

            if scheduler:
                scheduler.step(val_loss / len(val_loader))
            
            if plot:
                plot_results(results, epoch, num_epochs)

    def eval_(self, criterion, test_loader):
        self.eval()
        test_loss = 0.0
        test_correct = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self(data)
                loss = criterion(output, target)
                test_loss += loss.item()
                predicted = torch.argmax(output, dim=1)
                test_correct += (target == predicted).sum().cpu().item()
                
        return {
            'loss': test_loss / len(test_loader), 
            'accuracy': test_correct / len(test_loader.dataset) * 100
        }