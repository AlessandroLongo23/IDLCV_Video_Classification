from glob import glob
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms as T

from utils.globalConst import *

class FrameImageDataset(torch.utils.data.Dataset):
    def __init__(self, split='train', transform=None, leakage=True):
        if (leakage):
            self.frame_paths = sorted(glob(f'{LEAKAGE_DATASET_DIR}/frames/{split}/*/*/*.jpg'))
            for i in range(len(self.frame_paths)):
                self.frame_paths[i] = self.frame_paths[i].replace('\\', '/')
            self.df = pd.read_csv(f'{LEAKAGE_DATASET_DIR}/metadata/{split}.csv')
                
        else:
            self.frame_paths = sorted(glob(f'{NO_LEAKAGE_DATASET_DIR}/frames/{split}/*/*/*.jpg'))
            for i in range(len(self.frame_paths)):
                self.frame_paths[i] = self.frame_paths[i].replace('\\', '/')
            self.df = pd.read_csv(f'{NO_LEAKAGE_DATASET_DIR}/metadata/{split}.csv')
        
        self.split = split
        self.transform = transform
       
    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx):
        frame_path = self.frame_paths[idx]
        video_name = frame_path.split('/')[-2]
        video_meta = self._get_meta('video_name', video_name)
        label = video_meta['label'].item()

        frame = Image.open(frame_path).convert("RGB")

        if self.transform:
            frame = self.transform(frame)
        else:
            frame = T.ToTensor()(frame)

        return frame, label
    
    def _get_meta(self, attr, value):
        return self.df.loc[self.df[attr] == value]