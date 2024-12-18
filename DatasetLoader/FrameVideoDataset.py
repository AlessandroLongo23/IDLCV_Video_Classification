from glob import glob
import os
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms as T

from utils.globalConst import *

class FrameVideoDataset(torch.utils.data.Dataset):
    def __init__(self, split, transform=None, stack_frames=True, leakage=True): 
        if (leakage):
            self.video_paths = sorted(glob(f'{LEAKAGE_DATASET_DIR}/videos/{split}/*/*.avi'))
            for i in range(len(self.video_paths)):
                self.video_paths[i] = self.video_paths[i].replace('\\', '/')
            self.df = pd.read_csv(f'{LEAKAGE_DATASET_DIR}/metadata/{split}.csv')
            
        else:
            self.video_paths = sorted(glob(f'{NO_LEAKAGE_DATASET_DIR}/videos/{split}/*/*.avi'))
            for i in range(len(self.video_paths)):
                self.video_paths[i] = self.video_paths[i].replace('\\', '/')
            self.df = pd.read_csv(f'{NO_LEAKAGE_DATASET_DIR}/metadata/{split}.csv')
            
        self.split = split
        self.transform = transform
        self.stack_frames = stack_frames
        
        self.n_sampled_frames = 10

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        video_name = video_path.split('/')[-1].split('.avi')[0]
        video_meta = self._get_meta('video_name', video_name)
        label = video_meta['label'].item()

        video_frames_dir = self.video_paths[idx].split('.avi')[0].replace('videos', 'frames')
        video_frames = self.load_frames(video_frames_dir)

        if self.transform:
            frames = [self.transform(frame) for frame in video_frames]
        else:
            frames = [T.ToTensor()(frame) for frame in video_frames]
        
        if self.stack_frames:
            frames = torch.stack(frames).permute(1, 0, 2, 3)

        return frames, label
    
    def _get_meta(self, attr, value):
        return self.df.loc[self.df[attr] == value]
    
    def load_frames(self, frames_dir):
        frames = []
        for i in range(1, self.n_sampled_frames + 1):
            frame_file = os.path.join(frames_dir, f"frame_{i}.jpg")
            frame = Image.open(frame_file).convert("RGB")
            frames.append(frame)

        return frames