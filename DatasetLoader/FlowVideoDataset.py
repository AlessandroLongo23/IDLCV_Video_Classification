from glob import glob
import os
import pandas as pd
import torch
from torchvision import transforms as T
import numpy as np
import torch.nn.functional as F

from utils.globalConst import *

class FlowVideoDataset(torch.utils.data.Dataset):
    def __init__(self, split='train', transform=None):
        self.video_paths = sorted(glob(f'{NO_LEAKAGE_DATASET_DIR}/videos/{split}/*/*.avi'))
        for i in range(len(self.video_paths)):
            self.video_paths[i] = self.video_paths[i].replace('\\', '/')
            
        self.df = pd.read_csv(f'{NO_LEAKAGE_DATASET_DIR}/metadata/{split}.csv')
        self.split = split
        self.transform = transform    
        self.n_sampled_frames = 10

    def __len__(self):
        return len(self.video_paths)
    
    def _get_meta(self, attr, value):
        return self.df.loc[self.df[attr] == value]

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        video_name = video_path.split('/')[-1].split('.avi')[0]
        video_meta = self._get_meta('video_name', video_name)
        label = video_meta['label'].item()

        video_flows_dir = self.video_paths[idx].split('.avi')[0].replace('videos', 'flows')
        flows = self.load_flows(video_flows_dir)

        return flows, label

    def load_flows(self, flows_dir):
        flows = []

        for i in range(1, self.n_sampled_frames):
            flow_file = os.path.join(flows_dir, f"flow_{i}_{i+1}.npy")
            flow = np.load(flow_file)

            flow = torch.from_numpy(flow).float()

            if self.transform:
                flow = self.transform(flow)

            flows.append(flow)

        flows = torch.stack(flows)
        flows = flows.flatten(0, 1)
        
        return flows
