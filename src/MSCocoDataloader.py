from PIL import Image
import torch
import glob

class MSCocoDataloader(torch.utils.data.Dataset):
    def __init__(self, root_dir='/DATA/dataset/mscoco/'):
        self.root_dir = root_dir
        self.labels = glob.glob(self.root_dir)
    
    def __len__(self):
        return len(len(self.labels))

    def __getitem__(self, idx):
        return Image.open(self.labels[idx])
    
    def get_image_name(self, idx):
        return self.labels[idx]
