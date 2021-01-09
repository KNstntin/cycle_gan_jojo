import os
from PIL import Image
import torch
import torchvision

def de_norm(tensor):
    return (tensor + 1)/2

class FolderDataSet(torch.utils.data.Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        self.imgs = sorted(os.listdir(main_dir))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return self.transform(Image.open(os.path.join(self.main_dir, self.imgs[idx])).convert('RGB'))