import os
from torch.utils.data import Dataset
from PIL import Image
import torch
import json

class Colorswap(Dataset):

    def __init__(self, root=".", transform=None):
        from datasets import load_dataset
        self.ds = load_dataset("stanfordnlp/colorswap", cache_dir=root)["test"]
        self.transform = transform

    def __getitem__(self, idx):
        data = self.ds[idx]
        img0 = data["image_1"]
        img1 = data["image_2"]
        cap0 = data["caption_1"]
        cap1 = data["caption_2"]
        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
            imgs = torch.stack([img0, img1])
        else:
            imgs = [img0, img1]
        caps = [cap0, cap1]
        match = torch.BoolTensor([
            [True, False],
            [False, True]
        ])
        return imgs, caps, match

    def __len__(self):
        return len(self.ds)