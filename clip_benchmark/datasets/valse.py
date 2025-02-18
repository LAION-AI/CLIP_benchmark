import os
from torch.utils.data import Dataset
from PIL import Image
import torch
import json

class VALSE(Dataset):

    def __init__(self, task="counting_hard", root=".", transform=None):
        self.transform = transform
        self.task = task
        self.root = root
        self.ann = list(json.load(open(os.path.join(root, task.replace("_", "-") + ".json"))).values())

    def __getitem__(self, idx):
        data = self.ann[idx]        
        img_path = os.path.join(self.root, data["dataset"], data["image_file"])
        img = Image.open(img_path)
        caption = data["caption"]
        foil = data["foil"]
        if self.transform is not None:
            img = self.transform(img)
        return img, [caption, foil], torch.BoolTensor([[True, False]])

    def __len__(self):
        return len(self.ann)