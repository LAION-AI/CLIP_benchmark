import os
from torch.utils.data import Dataset
from PIL import Image
import torch
import json

class BLADataset(Dataset):

    def __init__(self, task="active_passive_captions", root=".", transform=None):
        self.transform = transform
        self.task = task
        self.root = root
        self.ann = json.load(open(os.path.join(root, "BLA_benchmark", "annotations", f"{task}.json")))

    def __getitem__(self, idx):
        data = self.ann[idx]
        img_id = data["image_id"]
        img_path = os.path.join(self.root, "BLA_benchmark", "images", f"{img_id}.jpg")
        img = Image.open(img_path)
        true1 = data["caption_group"][0]["True1"]
        true2 = data["caption_group"][0]["True2"]
        false1 = data["caption_group"][0]["False1"]
        false2 = data["caption_group"][0]["False2"]
        if self.transform is not None:
            img = self.transform(img)
        caps = [true1, true2, false1, false2]
        match = torch.BoolTensor(
            [[True, True, False, False]])
        return img, caps, match

    def __len__(self):
        return len(self.ann)