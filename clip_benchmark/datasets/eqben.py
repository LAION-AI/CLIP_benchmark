import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image
import torch
import json
import requests
from subprocess import call

class EQBen(Dataset):

    def __init__(self, root=".", transform=None, download=True):        
        self.transform = transform        
        self.root = root
        if download:
            self.download()
        self.ann =  json.load(open(os.path.join(root, "eqben.json")))
        self.image_folder = os.path.join(root, "image_subset")
        self.ann = [
            {
                "image0": os.path.join(self.image_folder, ann["image0"]),
                "caption0": ann["caption0"],
                "image1": os.path.join(self.image_folder, ann["image1"]),
                "caption1": ann["caption1"]
            } for ann in self.ann
            
        ]
        
    
    def download(self):
        root = self.root
        images_gdrive_id = "13Iuirsvx34-9F_1Mjhs4Dqn59yokyUjy"
        ann_gdrive_id = "18BSRf1SnBtGiEc42mzRLirXaBLzYE5Tt"
        os.makedirs(root, exist_ok=True)
        if not os.path.exists(os.path.join(root, "eqben.json")):
            call(f"gdown --id {ann_gdrive_id} -O {root}/eqben.json", shell=True)
        if not os.path.exists(os.path.join(root, "image_subset")):
            call(f"gdown --id {images_gdrive_id} -O {root}/images.tgz", shell=True)
            call(f"tar xvf {root}/images.tgz -C {root}", shell=True)
        
    
    def __getitem__(self, idx):
        ann = self.ann[idx]
        img1, c1, img2, c2 = ann["image0"], ann["caption0"], ann["image1"], ann["caption1"]
        
        img1 = Image.fromarray(np.load(img1)) if img1.endswith(".npy") else Image.open(img1)
        img2 = Image.fromarray(np.load(img2)) if img2.endswith(".npy") else Image.open(img2)
        
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img = torch.stack([img1, img2])
        else:
            img = [img1, img2]
        match = torch.BoolTensor([
            [True, False],
            [False, True]
        ])
        return img, [c1, c2], match

    def __len__(self):
        return len(self.ann)