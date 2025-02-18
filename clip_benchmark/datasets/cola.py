import os
from torch.utils.data import Dataset
from PIL import Image
import torch
import json
import requests
from subprocess import call

class COLA(Dataset):

    def __init__(self, root=".", transform=None, download=True):        
        self.transform = transform        
        self.root = root
        if download:
            self.download()
        self.ann =  json.load(open(os.path.join(root, "COLA_multiobjects_matching_benchmark.json")))
    
    def download(self):
        root = self.root
        os.makedirs(root, exist_ok=True)
        url = "https://raw.githubusercontent.com/arijitray1993/COLA/refs/heads/main/data/COLA_multiobjects_matching_benchmark.json"
        if not os.path.exists(os.path.join(root, "COLA_multiobjects_matching_benchmark.json")):
            call(f"wget '{url}' --output-document={root}/COLA_multiobjects_matching_benchmark.json", shell=True)
        for im1, c2, im2, c2 in self.ann:
            for im in [im1, im2]:
                path = os.path.join(self.root, os.path.basename(im))
                if os.path.exists(path):
                    continue
                data = requests.get(im, stream=True).raw
                with open(path, "wb") as fd:
                    for chunk in data:
                        fd.write(chunk)
                

    def __getitem__(self, idx):
        im1, c1, im2, c2 = self.ann[idx]
        url1, url2 = im1, im2       
        img1 = Image.open(os.path.join(self.root, os.path.basename(url1)))
        img2 = Image.open(os.path.join(self.root, os.path.basename(url2)))
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