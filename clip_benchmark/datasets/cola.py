import os
from torch.utils.data import Dataset
from PIL import Image
import torch
import json
import requests

class COLA(Dataset):

    def __init__(self, ann="COLA_multiobjects_matching_benchmark.json", root=".", transform=None):        
        self.transform = transform        
        self.root = root
        self.ann =  json.load(open(ann))
        self.download()
    
    def download(self):
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