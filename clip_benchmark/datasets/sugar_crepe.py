import os
from torch.utils.data import Dataset
from PIL import Image
import json
class SugarCrepe(Dataset):

    def __init__(self,  root, ann_file, transform=None):
        self.root = root
        self.ann = json.load(open(ann_file))
        self.transform = transform
        self.idx_strings = list(self.ann.keys()) # NOTE : indices may be non-contiguous

    def __getitem__(self, idx):
        idx_str = self.idx_strings[idx]
        data = self.ann[idx_str]
        img = Image.open(os.path.join(self.root, data['filename']))
        if self.transform is not None:
            img = self.transform(img)
        caption = data['caption']
        negative_caption = data['negative_caption']
        return img, [caption, negative_caption]

    def __len__(self):
        return len(self.ann)