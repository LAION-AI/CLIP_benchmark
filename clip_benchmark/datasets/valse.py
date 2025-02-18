import os
from torch.utils.data import Dataset
from PIL import Image
import torch
import json
from subprocess import call

class VALSE(Dataset):

    def __init__(self, task="counting_hard", root=".", transform=None, download=True):
        self.transform = transform
        self.task = task
        self.root = root
        if download:
            self.download()
        self.ann = list(json.load(open(os.path.join(root, task.replace("_", "-") + ".json"))).values())

    def download(self):
        task = self.task
        root = self.root
        tasks = [
            "actant_swap",
            "coreference_hard",
            "counting_adversarial",
            "counting_small_quant",
            "foil_it",
            "relations",
            "action_replacement",
            "coreference_standard",
            "counting_hard",
            "existence",
            "plurals"
        ]
        assert task in tasks
        task_path = task.replace("_", "-") + ".json"

        url = f"https://raw.githubusercontent.com/Heidelberg-NLP/VALSE/refs/heads/main/data/{task_path}"
        
        os.makedirs(root, exist_ok=True)
        if not os.path.exists(os.path.join(root, task_path)):
            call(f"wget '{url}' --output-document={root}/{task_path}", shell=True)
        if not os.path.exists(os.path.join(root, "visual7w")):
            url = "http://vision.stanford.edu/yukezhu/visual7w_images.zip"
            call(f"wget '{url}' --output-document={root}/visual7w.zip", shell=True)
            call(f"unzip {root}/visual7w.zip -d {root}", shell=True)
            call(f"mv {root}/images {root}/visual7w", shell=True)
        if not os.path.exists(os.path.join(root, "SWiG")):
            url = "https://swig-data-weights.s3.us-east-2.amazonaws.com/images_512.zip"
            call(f"wget '{url}' --output-document={root}/swig.zip", shell=True)
            call(f"unzip {root}/swig.zip -d {root}", shell=True)
            call(f"mv {root}/images_512 {root}/SWiG", shell=True)
        if not os.path.exists(os.path.join(root, "VisDial_v1.0")):
            url = "https://www.dropbox.com/s/twmtutniktom7tu/VisualDialog_val2018.zip?dl=1"
            call(f"wget '{url}' --output-document={root}/visdial.zip", shell=True)
            call(f"unzip {root}/visdial.zip -d {root}", shell=True)
            call(f"mv {root}/VisualDialog_val2018 {root}/VisDial_v1.0", shell=True)
        if not os.path.exists(os.path.join(root, "FOIL dataset")):
            call(f"wget http://images.cocodataset.org/zips/val2014.zip --output-document={root}/coco.zip", shell=True)
            call(f"unzip {root}/coco.zip -d {root}", shell=True)
            call(f"mv {root}/val2014 '{root}/FOIL dataset'", shell=True)
        if not os.path.exists(os.path.join(root, "coco_2017")):
            call(f"wget http://images.cocodataset.org/zips/val2017.zip --output-document={root}/coco2017.zip", shell=True)
            call(f"unzip {root}/coco2017.zip -d {root}", shell=True)
            call(f"mv {root}/val2017 '{root}/coco_2017'", shell=True)
        
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