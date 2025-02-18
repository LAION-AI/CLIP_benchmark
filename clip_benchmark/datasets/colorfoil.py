# Thanks to https://github.com/samin9796/ColorFoil/blob/main/zero_shot_evaluation.ipynb
import json
import webcolors
import random
import os
import os
from torch.utils.data import Dataset
from PIL import Image
import torch
import json

class ColorFoil(Dataset):
    
    def __init__(self, image_folder=".", ann="captions_val2017.json", transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.data = self.prepare_data(image_folder, ann)
    
    def prepare_data(self, image_folder, ann):
        img_list = [] # list of image urls
        cap_list = [] # list of captions
        foil_list = [] # list of foiled captions

        with open(ann) as f: # read the data
            d = json.load(f)
            img_length = len(d["images"]) # total number of images in the MS COCO val set (2017)
            # create three lists of images, captions and foils
            rows = []
            for i in d["images"]:
                id = i["id"]
                flag = False
                for j in d["annotations"]:
                    if j["image_id"]==id:
                        caption = j["caption"]
                        for word in caption.split(' '):
                            if word in webcolors.CSS3_NAMES_TO_HEX: # using the webcolor python package
                                flag = True
                            if flag == True:
                                foil = create_foil(caption) # call the create_foil function. it will randomly choose a foil color.
                                img_list.append(i["coco_url"])
                                cap_list.append(caption)
                                foil_list.append(foil)
                                path = os.path.join(image_folder, os.path.basename(i["coco_url"]))
                                assert os.path.exists(path)
                                rows.append({'image': path, 'caption': caption, 'foil': foil})
                                break
        return rows

    def __getitem__(self, idx):
        data = self.data[idx]
        img = Image.open(data['image'])
        if self.transform is not None:
            img = self.transform(img)
        caption = data['caption']
        foil = data['foil']
        match = torch.BoolTensor([[True, False]])
        return img, [caption, foil], match

    def __len__(self):
        return len(self.data)


def create_foil(caption):
  # most commonly used colors
  colors = ["blue", "black", "red", "pink", "yellow", "grey", "orange", "white", "green", "brown"]
  lst = caption.split(' ')
  for color in lst:
    if color in webcolors.CSS3_NAMES_TO_HEX:
      num = random.randint(0, 9)
      if colors[num] == color:
        foiling_color = colors[num-1]
      else: foiling_color = colors[num]
      caption = caption.replace(color, foiling_color)
  return caption