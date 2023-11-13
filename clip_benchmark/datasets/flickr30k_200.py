import codecs
import json
import os

import requests
from datasets import load_dataset
from PIL import Image
from torchvision.datasets import VisionDataset

from .flores_langs import flores_languages

GITHUB_DATA_PATH = (
    "https://raw.githubusercontent.com/visheratin/nllb-clip/main/data/flickr30k-200/"
)
SUPPORTED_LANGUAGES = flores_languages

CAPTIONS_FILENAME_TEMPLATE = "{}.txt"
OUTPUT_FILENAME_TEMPLATE = "flickr30k_200-{}.json"


class Flickr30k_200(VisionDataset):
    def __init__(self, root, ann_file, transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.ann_file = os.path.expanduser(ann_file)
        with codecs.open(ann_file, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        self.data = [
            (img_path, txt)
            for img_path, txt in zip(data["image_paths"], data["annotations"])
        ]

    def __getitem__(self, index):
        img, captions = self.data[index]

        # Image
        img = Image.open(os.path.join(self.root, img)).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        # Captions
        target = [
            captions,
        ]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)


def _get_lines(url):
    response = requests.get(url, timeout=30)
    return response.text.splitlines()


def create_annotation_file(root, lang_code):
    if lang_code not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Language code {lang_code} not supported. Supported languages are {SUPPORTED_LANGUAGES}"
        )
    images_dir = os.path.join(root, "flickr30k-200", "images")
    if not os.path.exists(images_dir):
        os.makedirs(images_dir, exist_ok=True)
        dataset = load_dataset("nlphuji/flickr30k")
        for item in dataset["test"]:
            if item["split"] != "test":
                continue
            image=item["image"]
            filename = item["filename"]
            image.save(os.path.join(images_dir, filename))
    
    target_images = []
    image_files = os.listdir(images_dir)
    for item in image_files:
        target_images.append(item)

    print("Downloading flickr30k-200 captions:", lang_code)
    captions_path = GITHUB_DATA_PATH
    download_path = os.path.join(
        captions_path, CAPTIONS_FILENAME_TEMPLATE.format(lang_code)
    )
    target_captions = _get_lines(download_path)

    number_of_missing_images = 0
    valid_images, valid_annotations, valid_indicies = [], [], []
    for i, (img, txt) in enumerate(zip(target_images, target_captions)):
        image_path = os.path.join(images_dir, img)
        if not os.path.exists(image_path):
            print("Missing image file", img)
            number_of_missing_images += 1
            continue

        valid_images.append(image_path)
        valid_annotations.append(txt)
        valid_indicies.append(i)

    if number_of_missing_images > 0:
        print(f"*** WARNING *** missing {number_of_missing_images} files.")

    with codecs.open(
        os.path.join(root, OUTPUT_FILENAME_TEMPLATE.format(lang_code)),
        "w",
        encoding="utf-8",
    ) as fp:
        json.dump(
            {
                "image_paths": valid_images,
                "annotations": valid_annotations,
                "indicies": valid_indicies,
            },
            fp,
            ensure_ascii=False,
        )
