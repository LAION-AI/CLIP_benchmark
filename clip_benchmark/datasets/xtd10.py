import codecs
import json
import os
from subprocess import call

import requests
from PIL import Image
from torchvision.datasets import VisionDataset


GITHUB_DATA_PATH = "https://raw.githubusercontent.com/adobe-research/Cross-lingual-Test-Dataset-XTD10/main/XTD10"
GITHUB_MIC_DATA_PATH = "https://raw.githubusercontent.com/adobe-research/Cross-lingual-Test-Dataset-XTD10/main/MIC"
GITHUB_STAIR_DATA_PATH = "https://raw.githubusercontent.com/adobe-research/Cross-lingual-Test-Dataset-XTD10/main/STAIR"
SUPPORTED_LANGUAGES = [
    "de",
    "en",
    "es",
    "fr",
    "it",
    "jp",
    "ko",
    "pl",
    "ru",
    "tr",
    "zh",
]

IMAGE_INDEX_FILENAME = "test_image_names.txt"

CAPTIONS_FILENAME_TEMPLATE = "test_1kcaptions_{}.txt"
OUTPUT_FILENAME_TEMPLATE = "xtd10-{}.json"

IMAGES_DOWNLOAD_URL = "https://nllb-data.com/test/xtd10/images.tar.gz"


class XTD10(VisionDataset):
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
        img = Image.open(img).convert("RGB")
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


def _download_images(out_path):
    os.makedirs(out_path, exist_ok=True)
    print("Downloading images")
    call(f"wget {IMAGES_DOWNLOAD_URL} -O images.tar.gz", shell=True)
    call(f"tar -xzf images.tar.gz -C {out_path}", shell=True)
    call("rm images.tar.gz", shell=True)


def create_annotation_file(root, lang_code):
    if lang_code not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Language code {lang_code} not supported. Supported languages are {SUPPORTED_LANGUAGES}"
        )
    data_dir = os.path.join(root, "xtd10")
    if not os.path.exists(data_dir):
        _download_images(data_dir)
    images_dir = os.path.join(data_dir, "images")
    print("Downloading xtd10 index file")
    download_path = os.path.join(GITHUB_DATA_PATH, IMAGE_INDEX_FILENAME)
    target_images = _get_lines(download_path)

    print("Downloading xtd10 captions:", lang_code)
    captions_path = GITHUB_DATA_PATH
    match lang_code:
        case "de" | "fr":
            captions_path = GITHUB_MIC_DATA_PATH
        case "jp":
            captions_path = GITHUB_STAIR_DATA_PATH
        case _:
            captions_path = GITHUB_DATA_PATH
    download_path = os.path.join(
        captions_path, CAPTIONS_FILENAME_TEMPLATE.format(lang_code)
    )
    target_captions = _get_lines(download_path)

    number_of_missing_images = 0
    valid_images, valid_annotations, valid_indices = [], [], []
    for i, (img, txt) in enumerate(zip(target_images, target_captions)):
        image_path = os.path.join(images_dir, img)
        if not os.path.exists(image_path):
            print("Missing image file", img)
            number_of_missing_images += 1
            continue

        valid_images.append(image_path)
        valid_annotations.append(txt)
        valid_indices.append(i)

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
                "indices": valid_indices,
            },
            fp,
            ensure_ascii=False,
        )
