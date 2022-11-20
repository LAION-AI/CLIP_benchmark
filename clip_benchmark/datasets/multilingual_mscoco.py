from subprocess import call
import os, json

from torchvision.datasets import VisionDataset
from PIL import Image


GITHUB_MAIN_ORIGINAL_ANNOTATION_PATH = 'https://github.com/mehdidc/retrieval_annotations/releases/download/1.0.0/coco_{}_karpathy.json'
GITHUB_MAIN_PATH = 'https://raw.githubusercontent.com/adobe-research/Cross-lingual-Test-Dataset-XTD10/main/XTD10/'
SUPPORTED_LANGUAGES = ['es', 'it', 'ko', 'pl', 'ru', 'tr', 'zh', 'en']

IMAGE_INDEX_FILE = 'mscoco-multilingual_index.json'
IMAGE_INDEX_FILE_DOWNLOAD_NAME = 'test_image_names.txt'

CAPTIONS_FILE_DOWNLOAD_NAME = 'test_1kcaptions_{}.txt'
CAPTIONS_FILE_NAME = 'multilingual_mscoco_captions-{}.json'

ORIGINAL_ANNOTATION_FILE_NAME = 'coco_{}_karpathy.json'



class Multilingual_MSCOCO(VisionDataset):

    def __init__(self, root, ann_file, transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.ann_file = os.path.expanduser(ann_file)
        with open(ann_file, 'r') as fp:
            data = json.load(fp)
            
        self.data = [(img_path, txt) for img_path, txt in zip(data['image_paths'], data['annotations'])]
    
    def __getitem__(self, index):
        img, captions = self.data[index]

        # Image
        img = Image.open(os.path.join(self.root, img)).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        # Captions
        target =  [captions, ]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


    def __len__(self) -> int:
        return len(self.data)


def _get_downloadable_file(filename, download_url, is_json=True):
    if (os.path.exists(filename) == False):
        print("Downloading", download_url)
        call("wget {} -O {}".format(download_url, filename), shell=True)
    with open(filename, 'r') as fp:
        if (is_json):
            return json.load(fp)
        return [line.strip() for line in fp.readlines()]


def create_english_annotation_file(root):
    print("Downloading multilingual_ms_coco index file")
    download_path = os.path.join(GITHUB_MAIN_PATH, IMAGE_INDEX_FILE_DOWNLOAD_NAME)
    target_images = _get_downloadable_file("multilingual_coco_images.txt", download_path, False)

    print("Downloading multilingual_ms_coco english captions")
    download_path = os.path.join(GITHUB_MAIN_PATH, CAPTIONS_FILE_DOWNLOAD_NAME.format('en'))
    target_captions = _get_downloadable_file("multilingual_coco_en_captions.txt", download_path, False)

    number_of_missing_images = 0
    valid_images, valid_annotations, valid_indicies = [], [], []
    for i, (img, txt) in enumerate(zip(target_images, target_captions)):
        # Create a new file name that includes the root split
        root_split = 'val2014' if 'val' in img else 'train2014'
        filename_with_root_split = "{}/{}".format(root_split, img)

        if (os.path.exists(filename_with_root_split)):
            print("Missing image file", img)
            number_of_missing_images += 1
            continue

        valid_images.append(filename_with_root_split)
        valid_annotations.append(txt)
        valid_indicies.append(i)

    if (number_of_missing_images > 0):
        print("*** WARNING *** missing {} files.".format(number_of_missing_images))

    with open(os.path.join(root, IMAGE_INDEX_FILE), 'w') as fp:
        json.dump({'image_paths': valid_images, 'annotations': valid_annotations, 'indicies': valid_indicies}, fp)

def create_translation_file(index_data, root, lang_code):
    print("Downloading ms_coco translation file", lang_code)

    download_file_name = CAPTIONS_FILE_DOWNLOAD_NAME.format(lang_code)
    download_path = os.path.join(GITHUB_MAIN_PATH, download_file_name)
    new_captions = _get_downloadable_file("multilingual_coco_{}_captions.txt".format(lang_code), download_path, False)
    eng_captions = index_data['annotations']
    
    translation_mapping = {eng_captions[i]: new_captions[i] for i in index_data['indicies']}
    with open(os.path.join(root, CAPTIONS_FILE_NAME.format(lang_code)), 'w') as fp:
        json.dump(translation_mapping, fp)

