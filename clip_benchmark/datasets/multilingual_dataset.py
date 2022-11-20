from clip_benchmark.datasets import multilingual_mscoco
import json, os
import torch


class Multilingual_Dataset(torch.utils.data.Dataset):

    def __init__(self, original_dataset, new_captions):
        self.original_dataset = original_dataset
        self.new_captions = new_captions

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, item):
        img, txt = self.original_dataset[item]
        if type(txt) == list:
            new_txt = [self.new_captions[t] for t in txt]
        else:
            new_txt = self.new_captions[txt]

        return img, new_txt


def _create_dataset_from_path(original_dataset, translations_path):
    with open(translations_path, 'r') as fp:
        data = json.load(fp)

    return Multilingual_Dataset(original_dataset, data)


def create_dataset_from_language(original_dataset, dataset_name, lang_code, root='root'):
    file_path = os.path.join(root, "{}-{}.json".format(dataset_name, lang_code))

    if (dataset_name == 'multilingual_mscoco_captions'):
        if (lang_code not in multilingual_mscoco.SUPPORTED_LANGUAGES):
            raise ValueError("mscoco_captions does not support language", lang_code)

        image_index_file = os.path.join(root, multilingual_mscoco.IMAGE_INDEX_FILE)
        if (os.path.exists(image_index_file) == False):
            multilingual_mscoco.create_english_annotation_file(root)
        with open(image_index_file, 'r') as fp:
            index_data = json.load(fp)

        language_file = os.path.join(root, multilingual_mscoco.CAPTIONS_FILE_NAME.format(lang_code))
        if (os.path.exists(language_file) == False):
            multilingual_mscoco.create_translation_file(index_data, root, lang_code)

    else:
        raise ValueError(f"Unsupported dataset and language combination: {dataset_name} - {lang_code}.")

    return _create_dataset_from_path(original_dataset, file_path)
