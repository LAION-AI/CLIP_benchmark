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


def _create_dataset_from_language(original_dataset, dataset_name, lang_code, root='root'):
    file_path = os.path.join(root, "{}-{}.json".format(dataset_name, lang_code))
    return _create_dataset_from_path(original_dataset, file_path)
