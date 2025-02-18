# Thanks to https://github.com/mertyg/vision-language-models-are-bows

import os
import json
import re
import subprocess

import numpy as np

from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
from easydict import EasyDict as edict
from torchvision.datasets.utils import download_url
import torch
import random
import numpy as np
from functools import partial
import torch.nn.functional as nnf
from torchvision import transforms as T


def pre_caption(caption,max_words=50):
    caption = re.sub(
        r"([.!\"()*#:;~])",       
        ' ',
        caption.lower(),
    )
    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')

    #truncate caption
    caption_words = caption.split(' ')
    if len(caption_words)>max_words:
        caption = ' '.join(caption_words[:max_words])
    
    return caption
# A lot of the approaches here are inspired from the wonderful paper from O'Connor and Andreas 2021.
# https://github.com/lingo-mit/context-ablations

  


class TextShuffler:

    def __init__(self):
        import spacy
        self.nlp = spacy.load("en_core_web_sm")

    def shuffle_nouns_and_adj(self, ex):

        doc = self.nlp(ex)
        tokens = [token.text for token in doc]
        text = np.array(tokens)
        noun_idx = [i for i, token in enumerate(doc) if token.tag_ in ['NN', 'NNS', 'NNP', 'NNPS']]
        ## Finding adjectives
        adjective_idx = [i for i, token in enumerate(doc) if token.tag_ in ['JJ', 'JJR', 'JJS']]
        ## Shuffle the nouns of the text
        text[noun_idx] = np.random.permutation(text[noun_idx])
        ## Shuffle the adjectives of the text
        text[adjective_idx] = np.random.permutation(text[adjective_idx])

        return " ".join(text)

    def shuffle_all_words(self, ex):
        return " ".join(np.random.permutation(ex.split(" ")))


    def shuffle_allbut_nouns_and_adj(self, ex):
        doc = self.nlp(ex)
        tokens = [token.text for token in doc]
        text = np.array(tokens)
        noun_adj_idx = [i for i, token in enumerate(doc) if token.tag_ in ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS']]
        ## Finding adjectives

        else_idx = np.ones(text.shape[0])
        else_idx[noun_adj_idx] = 0

        else_idx = else_idx.astype(bool)
        ## Shuffle everything that are nouns or adjectives
        text[else_idx] = np.random.permutation(text[else_idx])
        return " ".join(text)


    def get_trigrams(self, sentence):
        # Taken from https://github.com/lingo-mit/context-ablations/blob/478fb18a9f9680321f0d37dc999ea444e9287cc0/code/transformers/src/transformers/data/data_augmentation.py
        trigrams = []
        trigram = []
        for i in range(len(sentence)):
            trigram.append(sentence[i])
            if i % 3 == 2:
                trigrams.append(trigram[:])
                trigram = []
        if trigram:
            trigrams.append(trigram)
        return trigrams

    def trigram_shuffle(self, sentence):
        trigrams = self.get_trigrams(sentence)
        for trigram in trigrams:
            random.shuffle(trigram)
        return " ".join([" ".join(trigram) for trigram in trigrams])


    def shuffle_within_trigrams(self, ex):
        import nltk
        tokens = nltk.word_tokenize(ex)
        shuffled_ex = self.trigram_shuffle(tokens)
        return shuffled_ex


    def shuffle_trigrams(self, ex):
        import nltk
        tokens = nltk.word_tokenize(ex)
        trigrams = self.get_trigrams(tokens)
        random.shuffle(trigrams)
        shuffled_ex = " ".join([" ".join(trigram) for trigram in trigrams])
        return shuffled_ex


def _handle_image_4shuffle(x):
    return_image = False
    if not isinstance(x, torch.Tensor):
        # print(f"x is not a tensor: {type(x)}. Trying to handle but fix this or I'll annoy you with this log")
        t = torch.tensor(np.array(x)).unsqueeze(dim=0).float()
        t = t.permute(0, 3, 1, 2)
        return_image = True
        return t, return_image
    if len(x.shape) != 4:
        #print("You did not send a tensor of shape NxCxWxH. Unsqueezing not but fix this or I'll annoy you with this log")
        return x.unsqueeze(dim=0), return_image
    else:
        # Good boi
        return x, return_image
        

def shuffle_rows(x, n_rows=7):
    """
    Shuffle the rows of the image tensor where each row has a size of 14 pixels.
    Tensor is of shape N x C x W x H
    """
    x, return_image = _handle_image_4shuffle(x)
    patch_size = x.shape[-2]//n_rows
    u = nnf.unfold(x, kernel_size=(patch_size, x.shape[-1]), stride=patch_size, padding=0)
    # permute the patches of each image in the batch
    pu = torch.cat([b_[:, torch.randperm(b_.shape[-1])][None,...] for b_ in u], dim=0)
    # fold the permuted patches back together
    f = nnf.fold(pu, x.shape[-2:], kernel_size=(patch_size, x.shape[-1]), stride=patch_size, padding=0)
    
    image = f.squeeze() # C W H
    if return_image:
        return T.ToPILImage()(image.type(torch.uint8))
    else:
        return image


def shuffle_columns(x, n_cols=7):
    """
    Shuffle the columns of the image tensor where we'll have n_cols columns.
    Tensor is of shape N x C x W x H
    """
    x, return_image = _handle_image_4shuffle(x)
    patch_size = x.shape[-1]//n_cols
    u = nnf.unfold(x, kernel_size=(x.shape[-2], patch_size), stride=patch_size, padding=0)
    # permute the patches of each image in the batch
    pu = torch.cat([b_[:, torch.randperm(b_.shape[-1])][None,...] for b_ in u], dim=0)
    # fold the permuted patches back together
    f = nnf.fold(pu, x.shape[-2:], kernel_size=(x.shape[-2], patch_size), stride=patch_size, padding=0)
    image = f.squeeze() # C W H
    if return_image:
        return T.ToPILImage()(image.type(torch.uint8))
    else:
        return image



def shuffle_patches(x, n_ratio=4):
    """
    Shuffle the rows of the image tensor where each row has a size of 14 pixels.
    Tensor is of shape N x C x W x H
    """
    x, return_image = _handle_image_4shuffle(x)
    patch_size_x = x.shape[-2]//n_ratio
    patch_size_y = x.shape[-1]//n_ratio
    u = nnf.unfold(x, kernel_size=(patch_size_x, patch_size_y), stride=(patch_size_x, patch_size_y), padding=0)
    # permute the patches of each image in the batch
    pu = torch.cat([b_[:, torch.randperm(b_.shape[-1])][None,...] for b_ in u], dim=0)
    # fold the permuted patches back together
    f = nnf.fold(pu, x.shape[-2:], kernel_size=(patch_size_x, patch_size_y), stride=(patch_size_x, patch_size_y), padding=0)
    image = f.squeeze() # C W H
    if return_image:
        return T.ToPILImage()(image.type(torch.uint8))
    else:
        return image

def get_text_perturb_fn(text_perturb_fn):
    if text_perturb_fn == "shuffle_nouns_and_adj":
        return shuffle_nouns_and_adj
    elif text_perturb_fn == "shuffle_allbut_nouns_and_adj":
        return shuffle_allbut_nouns_and_adj
    elif text_perturb_fn == "shuffle_within_trigrams":
        return shuffle_within_trigrams
    elif text_perturb_fn == "shuffle_all_words":
        return shuffle_all_words
    elif text_perturb_fn == "shuffle_trigrams":
        return shuffle_trigrams
    elif text_perturb_fn is None:
        return None
    else:
        print("Unknown text perturbation function: {}, returning None".format(text_perturb_fn))
        return None
    
    
def get_image_perturb_fn(image_perturb_fn):
    if image_perturb_fn == "shuffle_rows_4":
        return partial(shuffle_rows, n_rows=4)
    elif image_perturb_fn == "shuffle_patches_9":
        return partial(shuffle_patches, n_ratio=3)
    elif image_perturb_fn == "shuffle_cols_4":
        return partial(shuffle_columns, n_cols=4)
    elif image_perturb_fn is None:
        return None
    else:
        print("Unknown image perturbation function: {}, returning None".format(image_perturb_fn))
        return None
    


class TextShuffler:

    def __init__(self):
        import spacy
        self.nlp = spacy.load("en_core_web_sm")

    def shuffle_nouns_and_adj(self, ex):

        doc = self.nlp(ex)
        tokens = [token.text for token in doc]
        text = np.array(tokens)
        noun_idx = [i for i, token in enumerate(doc) if token.tag_ in ['NN', 'NNS', 'NNP', 'NNPS']]
        ## Finding adjectives
        adjective_idx = [i for i, token in enumerate(doc) if token.tag_ in ['JJ', 'JJR', 'JJS']]
        ## Shuffle the nouns of the text
        text[noun_idx] = np.random.permutation(text[noun_idx])
        ## Shuffle the adjectives of the text
        text[adjective_idx] = np.random.permutation(text[adjective_idx])

        return " ".join(text)

    def shuffle_all_words(self, ex):
        return " ".join(np.random.permutation(ex.split(" ")))


    def shuffle_allbut_nouns_and_adj(self, ex):
        doc = self.nlp(ex)
        tokens = [token.text for token in doc]
        text = np.array(tokens)
        noun_adj_idx = [i for i, token in enumerate(doc) if token.tag_ in ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS']]
        ## Finding adjectives

        else_idx = np.ones(text.shape[0])
        else_idx[noun_adj_idx] = 0

        else_idx = else_idx.astype(bool)
        ## Shuffle everything that are nouns or adjectives
        text[else_idx] = np.random.permutation(text[else_idx])
        return " ".join(text)


    def get_trigrams(self, sentence):
        # Taken from https://github.com/lingo-mit/context-ablations/blob/478fb18a9f9680321f0d37dc999ea444e9287cc0/code/transformers/src/transformers/data/data_augmentation.py
        trigrams = []
        trigram = []
        for i in range(len(sentence)):
            trigram.append(sentence[i])
            if i % 3 == 2:
                trigrams.append(trigram[:])
                trigram = []
        if trigram:
            trigrams.append(trigram)
        return trigrams

    def trigram_shuffle(self, sentence):
        trigrams = self.get_trigrams(sentence)
        for trigram in trigrams:
            random.shuffle(trigram)
        return " ".join([" ".join(trigram) for trigram in trigrams])


    def shuffle_within_trigrams(self, ex):
        import nltk
        tokens = nltk.word_tokenize(ex)
        shuffled_ex = self.trigram_shuffle(tokens)
        return shuffled_ex


    def shuffle_trigrams(self, ex):
        import nltk
        tokens = nltk.word_tokenize(ex)
        trigrams = self.get_trigrams(tokens)
        random.shuffle(trigrams)
        shuffled_ex = " ".join([" ".join(trigram) for trigram in trigrams])
        return shuffled_ex


def _handle_image_4shuffle(x):
    return_image = False
    if not isinstance(x, torch.Tensor):
        # print(f"x is not a tensor: {type(x)}. Trying to handle but fix this or I'll annoy you with this log")
        t = torch.tensor(np.array(x)).unsqueeze(dim=0).float()
        t = t.permute(0, 3, 1, 2)
        return_image = True
        return t, return_image
    if len(x.shape) != 4:
        #print("You did not send a tensor of shape NxCxWxH. Unsqueezing not but fix this or I'll annoy you with this log")
        return x.unsqueeze(dim=0), return_image
    else:
        # Good boi
        return x, return_image
        

def shuffle_rows(x, n_rows=7):
    """
    Shuffle the rows of the image tensor where each row has a size of 14 pixels.
    Tensor is of shape N x C x W x H
    """
    x, return_image = _handle_image_4shuffle(x)
    patch_size = x.shape[-2]//n_rows
    u = nnf.unfold(x, kernel_size=(patch_size, x.shape[-1]), stride=patch_size, padding=0)
    # permute the patches of each image in the batch
    pu = torch.cat([b_[:, torch.randperm(b_.shape[-1])][None,...] for b_ in u], dim=0)
    # fold the permuted patches back together
    f = nnf.fold(pu, x.shape[-2:], kernel_size=(patch_size, x.shape[-1]), stride=patch_size, padding=0)
    
    image = f.squeeze() # C W H
    if return_image:
        return T.ToPILImage()(image.type(torch.uint8))
    else:
        return image


def shuffle_columns(x, n_cols=7):
    """
    Shuffle the columns of the image tensor where we'll have n_cols columns.
    Tensor is of shape N x C x W x H
    """
    x, return_image = _handle_image_4shuffle(x)
    patch_size = x.shape[-1]//n_cols
    u = nnf.unfold(x, kernel_size=(x.shape[-2], patch_size), stride=patch_size, padding=0)
    # permute the patches of each image in the batch
    pu = torch.cat([b_[:, torch.randperm(b_.shape[-1])][None,...] for b_ in u], dim=0)
    # fold the permuted patches back together
    f = nnf.fold(pu, x.shape[-2:], kernel_size=(x.shape[-2], patch_size), stride=patch_size, padding=0)
    image = f.squeeze() # C W H
    if return_image:
        return T.ToPILImage()(image.type(torch.uint8))
    else:
        return image



def shuffle_patches(x, n_ratio=4):
    """
    Shuffle the rows of the image tensor where each row has a size of 14 pixels.
    Tensor is of shape N x C x W x H
    """
    x, return_image = _handle_image_4shuffle(x)
    patch_size_x = x.shape[-2]//n_ratio
    patch_size_y = x.shape[-1]//n_ratio
    u = nnf.unfold(x, kernel_size=(patch_size_x, patch_size_y), stride=(patch_size_x, patch_size_y), padding=0)
    # permute the patches of each image in the batch
    pu = torch.cat([b_[:, torch.randperm(b_.shape[-1])][None,...] for b_ in u], dim=0)
    # fold the permuted patches back together
    f = nnf.fold(pu, x.shape[-2:], kernel_size=(patch_size_x, patch_size_y), stride=(patch_size_x, patch_size_y), padding=0)
    image = f.squeeze() # C W H
    if return_image:
        return T.ToPILImage()(image.type(torch.uint8))
    else:
        return image



class VG_Relation(Dataset):
    def __init__(self, image_preprocess, text_perturb_fn=None, image_perturb_fn=None, root_dir="root", download=False):
        '''
        image_preprocess: a function that takes in a PIL image and returns a tensor.
        text_perturb_fn: Not used for this dataset. Just for compatibility with other datasets.
        image_perturb_fn: Not used for this dataset. Just for compatibility with other datasets.
        root_dir: Directory for the VG-R dataset.
        download: Whether to download the dataset if it does not exist.
        '''
        self.root_dir = root_dir
        annotation_file = os.path.join(root_dir, "visual_genome_relation.json")
        image_dir = os.path.join(root_dir, "images")
        if not os.path.exists(image_dir):
            print("Image Directory for VG_Relation could not be found!")
            if download:
                self.download()
            else:
                raise RuntimeError("Please either download the dataset by letting `--download` or specify the correct directory.")
        
        if not os.path.exists(annotation_file):
            subprocess.call(["gdown", "--id", "1kX2iCHEv0CADL8dSO1nMdW-V0NqIAiP3", "--output", annotation_file])
        
        with open(annotation_file, "r") as f:
            self.dataset = json.load(f)
        
        self.all_relations = list()
        for item in self.dataset:
            item["image_path"] = os.path.join(image_dir, item["image_path"])
            self.all_relations.append(item["relation_name"])

        self.image_preprocess = image_preprocess

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        test_case = self.dataset[index]
        image = Image.open(test_case["image_path"]).convert('RGB')
        # Get the bounding box that contains the relation. This is to remove the irrelevant details in the scene.
        image = image.crop((test_case["bbox_x"], test_case["bbox_y"], test_case["bbox_x"] + test_case["bbox_w"], test_case["bbox_y"] + test_case["bbox_h"]))

        if self.image_preprocess is not None:
            image = self.image_preprocess(image)

        # Each test case has a correct and incorrect caption.
        true_caption = test_case["true_caption"]
        false_caption = test_case["false_caption"]
        item = edict({"image_options": [image], "caption_options": [false_caption, true_caption]})
        return item
    
    def download(self):
        os.makedirs(self.root_dir, exist_ok=True)
        image_zip_file = os.path.join(self.root_dir, "vgr_vga_images.zip")
        subprocess.call(["gdown", "--no-cookies", "1qaPlrwhGNMrR3a11iopZUT_GPP_LrgP9", "--output", image_zip_file])
        subprocess.call(["unzip", "vgr_vga_images.zip"], cwd=self.root_dir)

        
    def evaluate_scores(self, scores):
        """
        Scores: N x 1 x 2, i.e. first caption is the perturbed one, second is the positive one
        """
        if isinstance(scores, tuple):
            scores_i2t = scores[1]
            scores_t2i = scores[0] 
        else:
            scores_t2i = scores
            scores_i2t = scores

        metrics = {"Accuracy": None}
        preds = np.argmax(np.squeeze(scores_i2t, axis=1), axis=-1)
        correct_mask = (preds == 1)
        metrics["Accuracy"] = np.mean(correct_mask)

        all_relations = np.array(self.all_relations)

        result_records = []
        # Log the accuracy of all relations
        for relation in np.unique(all_relations):
            relation_mask = (all_relations == relation)
            if relation_mask.sum() == 0:
                continue
            result_records.append({
                "Relation": relation,
                "Accuracy": correct_mask[relation_mask].mean(),
                "Count": relation_mask.sum(),
                "Dataset": "Visual Genome Relation"
            })
        return result_records



class VG_Attribution(Dataset):
    def __init__(self, image_preprocess, text_perturb_fn=None, image_perturb_fn=None, root_dir="root", download=False):
        '''
        image_preprocess: a function that takes in a PIL image and returns a tensor.
        text_perturb_fn: Not used for this dataset. Just for compatibility with other datasets.
        image_perturb_fn: Not used for this dataset. Just for compatibility with other datasets.
        root_dir: Directory for the VG-A dataset.
        '''
        self.root_dir = root_dir
        annotation_file = os.path.join(root_dir, "visual_genome_attribution.json")
        image_dir = os.path.join(root_dir, "images")
        if not os.path.exists(image_dir):
            print("Image Directory for VG_Attribution could not be found!")
            if download:
                self.download()
            else:
                raise RuntimeError("Please either download the dataset by letting `--download` or specify the correct directory.")
        
        
        if not os.path.exists(annotation_file):
            subprocess.call(["gdown", "--id", "13tWvOrNOLHxl3Rm9cR3geAdHx2qR3-Tw", "--output", annotation_file])

        with open(annotation_file, "r") as f:
            self.dataset = json.load(f)
        
        for item in self.dataset:
            item["image_path"] = os.path.join(image_dir, item["image_path"])
        
        # Set of attributes in each test case
        self.all_attributes = [f"{item['attributes'][0]}_{item['attributes'][1]}" for item in self.dataset]
        self.image_preprocess = image_preprocess

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        test_case = self.dataset[index]
        image = Image.open(test_case["image_path"]).convert('RGB')
        # Get the bounding box that contains the relation. This is to remove the irrelevant details in the scene.
        image = image.crop((test_case["bbox_x"], test_case["bbox_y"], test_case["bbox_x"] + test_case["bbox_w"], test_case["bbox_y"] + test_case["bbox_h"]))

        if self.image_preprocess is not None:
            image = self.image_preprocess(image)

        # Each test case has a correct and incorrect caption.
        true_caption = test_case["true_caption"]
        false_caption = test_case["false_caption"]
        item = edict({"image_options": [image], "caption_options": [false_caption, true_caption]})
        return item
    
    def download(self):
        os.makedirs(self.root_dir, exist_ok=True)
        image_zip_file = os.path.join(self.root_dir, "vgr_vga_images.zip")
        subprocess.call(["gdown", "--no-cookies",  "1qaPlrwhGNMrR3a11iopZUT_GPP_LrgP9", "--output", image_zip_file])
        subprocess.call(["unzip", "vgr_vga_images.zip"], cwd=self.root_dir)

    
    def evaluate_scores(self, scores):
        """
        Scores: N x 1 x 2, i.e. first caption is the perturbed one, second is the positive one
        """
        if isinstance(scores, tuple):
            scores_i2t = scores[1]
            scores_t2i = scores[0] 
        else:
            scores_t2i = scores
            scores_i2t = scores

        preds = np.argmax(np.squeeze(scores_i2t, axis=1), axis=-1)
        correct_mask = (preds == 1)
        result_records = []
        all_attributes = np.array(self.all_attributes)
        for attr in np.unique(all_attributes):
            attr_mask = (all_attributes == attr)
            if attr_mask.sum() < 25:
                continue
            result_records.append({
                "Attributes": attr,
                "Accuracy": correct_mask[attr_mask].mean(),
                "Count": attr_mask.sum(),
                "Dataset": "Visual Genome Attribution"
            })
        return result_records




class COCO_Order(Dataset):
    def __init__(self, image_preprocess=None, root_dir="root", max_words=30, split="test",
                 image_perturb_fn=None, download=False):  
        """
        COCO Order Dataset.
        image_preprocess: image preprocessing function
        root_dir: The directory of the coco dataset. This directory should contain test2014 files.
        max_words: Cropping the caption to max_words.
        split: 'val' or 'test'
        image_perturb_fn: not used; for compatibility.
        download: Whether to download the dataset if it does not exist.
        """
        super().__init__()
        shuffler = TextShuffler()
        perturb_functions = [shuffler.shuffle_nouns_and_adj, shuffler.shuffle_allbut_nouns_and_adj,
                             shuffler.shuffle_within_trigrams, shuffler.shuffle_trigrams]

        self.root_dir = root_dir
        if not os.path.exists(root_dir):
            print("Directory for COCO could not be found!")
            if download:
                print("Downloading COCO now.")
                self.download()
            else:
                raise RuntimeError("Please either download the dataset by letting `--download` or specify the correct directory.")
        
        urls = {'val':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val.json',
                'test':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test.json'}
        filenames = {'val':'coco_karpathy_val.json','test':'coco_karpathy_test.json'}
        download_url(urls[split],root_dir)
        
        self.annotation = json.load(open(os.path.join(root_dir,filenames[split]),'r'))
        self.image_preprocess = image_preprocess
        self.image_root = root_dir
        
        self.test_cases = []
        
        for img_id, ann in tqdm(enumerate(self.annotation)):
            for i, caption in enumerate(ann['caption']):
                test_case = {}
                test_case["image"] = ann["image"]
                test_case["caption_options"] = [pre_caption(caption,max_words)]

                for perturb_fn in perturb_functions:
                    test_case["caption_options"].append(pre_caption(perturb_fn(caption), max_words))
                self.test_cases.append(test_case)
                                    
    def __len__(self):
        return len(self.test_cases)
    
    def __getitem__(self, index):  
        test_case = self.test_cases[index]  
        image_path = os.path.join(self.image_root, test_case["image"])       
         
        image = Image.open(image_path).convert('RGB')    
        if self.image_preprocess is not None: 
            image = self.image_preprocess(image)  
        
        #item = edict({"image_options": [image], "caption_options": test_case["caption_options"]})
        #return item
        return image, test_case["caption_options"], torch.BoolTensor([[True] + [False] * len(test_case["caption_options"][1:])])
    
    def download(self):
        import subprocess
        os.makedirs(self.root_dir, exist_ok=True)
        #subprocess.call(["wget", "http://images.cocodataset.org/zips/train2014.zip"], cwd=self.root_dir)
        #subprocess.call(["unzip", "train2014.zip"], cwd=self.root_dir)
        
        subprocess.call(["wget", "http://images.cocodataset.org/zips/val2014.zip"], cwd=self.root_dir)
        subprocess.call(["unzip", "val2014.zip"], cwd=self.root_dir)
        
        subprocess.call(["wget", "http://images.cocodataset.org/zips/test2014.zip"], cwd=self.root_dir)
        subprocess.call(["unzip", "test2014.zip"], cwd=self.root_dir)
        
    
    def evaluate_scores(self, scores):
        if isinstance(scores, tuple):
            scores_i2t = scores[0]
            scores_t2i = scores[1].T # Make it N_ims x N_text
        
        else:
            scores_t2i = scores
            scores_i2t = scores
        
        preds = np.argmax(np.squeeze(scores_i2t, axis=1), axis=-1)
        correct_mask = (preds == 0)
        records = [{"Precision@1": np.mean(correct_mask)}]
        return records


class Flickr30k_Order(Dataset):
    def __init__(self, image_preprocess, split, root_dir="root", max_words=30,
                 *args, **kwargs):  
        """
        image_preprocess: image preprocessing function
        split: 'val' or 'test'
        root_dir: The directory of the flickr30k images. This should contain the `flickr30k-images` directory that \
            contains all the images. 
        """
        
        urls = {'val':'https://storage.googleapis.com/sfr-vision-language-research/datasets/flickr30k_val.json',
                'test':'https://storage.googleapis.com/sfr-vision-language-research/datasets/flickr30k_test.json'}
        filenames = {'val':'flickr30k_val.json','test':'flickr30k_test.json'}
        if not os.path.exists(root_dir):
            print("Directory for Flickr30k could not be found!")
            flickr_url = "https://forms.illinois.edu/sec/229675"
            raise RuntimeError(f"You need to manually sign up and download the dataset from {flickr_url} and place it in the `root_dir`.")
        
        download_url(urls[split],root_dir)
        
        self.annotation = json.load(open(os.path.join(root_dir,filenames[split]),'r'))
        self.image_preprocess = image_preprocess
        self.root_dir = root_dir
        
        self.test_cases = []
        
        shuffler = TextShuffler()
        perturb_functions = [shuffler.shuffle_nouns_and_adj, shuffler.shuffle_allbut_nouns_and_adj,
                             shuffler.shuffle_within_trigrams, shuffler.shuffle_trigrams]
        for img_id, ann in tqdm(enumerate(self.annotation)):
            for i, caption in enumerate(ann['caption']):
                test_case = {}
                test_case["image"] = ann["image"]
                test_case["caption_options"] = [pre_caption(caption,max_words)]

                for perturb_fn in perturb_functions:
                    test_case["caption_options"].append(pre_caption(perturb_fn(caption), max_words))
                self.test_cases.append(test_case)
                                
    def __len__(self):
        return len(self.test_cases)
    
    def __getitem__(self, index):  
        test_case = self.test_cases[index]  
        image_path = os.path.join(self.root_dir, test_case["image"])        
        image = Image.open(image_path).convert('RGB')    
        
        if self.image_preprocess is not None: 
            image = self.image_preprocess(image)  
            
        #item = edict({"image_options": [image], "caption_options": test_case["caption_options"]})
        #return item
        return image, test_case["caption_options"], torch.BoolTensor([[True] + [False] * len(test_case["caption_options"][1:])])
    
    def evaluate_scores(self, scores):
        if isinstance(scores, tuple):
            scores_i2t = scores[0]
            scores_t2i = scores[1].T # Make it N_ims x N_text
        else:
            scores_t2i = scores
            scores_i2t = scores
        
        preds = np.argmax(np.squeeze(scores_i2t, axis=1), axis=-1)
        correct_mask = (preds == 0)
        result_records = [{"Precision@1": np.mean(correct_mask)}]
        return result_records
