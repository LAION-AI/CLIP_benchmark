# Thanks to authors https://github.com/amitakamath/whatsup_vlms/
import os
import json
import subprocess

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
from easydict import EasyDict as edict
from torchvision.datasets.utils import download_url


class VG_QA(Dataset):
    def __init__(self, image_preprocess, text_perturb_fn=None, image_perturb_fn=None, root_dir=".", download=False, subset='one'):
        self.root_dir = root_dir
        if subset == 'one':
            annotation_file = os.path.join(root_dir, "vg_qa_one_obj.json")
            image_dir = os.path.join(root_dir, 'vg_images')
        else:
            annotation_file = os.path.join(root_dir, "vg_qa_two_obj.json")
            image_dir = os.path.join(root_dir, 'vg_images')
        if not os.path.exists(image_dir):
            print("Image directory for VG-QA could not be found!")
            if download:
                self.download()
            else:
                raise RuntimeError("Please either download the dataset by letting `--download` or specify the correct directory.")

        if not os.path.exists(annotation_file):
            if subset == 'one':
                subprocess.call(["gdown", "--id", "1ARMRzRdohs9QTr1gpIfzyUzvW20wYp_p", "--output", annotation_file])
            else:
                subprocess.call(["gdown", "--id", "1sjVG5O3QMY8s118k7kQM8zzDZH12i_95", "--output", annotation_file])


        self.dataset = json.load(open(annotation_file))
        self.subset = subset
        self.all_prepositions = []
        if self.subset == 'one':
            self.all_prepositions = [d[1].split()[-1] for d in self.dataset]
        else:
            for d in self.dataset:
                if 'left of' in d[1]:
                    self.all_prepositions.append('left')
                elif 'right of' in d[1]:
                    self.all_prepositions.append('right')
                elif 'front of' in d[1]:
                    self.all_prepositions.append('front')
                elif 'behind' in d[1]:
                    self.all_prepositions.append('behind')
                else:
                    self.all_prepositions.append('top')
        self.image_preprocess = image_preprocess

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        test_case = self.dataset[index]
        image = Image.open(os.path.join(self.root_dir, 'vg_images/{}.jpg'.format(test_case[0]))).convert('RGB')
        if self.image_preprocess is not None:
            image = self.image_preprocess(image)
        
        #Qitem = edict({"image_options": [image], "caption_options": [test_case[1], test_case[2]]})
        #return item
        return image, [test_case[1], test_case[2]], torch.BoolTensor([[True, False]])

    def download(self):
        os.makedirs(self.root_dir, exist_ok=True)
        image_zip_file = os.path.join(self.root_dir, "vg_images.tar.gz")
        subprocess.call(["gdown", "--no-cookies",  "1idW7Buoz7fQm4-670n-oERw9U-2JLJvE", "--output", image_zip_file])
        subprocess.call(["tar", "-xvf", "vg_images.tar.gz"], cwd=self.root_dir)


    def evaluate_scores(self, scores):
        """
        Scores: N x 1 x 2, i.e. first caption is right, next is wrong
        """
        if isinstance(scores, tuple):
            scores_i2t = scores[1]
            scores_t2i = scores[0]
        else:
            scores_t2i = scores
            scores_i2t = scores

        metrics = {"Accuracy": None}
        preds = np.argmax(np.squeeze(scores_i2t, axis=1), axis=-1)
        correct_mask = (preds == 0)
        metrics["Accuracy"] = np.mean(correct_mask)
        print(metrics['Accuracy']*100)

        all_prepositions = np.array(self.all_prepositions)
        
        prepositions = list(set(self.all_prepositions)) + ['below', 'bottom', 'front']
        prep_counts = {p: {p1: 0 for p1 in prepositions} for p in prepositions}
        opposite = {'left': 'right', 'right': 'left', 'behind': 'front', 'front': 'behind', 'above': 'below', 'below': 'above', 'bottom': 'top', 'top': 'bottom'}

        for prep, pred in zip(self.all_prepositions, preds):
            if pred == 0:
                prep_counts[prep][prep] += 1
            else:
                prep_counts[prep][opposite[prep]] += 1
        #print(prep_counts)
        result_records = []
        # Log the accuracy of all prepositions
        for prepositions in np.unique(all_prepositions):
            prepositions_mask = (all_prepositions == prepositions)
            if prepositions_mask.sum() == 0:
                continue
            result_records.append({
                "Preposition": prepositions,
                "Accuracy": correct_mask[prepositions_mask].mean(),
                "Count": prepositions_mask.sum(),
                "Dataset": "VG-QA {}-object".format(self.subset)
            })
        return result_records


class Controlled_Images(Dataset):
    def __init__(self, image_preprocess, text_perturb_fn=None, image_perturb_fn=None, root_dir=".", download=False, subset='A'):
        self.root_dir = root_dir
        if subset == 'A':
            annotation_file = os.path.join(root_dir, "controlled_images_dataset.json")
            image_dir = os.path.join(root_dir, 'controlled_images')

            if not os.path.exists(image_dir):
                print("Image directory for Controlled Images A could not be found!")
                if download:
                    self.download()
                else:
                    raise RuntimeError("Please either download the dataset by letting `--download` or specify the correct directory.")

            if not os.path.exists(annotation_file):
                subprocess.call(["gdown", "--id", "1ap8mmmpQjLIjPGuplkpBgc1hoEHCj4hm", "--output", annotation_file])

        else:
            annotation_file = os.path.join(root_dir, "controlled_clevr_dataset.json")
            image_dir = os.path.join(root_dir, 'controlled_clevr')
            if not os.path.exists(image_dir):
                print("Image directory for Controlled Images B could not be found!")
                if download:
                    self.download()
                else:
                    raise RuntimeError("Please either download the dataset by letting `--download` or specify the correct directory.")

            if not os.path.exists(annotation_file):
                subprocess.call(["gdown", "--id", "1unNNosLbdy9NDjgj4l8fsQP3WiAAGA6z", "--output", annotation_file])


        self.dataset = json.load(open(annotation_file))
        self.subset = subset
        self.all_prepositions = []
        if self.subset == 'A':
            for d in self.dataset:
                if 'left_of' in d['image_path']:
                    self.all_prepositions.append('left_of')
                elif 'right_of' in d['image_path']:
                    self.all_prepositions.append('right_of')
                elif '_on_' in d['image_path']:
                    self.all_prepositions.append('on')
                else:
                    self.all_prepositions.append('under')
            self.eval_dict = {(d['image_path'].split('/')[-1].split('_')[0], \
                                d['image_path'].split('/')[-1].split('_')[-1][:-5]): \
                                {'left': 0, 'right': 0, \
                                'on': 0, 'under': 0} for d in self.dataset}
            self.pred_dict = {(d['image_path'].split('/')[-1].split('_')[0], \
                                d['image_path'].split('/')[-1].split('_')[-1][:-5]): \
                                {'left': '', 'right': '', \
                                'on': '', 'under': ''} for d in self.dataset}


        else:
            for d in self.dataset:
                if 'left_of' in d['image_path']:
                    self.all_prepositions.append('left_of')
                elif 'right_of' in d['image_path']:
                    self.all_prepositions.append('right_of')
                elif '_in-front_of_' in d['image_path']:
                    self.all_prepositions.append('in-front_of')
                else:
                    self.all_prepositions.append('behind')
            self.eval_dict = {(d['image_path'].split('/')[-1].split('_')[0], \
                                d['image_path'].split('/')[-1].split('_')[-1][:-5]): \
                                {'left': 0, 'right': 0, \
                                'in-front': 0, 'behind': 0} for d in self.dataset}
            self.pred_dict = {(d['image_path'].split('/')[-1].split('_')[0], \
                                d['image_path'].split('/')[-1].split('_')[-1][:-5]): \
                                {'left': '', 'right': '', \
                                'in-front': '', 'behind': ''} for d in self.dataset}

        self.image_preprocess = image_preprocess

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        test_case = self.dataset[index]
        
        path = test_case["image_path"]
        path = path.replace("data/", self.root_dir + "/")
        image = Image.open(path).convert('RGB')
        if self.image_preprocess is not None:
            image = self.image_preprocess(image)
        
        #item = edict({"image_options": [image], "caption_options": test_case['caption_options']})
        #return item
        return image, test_case['caption_options'], torch.BoolTensor([[True] + [False]*len(test_case['caption_options'][1:])])


    def download(self):
        os.makedirs(self.root_dir, exist_ok=True)
        image_zip_file = os.path.join(self.root_dir, "controlled_images.tar.gz")
        subprocess.call(["gdown", "--no-cookies",  "19KGYVQjrV3syb00GgcavB2nZTW5NXX0H", "--output", image_zip_file])
        subprocess.call(["tar", "-xvf", "controlled_images.tar.gz"], cwd=self.root_dir)
        image_zip_file = os.path.join(self.root_dir, "controlled_clevr.tar.gz")
        subprocess.call(["gdown", "--no-cookies",  "13jdBpg8t3NqW3jrL6FK8HO93vwsUjDxG", "--output", image_zip_file])
        subprocess.call(["tar", "-xvf", "controlled_clevr.tar.gz"], cwd=self.root_dir)



    def evaluate_scores(self, scores):
        """
        Scores: N x 1 x 4, i.e. first caption is right, next three captions are wrong
        """
        if isinstance(scores, tuple):
            scores_i2t = scores[1]
            scores_t2i = scores[0]
        else:
            scores_t2i = scores
            scores_i2t = scores

        metrics = {"Accuracy": None}
        preds = np.argmax(np.squeeze(scores_i2t, axis=1), axis=-1)
        correct_mask = (preds == 0)
        metrics["Accuracy"] = np.mean(correct_mask)
        print("Individual accuracy: {}".format(metrics['Accuracy']*100))

        prepositions = ['on', 'under', 'front', 'behind', 'left', 'right']
        prep_counts = {p: {p1: 0 for p1 in prepositions} for p in prepositions}
        for i, d in enumerate(self.dataset):
            prep = list(set(prepositions).intersection(set(d['caption_options'][preds[i]].split())))
            gold_prep = list(set(prepositions).intersection(set(d['caption_options'][0].split())))
            #if len(prep) != 1 or len(gold_prep)!=1:
            #    pdb.set_trace()
            #    print("?")
            prep = prep[0]
            gold_prep = gold_prep[0]
            prep_counts[gold_prep][prep] += 1

            self.pred_dict[(d['image_path'].split('/')[-1].split('_')[0], \
                            d['image_path'].split('/')[-1].split('_')[-1][:-5])][d['image_path'].split('/')[-1].split('_')[1]] = prep
        #print(prep_counts)
        for d, correct in zip(self.dataset, correct_mask):
            self.eval_dict[(d['image_path'].split('/')[-1].split('_')[0], \
                            d['image_path'].split('/')[-1].split('_')[-1][:-5])][d['image_path'].split('/')[-1].split('_')[1]] = correct

        
        pair_correct = 0
        set_correct = 0
        for obj_pair, correct_dict in self.eval_dict.items():
            if correct_dict['left'] and correct_dict['right']:
                pair_correct += 1
            if self.subset == 'A':
                if correct_dict['on'] and correct_dict['under']:
                    pair_correct += 1
            else:
                if correct_dict['in-front'] and correct_dict['behind']:
                    pair_correct += 1
            if sum(correct_dict.values()) == 4:
                set_correct += 1
        pair_accuracy = pair_correct*100/(len(self.dataset)/2)
        set_accuracy = set_correct*100/(len(self.dataset)/4)
        print("Pair accuracy: {}".format(pair_accuracy))
        print("Set accuracy: {}".format(set_accuracy))
        all_prepositions = np.array(self.all_prepositions)

        result_records = []
        # Log the accuracy of all prepositions
        for prepositions in np.unique(all_prepositions):
            prepositions_mask = (all_prepositions == prepositions)
            if prepositions_mask.sum() == 0:
                continue
            result_records.append({
                "Preposition": prepositions,
                "Accuracy": correct_mask[prepositions_mask].mean(),
                "Count": prepositions_mask.sum(),
                "Dataset": "Controlled Images - {}".format(self.subset)
            })
        return result_records


class COCO_QA(Dataset):
    def __init__(self, image_preprocess, text_perturb_fn=None, image_perturb_fn=None, root_dir=".", download=False, subset='one'):
        self.root_dir = root_dir
        if subset == 'one':
            annotation_file = os.path.join(root_dir, "coco_qa_one_obj.json")
            image_dir = os.path.join(root_dir, 'val2017')
        else:
            annotation_file = os.path.join(root_dir, "coco_qa_two_obj.json")
            image_dir = os.path.join(root_dir, 'val2017')
        if not os.path.exists(image_dir):
            print("Image directory for COCO-QA could not be found!")
            if download:
                self.download()
            else:
                raise RuntimeError("Please either download the dataset by letting `--download` or specify the correct directory.")

        if not os.path.exists(annotation_file):
            if subset == 'one':
                subprocess.call(["gdown", "--id", "1RsMdpE9mmwnK4zzMPpC1-wTU_hNis-dq", "--output", annotation_file])
            else:
                subprocess.call(["gdown", "--id", "1TCEoM0mgFmz8T4cF7PQ3XJmO6JjtiQ-s", "--output", annotation_file])


        self.dataset = json.load(open(annotation_file))
        self.subset = subset
        self.all_prepositions = []
        if self.subset == 'one':
            self.all_prepositions = [d[1].split()[-1] for d in self.dataset]
        else:
            for d in self.dataset:
                if 'left of' in d[1]:
                    self.all_prepositions.append('left')
                elif 'right of' in d[1]:
                    self.all_prepositions.append('right')
                elif 'above' in d[1]:
                    self.all_prepositions.append('above')
                else:
                    self.all_prepositions.append('below')
        self.image_preprocess = image_preprocess

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        test_case = self.dataset[index]
        image = Image.open(os.path.join(self.root_dir, 'val2017/{}.jpg'.format(str(test_case[0]).zfill(12)))).convert('RGB')
        if self.image_preprocess is not None:
            image = self.image_preprocess(image)
   
        return image, [test_case[1], test_case[2]], torch.BoolTensor([[True, False]]) 


    def download(self):
        os.makedirs(self.root_dir, exist_ok=True)
        image_zip_file = os.path.join(self.root_dir, "val2017.zip")
        subprocess.call(["gdown", "--no-cookies",  "1zp5vBRRM4_nSik6o9PeVspDvOsHgPT4l", "--output", image_zip_file])
        subprocess.call(["unzip", "val2017.zip"], cwd=self.root_dir)


    def evaluate_scores(self, scores):
        """
        Scores: N x 1 x 2, i.e. first caption is right, next is wrong
        """
        if isinstance(scores, tuple):
            scores_i2t = scores[1]
            scores_t2i = scores[0]
        else:
            scores_t2i = scores
            scores_i2t = scores

        metrics = {"Accuracy": None}
        preds = np.argmax(np.squeeze(scores_i2t, axis=1), axis=-1)
        correct_mask = (preds == 0)
        metrics["Accuracy"] = np.mean(correct_mask)
        print(metrics['Accuracy']*100)

        all_prepositions = np.array(self.all_prepositions)

        prepositions = list(set(self.all_prepositions))
        prep_counts = {p: {p1: 0 for p1 in prepositions} for p in prepositions}
        opposite = {'left': 'right', 'right': 'left', 'above': 'below', 'below': 'above', 'top': 'bottom', 'bottom': 'top'}

        for prep, pred in zip(self.all_prepositions, preds):
            if pred == 0:
                prep_counts[prep][prep] += 1
            else:
                prep_counts[prep][opposite[prep]] += 1
        #print(prep_counts)
        result_records = []
        # Log the accuracy of all prepositions
        for prepositions in np.unique(all_prepositions):
            prepositions_mask = (all_prepositions == prepositions)
            if prepositions_mask.sum() == 0:
                continue
            result_records.append({
                "Preposition": prepositions,
                "Accuracy": correct_mask[prepositions_mask].mean(),
                "Count": prepositions_mask.sum(),
                "Dataset": "COCO-QA {}-object".format(self.subset)
            })
        return result_records