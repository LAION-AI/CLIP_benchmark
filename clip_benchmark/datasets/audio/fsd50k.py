import torch
import os
import pandas as pd
import librosa
import numpy as np
from torch.utils.data import Dataset
from typing import Optional, Callable, Tuple, List
from .utils import pad_or_crop

class FSD50K(Dataset):
    """
    FSD50K Dataset wrapper.

    Freesound Dataset 50k (FSD50K) is an open dataset of human-labeled sound events 
    containing 51,197 audio clips unequally distributed in 200 classes of the AudioSet Ontology.
    Download at: https://zenodo.org/records/4060432

    Args:
        root (str): Root directory where the dataset is stored.
                    Expects 'FSD50K.ground_truth' subdir and 'FSD50K.dev_audio'/'FSD50K.eval_audio' subdirs.
        split (str): Dataset split to load. "train" (dev set) or "eval" (eval set).
        transform (Optional[Callable]): A function/transform that takes in a raw audio tensor
                                        and returns a transformed version.
        target_len (int): Target length of the audio in samples. Default is 192000 (4s at 48kHz). Unfitting values will be padded or cropped.
    """
    def __init__(self, root: str, split: str = "train", transform: Optional[Callable] = None, target_len: int = 192000) -> None:
        self.root = root
        self.transform = transform
        self.TARGET_LEN = target_len

        # Bottom-up imports/checks
        if not os.path.exists(os.path.join(self.root, 'FSD50K.ground_truth')):
             raise FileNotFoundError(f"FSD50K.ground_truth not found in {self.root}. You need to download the dataset first and then pass the root directory to the dataset.")

        # Paths
        self.gt_path = os.path.join(self.root, 'FSD50K.ground_truth')
            
        # Load Vocabulary
        vocab_path = os.path.join(self.gt_path, 'vocabulary.csv')
        self.classes = []
        if os.path.exists(vocab_path):
            vocab_df = pd.read_csv(vocab_path, header=None, names=['label', 'display_name', '_'])
            self.classes = vocab_df['display_name'].tolist()
            self.label_to_display = dict(zip(vocab_df['label'], vocab_df['display_name']))
        
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        # Load Split Metadata and define audio dir
        if split == "train":
            # Dev set
            csv_name = "dev.csv"
            self.audio_dir = os.path.join(self.root, "FSD50K.dev_audio")
            self.metadata = pd.read_csv(os.path.join(self.gt_path, csv_name))
        else:
            # Eval set
            csv_name = "eval.csv"
            self.audio_dir = os.path.join(self.root, "FSD50K.eval_audio")
            self.metadata = pd.read_csv(os.path.join(self.gt_path, csv_name))
    
    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get item at index idx.

        Args:
            idx (int): Index of the item.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (audio_tensor, target_multihot)
        """
        row = self.metadata.iloc[idx]
        fname = str(row['fname'])
        labels_str = row['labels']
        
        # Multi-hot encoding
        target = torch.zeros(len(self.classes))
        for l in labels_str.split(','):
            if l in self.class_to_idx:
                target[self.class_to_idx[l]] = 1.0
        
        
        # Load audio
        file_path = os.path.join(self.audio_dir, fname + ".wav")
        try:
            audio_data, _ = librosa.load(file_path, sr=48000)
        except Exception as e:
            print(f"Warning: Failed to load {file_path}, returning silence.")
            audio_data = np.zeros(self.TARGET_LEN)

        # Pad/Crop to target length
        audio_data = pad_or_crop(audio_data, self.TARGET_LEN)
        
        audio_tensor = torch.from_numpy(audio_data).float()

        if self.transform:
            audio_tensor = self.transform(audio_tensor)

        return audio_tensor, target

    @staticmethod
    def available_splits() -> List[str]:
        return ["train", "test"]
