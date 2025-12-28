import torch
import os
import pandas as pd
import librosa
import numpy as np
from torch.utils.data import Dataset
from typing import Optional, Callable, Tuple, List

class US8K(Dataset):
    """
    UrbanSound8K Dataset wrapper.

    This dataset contains 8732 labeled sound excerpts (<=4s) of urban sounds from 10 classes.

    Args:
        root (str): Root directory where the dataset is stored.
        split (str): Dataset split to load. 'train' (folds 1-9), 'test' (fold 10), or 'all'.
        transform (Optional[Callable]): A function/transform that takes in a raw audio tensor
                                        and returns a transformed version.
        target_len (int): Target length of the audio in samples. Default is 192000 (4s at 48kHz).
    """
    def __init__(self, root: str, split: str = "train", transform: Optional[Callable] = None, target_len: int = 192000) -> None:
        self.root = root
        self.transform = transform
        self.TARGET_LENGTH = target_len
        
        meta_path = os.path.join(self.root, 'metadata', 'UrbanSound8K.csv')
        if not os.path.exists(meta_path):
             raise FileNotFoundError(f"Metadata not found at {meta_path}. You have to download the dataset first (https://urbansounddataset.weebly.com/urbansound8k.html) and extract it to the root directory.")
        
        self.metadata = pd.read_csv(meta_path)
        
        # Extract classes (ensure they are sorted by classID)
        # We can drop duplicates on 'classID' and 'class' and sort by 'classID'
        class_info = self.metadata[['classID', 'class']].drop_duplicates().sort_values('classID')
        self.classes = class_info['class'].tolist()

        # Standard US8K 10-fold cross-validation
        if split == "train":
            self.metadata = self.metadata[self.metadata['fold'] != 10]
        elif split == "test":
            self.metadata = self.metadata[self.metadata['fold'] == 10]
        elif split == "all":
            pass
        else:
            raise ValueError(f"split must be 'train', 'test' or 'all'. train is fold 1-9, test is fold 10. Got: {split}")
        
    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get item at index idx.

        Args:
            idx (int): Index of the item.

        Returns:
            Tuple[torch.Tensor, int]: (audio_tensor, target_label_index)
        """
        row = self.metadata.iloc[idx]
        
        file_name = row['slice_file_name']
        fold = row['fold']
        label_idx = row['classID']
        
        # Construct full path: root/audio/foldN/filename
        file_path = os.path.join(self.root, 'audio', f'fold{fold}', file_name)
        
        # Load audio
        try:
            audio_data, _ = librosa.load(file_path, sr=48000)
        except Exception as e:
            print(f"Warning: Failed to load {file_path}, returning silence. Error: {e}")
            audio_data = np.zeros(self.TARGET_LENGTH)

        # Pad/Crop to target_len
        if len(audio_data) < self.TARGET_LENGTH:
            padding = self.TARGET_LENGTH - len(audio_data)
            audio_data = np.pad(audio_data, (0, padding), 'constant')
        else:
            audio_data = audio_data[:self.TARGET_LENGTH]
        
        audio_tensor = torch.from_numpy(audio_data).float()
        
        if self.transform:
            audio_tensor = self.transform(audio_tensor)

        return audio_tensor, label_idx

    @staticmethod
    def available_splits() -> List[str]:
        return ["train", "test", "all"]

