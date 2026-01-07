import os
import glob
import librosa
import numpy as np
import torch
import sys
from subprocess import call
from torch.utils.data import Dataset
from typing import Optional, Callable, Tuple, List
from .utils import pad_or_crop

class GTZAN(Dataset):
    """
    GTZAN Dataset wrapper.
    
    The GTZAN dataset is a dataset for evaluation in music genre recognition (MGR). 
    It consists of 1000 audio tracks each 30 seconds long. 
    It contains 10 genres, each represented by 100 tracks.

    Args:
        root (str): Path to the directory containing 'Data/genres_original' or 'genres_original'.
        split (str): only 'all' is supported. Argument is kept for consistency with other datasets.
        target_sr (int): Sampling rate (default 48000).
        target_len (int): Target length of the audio in samples (default 144000 = 3s @ 48kHz).
        transform (Optional[Callable]): Optional transform to be applied on a audio sample.
    """
    def __init__(self, root: str, split: str = "test", target_sr: int = 48000, target_len: int = 144000, transform: Optional[Callable] = None) -> None:
        self.root = root
        
        # Determine the correct root directory structure
        # builder.py downloads to `root`. Unzips to `root`.
        # Inside might be `gtzan-dataset.../genres_original` or `Data/genres_original`.
        # Original code checked `os.path.join(root, 'Data', 'genres_original')`.
        # Let's try to be robust.
        
        potential_paths = [
            os.path.join(root, 'Data', 'genres_original'),
            os.path.join(root, 'genres_original'),
            root
        ]
        
        found = False
        for p in potential_paths:
            if os.path.exists(p) and os.path.isdir(p):
                # Check if it has genres
                if len(glob.glob(os.path.join(p, "*", "*.wav"))) > 0:
                    self.root = p
                    found = True
                    break
        
        if not found:
            GTZAN.download(root)
            self.root = os.path.join(root, 'Data', 'genres_original')

        self.target_sr = target_sr
        self.target_len = target_len
        self.transform = transform
        
        # Find all .wav files in subdirectories
        # Structure: root_dir/genre/file.wav
        self.files = sorted(glob.glob(os.path.join(self.root, '*', '*.wav')))
        
        # Create class mapping
        self.classes = sorted(list(set([os.path.basename(os.path.dirname(f)) for f in self.files])))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get item at index idx.

        Args:
            idx (int): Index of the item.

        Returns:
            Tuple[torch.Tensor, int]: (audio_tensor, target_label_index)
        """
        file_path = self.files[idx]
        
        # Load audio at target_sr
        try:
            audio_data, _ = librosa.load(file_path, sr=self.target_sr)
        except Exception as e:
            audio_data = np.zeros(self.target_len)
            print(f"Warning: Failed to load {file_path}, returning silence.")
        
        # Pad/Crop
        audio_data = pad_or_crop(audio_data, self.target_len)
        
        # Convert to PyTorch Tensor
        audio_tensor = torch.from_numpy(audio_data).float()
        
        if self.transform:
            audio_tensor = self.transform(audio_tensor)
        
        # Get label
        genre = os.path.basename(os.path.dirname(file_path))
        label = self.class_to_idx[genre]

        return audio_tensor, label
    
    @staticmethod
    def available_splits() -> List[str]:
        return ["test"]

    @staticmethod
    def download(root: str) -> None:
        """
        Download and extract the GTZAN dataset using Kaggle API.
        
        Args:
            root (str): Root directory to download to.
        """
        assert(not os.path.exists(root)), f"Root directory {root} already exists."
        
        # Check for kaggle
        if call("which kaggle", shell=True) != 0:
            print("Kaggle is needed to download the dataset. Please install it via `pip install kaggle`")
            sys.exit(1)
            
        print("Downloading gtzan...")
        call(f"kaggle datasets download -d andradaolteanu/gtzan-dataset-music-genre-classification", shell=True)
        call(f"unzip gtzan-dataset-music-genre-classification.zip -d {root}", shell=True)
        call(f"rm gtzan-dataset-music-genre-classification.zip", shell=True)


