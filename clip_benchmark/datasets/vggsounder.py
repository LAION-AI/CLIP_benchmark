import torch
import os
import numpy as np
import librosa
from torch.utils.data import Dataset
import vggsounder
from typing import Optional, Callable, Tuple, List

class VGGSounder(Dataset):
    """
    VGGSounder Dataset wrapper.

    VGG-Sound is a large-scale audio-visual dataset containing ~200k video clips 
    with audio events from 309 classes.

    Args:
        root (str): Path to the directory containing audio files.
        split (str): only 'test' is supported in vggsounder. Argument is kept for consistency with other datasets.
        transform (Optional[Callable]): A function/transform that takes in a raw audio tensor
                                        and returns a transformed version.
        target_len (int): Target length of the audio in samples. Default is 384000 (8s at 48kHz).
    """
    def __init__(self, root: str, split: str = "test", transform: Optional[Callable] = None, target_len: int = 384000) -> None:
        self.TARGET_LENGTH = target_len
        self.root = root
        self.transform = transform
        
        # Initialize the annotation object
        self.vgg = vggsounder.VGGSounder()
        self.vgg.set_modality("A")

        # Get all labels
        self.classes = self.vgg.get_all_labels()
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self) -> int:
        return len(self.vgg) 

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Access by stored valid index
        video = self.vgg[idx]
        video_id = video.video_id
        
        # Construct multi-hot target for audible labels ONLY
        target = torch.zeros(len(self.classes))
        
        for label, modality in zip(video.labels, video.modalities):
            if label in self.class_to_idx and modality in ['A', 'AV']:
                target[self.class_to_idx[label]] = 1.0
        
        # Load Audio from Video
        audio_path = os.path.join(self.root, f"audio/{video_id}.wav")
        
        try:
            # Force 48kHz as standard for CLAP
            audio_data, _ = librosa.load(audio_path, sr=48000)
        except Exception as e:
            print(f"Warning: Failed to load {audio_path}, returning silence. Error: {e}")
            audio_data = np.zeros(self.TARGET_LENGTH) # 4s silence fallback
            
        if len(audio_data) < self.TARGET_LENGTH:
            padding = self.TARGET_LENGTH - len(audio_data)
            audio_data = np.pad(audio_data, (0, padding), 'constant')
        else:
            audio_data = audio_data[:self.TARGET_LENGTH]

        audio_tensor = torch.from_numpy(audio_data).float()

        if self.transform:
            audio_tensor = self.transform(audio_tensor)
            
        return audio_tensor, target

    @staticmethod
    def available_splits() -> List[str]:
        return ["test"]
    
