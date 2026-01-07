import torch
from datasets import load_dataset, Audio
from typing import Optional, Callable, Tuple, List

class ESC50(torch.utils.data.Dataset):
    """
    ESC-50 Dataset wrapper.
    
    The ESC-50 dataset is a labeled collection of 2000 environmental audio recordings
    suitable for benchmarking methods of environmental sound classification.
    
    Args:
        root (str): Root directory where the dataset is stored/cached.
        split (str): Dataset split to load. Default is "train".
                     Note: The huggingface dataset usually has just 'train', 
                     but we can use 'all' implies the full dataset.
        transform (Optional[Callable]): A function/transform that takes in a raw audio tensor
                                        and returns a transformed version.
    """
    def __init__(self, root: str, split: str = "all", transform: Optional[Callable] = None) -> None:
        # 'ashraq/esc50' on HF only has a 'train' split which contains all data. We keep the split argument for consistency with other datasets and to make sure the user is not mistaken..
        self.dataset = load_dataset("ashraq/esc50", split=split, cache_dir=root)
        self.dataset = self.dataset.cast_column("audio", Audio(sampling_rate=48000, num_channels=1))
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get item at index idx.
        
        Args:
            idx (int): Index of the item.
            
        Returns:
            Tuple[torch.Tensor, int]: (audio_tensor, target_label_index)
        """
        item = self.dataset[idx]
        audio_data = torch.tensor(item['audio']['array']).float()
        
        label = item['target'] 
        
        if self.transform:
            audio_data = self.transform(audio_data)
        
        return audio_data, label

    @staticmethod
    def available_splits() -> List[str]:
        return ["all"]