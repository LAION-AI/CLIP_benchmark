import torch
from torch import nn
from transformers import AutoModel, AutoProcessor
from functools import partial

class TransformerWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def encode_text(self, text):
        return self.model.get_text_features(**text)

    def encode_image(self, image):
        # we get an extended dimension possibly due to the collation in dataloader
        image = {key: value.squeeze(1) for key, value in image.items()}
        return self.model.get_image_features(**image)

def load_transformers_clip(model_name, pretrained, cache_dir, device):
    ckpt = f"{model_name}/{pretrained}"
    model = AutoModel.from_pretrained(ckpt, cache_dir=cache_dir, device_map=device)
    model = TransformerWrapper(model)
    
    processor = AutoProcessor.from_pretrained(ckpt)
    transforms = partial(processor.image_processor.preprocess, return_tensors="pt")
    tokenizer = partial(
        processor.tokenizer, return_tensors="pt", padding="max_length",
        max_length=64 # very specific to SG2
        )
    return model, transforms, tokenizer
