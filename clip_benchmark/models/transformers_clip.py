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
        return self.model.get_image_features(image["pixel_values"].squeeze(1))

def load_transformers_clip(model_name, pretrained, cache_dir, device):
    ckpt = f"{model_name}/{pretrained}"
    model = AutoModel.from_pretrained(ckpt, cache_dir=cache_dir, device_map=device)
    model = TransformerWrapper(model)
    processor = AutoProcessor.from_pretrained(ckpt)

    transforms = partial(processor.image_processor, return_tensors="pt") 
    tokenizer = partial(processor.tokenizer, return_tensors="pt", padding="max_length")
    return model, transforms, tokenizer
