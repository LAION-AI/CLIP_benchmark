from transformers import CLIPModel, CLIPProcessor
from functools import partial

def load_transformers_clip(model_name, pretrained, cache_dir, device):
    ckpt = f"{model_name}/{pretrained}"
    model = CLIPModel.from_pretrained(ckpt, cache_dir=cache_dir, device_map=device)
    processor = CLIPProcessor.from_pretrained(ckpt)

    transforms = partial(processor.image_processor, return_tensors="pt") 
    tokenizer = partial(processor.tokenizer, return_tensors="pt", padding="max_length")
    return model, transforms, tokenizer
