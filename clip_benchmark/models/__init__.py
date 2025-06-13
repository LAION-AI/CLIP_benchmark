from typing import Union
import torch
import all_clip

 # see https://github.com/rom1504/all-clip?tab=readme-ov-file#supported-models
MODEL_TYPES = ["openai_clip", "open_clip", "ja_clip", "hf_clip", "nm"]


def load_clip(
        model_type: str,
        model_name: str,
        pretrained: str,
        cache_dir: str,
        device: Union[str, torch.device] = "cuda"
):
    assert model_type in MODEL_TYPES, f"model_type={model_type} is invalid!"
    return all_clip.load_clip(
        clip_model=model_type+":"+model_name+"/"+pretrained,
        use_jit=True,
        warmup_batch_size=1,
        clip_cache_path=cache_dir,
        device=device,
    )
