from typing import Union
import torch
from .open_clip import load_open_clip
from .japanese_clip import load_japanese_clip
from .synthclip import load_synthclip
from .scaling import load_model

# loading function must return (model, transform, tokenizer)
TYPE2FUNC = {
    "open_clip": load_open_clip,
    "ja_clip": load_japanese_clip,
    "synthclip": load_synthclip,
    "scaling": load_model,
    "auto": None,
}
MODEL_TYPES = list(TYPE2FUNC.keys())


def load_clip(
        model_type: str,
        model_name: str,
        pretrained: str,
        cache_dir: str,
        device: Union[str, torch.device] = "cuda"
):
    assert model_type in MODEL_TYPES, f"model_type={model_type} is invalid!"
    if model_type != "auto":
        load_func = TYPE2FUNC[model_type]
    else:
        # It's a hack, but it works! you have a better way? push a PR 😃. EOM - Victor
        if "synthclip" in pretrained:
            load_func = TYPE2FUNC["synthclip"]
        elif "scaling" in pretrained:
            load_func = TYPE2FUNC["scaling"]
        elif pretrained in TYPE2FUNC:
            load_func = TYPE2FUNC[pretrained]
        else:
            print(f"{model_type} and {pretrained=} unsupported defaulting to "
                  "open_clip. The Lord be with you 🙏")
            load_func = TYPE2FUNC["open_clip"]
    return load_func(model_name=model_name, pretrained=pretrained, cache_dir=cache_dir, device=device)
