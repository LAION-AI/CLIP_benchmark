from typing import Union, NamedTuple, Optional
import torch


class ModelBundle(NamedTuple):
    """Consistent return type for all model loaders."""
    model: torch.nn.Module
    transform: object
    tokenizer: object
    audio_loader: Optional[object] = None


from .open_clip import load_open_clip
from .japanese_clip import load_japanese_clip

# loading function must return ModelBundle or (model, transform, tokenizer)
TYPE2FUNC = {
    "open_clip": load_open_clip,
    "ja_clip": load_japanese_clip,
}

try:
    from .clap import load_clap
    TYPE2FUNC["clap"] = load_clap
except ImportError:
    pass

MODEL_TYPES = list(TYPE2FUNC.keys())


def load_clip(
        model_type: str,
        model_name: str,
        pretrained: str,
        cache_dir: str,
        device: Union[str, torch.device] = "cuda"
) -> ModelBundle:
    assert model_type in MODEL_TYPES, f"model_type={model_type} is invalid!"
    load_func = TYPE2FUNC[model_type]
    result = load_func(model_name=model_name, pretrained=pretrained, cache_dir=cache_dir, device=device)
    if isinstance(result, ModelBundle):
        return result
    # Wrap legacy 3-tuple returns for backwards compatibility
    model, transform, tokenizer = result
    return ModelBundle(model=model, transform=transform, tokenizer=tokenizer)

