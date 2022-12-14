from typing import Dict
import torch


def load_japanese_clip(pretrained: str, device="cpu", **kwargs):
    """
    Load Japanese CLIP/CLOOB by rinna (https://github.com/rinnakk/japanese-clip)
    Remarks:
     - You must input not only input_ids but also attention_masks and position_ids when doing `model.encode_text()` to make it work correctly.
    """
    try:
        import japanese_clip as ja_clip
    except ImportError:
        raise ImportError("Install `japanese_clip` by `pip install git+https://github.com/rinnakk/japanese-clip.git`")
    model, transform = ja_clip.load(pretrained, device=device, **kwargs)

    class JaTokenizerWrapper:
        def __init__(self, ):
            self.tokenizer = ja_clip.load_tokenizer()

        def __call__(self, texts) -> Dict[str, torch.Tensor]:
            inputs = ja_clip.tokenize(texts, tokenizer=self.tokenizer, device="cpu")
            return inputs

        def __len__(self):
            return len(self.tokenizer)

    return model, transform, JaTokenizerWrapper()
