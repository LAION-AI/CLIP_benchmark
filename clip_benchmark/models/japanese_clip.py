import torch


def load_japanese_clip(model_path: str, device="cpu", **kwargs):
    try:
        import japanese_clip as ja_clip
    except ImportError:
        raise ImportError("Install `japanese_clip` by `pip install git+https://github.com/rinnakk/japanese-clip.git`")
    model, transform = ja_clip.load(model_path, device=device, **kwargs)

    class JaTokenizerWrapper:
        def __init__(self, ):
            self.tokenizer = ja_clip.load_tokenizer()

        def __call__(self, texts) -> torch.Tensor:
            inputs = ja_clip.tokenize(texts, tokenizer=self.tokenizer, device="cpu")
            return inputs

        def __len__(self):
            return len(self.tokenizer)

    return model, transform, JaTokenizerWrapper()
