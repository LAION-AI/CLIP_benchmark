import open_clip


def load_open_clip(model: str = "ViT-B-32-quickgelu", pretrained: str = "laion400m_e32", cache_dir: str = None, device="cpu"):
    model, _, transform = open_clip.create_model_and_transforms(model, pretrained=pretrained, cache_dir=cache_dir)
    model = model.to(device)
    tokenizer = open_clip.get_tokenizer(model)
    return model, transform, tokenizer
