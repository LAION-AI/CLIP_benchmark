"""
Loader for the CLAP2 (NaFlex) audio-text models of OpenCLIP.

`model_name` is any OpenCLIP audio-text model: a model name with a pretrained tag or checkpoint file,
"hf-hub:<repo>" or "local-dir:<directory>" (then `pretrained` is ignored).
Audio keeps its native length.
"""
import open_clip


def collate_patch_dicts(patch_dicts):
    """Pad the NaFlex patch dicts of a batch to the longest one (OpenCLIP's eval collate caps clips at a length)."""
    from open_clip_train.naflex_data import NaFlexBatchScheduler

    batch = NaFlexBatchScheduler._collate_patch_dicts(patch_dicts, None)
    del batch["seq_len"]  # the padded length, which the model does not take
    return batch


def load_clap2(model_name, pretrained="", cache_dir=None, device="cpu"):
    from .clap import batched

    model, _, transform_factory = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained or None, require_pretrained=True, cache_dir=cache_dir, device=device)
    return model.eval(), batched(transform_factory(), collate_patch_dicts, device), open_clip.get_tokenizer(model_name)
