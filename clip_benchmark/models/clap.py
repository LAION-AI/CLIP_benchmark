"""
Loader for LAION-CLAP models (HTSAT audio encoder), through OpenCLIP.

`model_name` is an OpenCLIP config and `pretrained` an OpenCLIP tag or checkpoint, e.g.
`--model CLAP-HTSAT-tiny-Roberta-base --pretrained laion` (the Hugging Face weights laion/clap-htsat-unfused);
use `CLAP-HTSAT-tiny-Roberta-base-fused` for laion/clap-htsat-fused.
Audio is fitted to the 10 s of the model the way OpenCLIP does at evaluation: longer clips are truncated and
shorter ones repeated and padded.
"""
import open_clip
import torch


def batched(preprocess, collate, device):
    """Model input for a batch of (waveform [samples] or [channels, samples], sample rate) pairs: OpenCLIP's
    `preprocess` (mono mix, resampling, features) for each clip, then `collate`, on `device`."""
    from open_clip.utils import move_to_device

    def transform(waveforms, sample_rates):
        audio = [preprocess((torch.as_tensor(waveform).float().cpu(), sample_rate))
                 for waveform, sample_rate in zip(waveforms, sample_rates, strict=True)]
        return move_to_device(collate(audio), device)

    return transform


def load_clap(model_name="CLAP-HTSAT-tiny-Roberta-base", pretrained="laion", cache_dir=None, device="cpu"):
    from open_clip.audio.transform import stack_audio_inputs

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, require_pretrained=True, cache_dir=cache_dir, device=device)
    return model.eval(), batched(preprocess, stack_audio_inputs, device), open_clip.get_tokenizer(model_name)
