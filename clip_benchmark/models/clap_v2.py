"""
CLAP model loader for our own open_clip CLAP training checkpoints.

Loads checkpoints produced by open_clip CLAP training directly (no key remapping).
Supports both non-fusion and fusion models:
  - Non-fusion: standard HTSAT encoder, waveform input
  - Fusion: HTSAT with AFF-2D fusion, mel_fusion input (auto-detected from checkpoint)

Usage:
  load_clap_v2(model_name="HTSAT-tiny-Roberta-base", pretrained="/path/to/epoch_45.pt")
"""
import logging
import torch
import torch.nn.functional as F
import io
import librosa

import open_clip

from .clap import FusionAudioLoader

logger = logging.getLogger(__name__)


def _detect_fusion(state_dict):
    """Detect if checkpoint uses fusion (has fusion_model / mel_conv2d keys)."""
    for k in state_dict:
        if "fusion_model" in k or "mel_conv2d" in k:
            return True
    return False


def load_clap_v2(model_name: str = "CLAP-HTSAT-tiny-Roberta-base", pretrained: str = "", device="cpu", **kwargs):
    from . import ModelBundle

    # Load checkpoint first to detect fusion before creating model
    state_dict = None
    needs_fusion = False

    if pretrained:
        checkpoint = torch.load(pretrained, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        needs_fusion = _detect_fusion(state_dict)

    # Create model — use fusion-enabled config if checkpoint requires it
    if needs_fusion:
        from open_clip.model import CLAP
        cfg = open_clip.get_model_config(model_name)
        cfg["audio_cfg"]["enable_fusion"] = True
        cfg["audio_cfg"]["fusion_type"] = "aff_2d"
        model = CLAP(**cfg, output_dict=True)
        logger.info("Created model with enable_fusion=True (detected from checkpoint)")
    else:
        model = open_clip.create_model(model_name, output_dict=True)

    if state_dict is not None:
        model.load_state_dict(state_dict)
        logger.info(f"Loaded checkpoint: {pretrained} (epoch {checkpoint.get('epoch', '?')})")

    model = model.to(device)
    model.eval()

    model_cfg = open_clip.get_model_config(model_name) or {}
    audio_cfg = model_cfg.get("audio_cfg", {})

    clap_wrapper = CLAPWrapperNewOpenClip(model)
    clap_tokenizer = open_clip.get_tokenizer(model_name)
    clap_transform = CLAPTransform()
    if needs_fusion:
        clap_loader = FusionAudioLoader(audio_cfg)
    else:
        clap_loader = AudioLoader(audio_cfg)

    return ModelBundle(model=clap_wrapper, transform=clap_transform, tokenizer=clap_tokenizer, audio_loader=clap_loader)


class CLAPWrapperNewOpenClip(torch.nn.Module):
    """Wrapper for new open_clip CLAP.

    Training forward:
        audio: encode_audio(normalize=True)  # projection inside AudioTower
        text:  encode_text(normalize=True)

    encode_audio/encode_text already return features in the shared embedding space.
    No separate projection step needed.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def encode_text(self, text, normalize=False):
        return self.model.encode_text(text, normalize=normalize)

    def encode_audio(self, audio, normalize=False):
        if isinstance(audio, list):
            audio = self._collate_audio_dicts(audio)
        features = self.model.encode_audio(audio, normalize=normalize)
        return features

    @staticmethod
    def _collate_audio_dicts(audio_list):
        """Collate a list of per-sample audio dicts into a batched dict of tensors."""
        batch = {}
        for k in audio_list[0]:
            vals = [d[k] for d in audio_list]
            if isinstance(vals[0], torch.Tensor):
                batch[k] = torch.stack(vals)
            else:
                batch[k] = torch.tensor(vals)
        return batch


class CLAPTransform:
    def __call__(self, audio):
        return audio


class AudioLoader:
    """Load audio from raw bytes, resample to target SR, pad/truncate to clip_samples."""

    def __init__(self, audio_cfg):
        self.sample_rate = audio_cfg.get("sample_rate", 48000)
        self.clip_samples = audio_cfg.get("clip_samples", 480000)

    def __call__(self, key, data):
        extension = key.split(".")[-1].lower()
        if extension not in ["wav", "flac", "mp3"]:
            return None

        try:
            audio_waveform, _ = librosa.load(io.BytesIO(data), sr=self.sample_rate)
            audio_waveform = torch.from_numpy(audio_waveform).float()

            # Pad or truncate
            longer = len(audio_waveform) > self.clip_samples
            if len(audio_waveform) < self.clip_samples:
                audio_waveform = F.pad(audio_waveform, (0, self.clip_samples - len(audio_waveform)))
            else:
                audio_waveform = audio_waveform[:self.clip_samples]

            return {"waveform": audio_waveform, "longer": torch.tensor(longer)}
        except Exception as e:
            print(f"Error loading {key}: {e}")
            return None
