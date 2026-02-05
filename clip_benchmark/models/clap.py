import torch
import numpy as np
import io
import sys
import librosa
import laion_clap
import collections.abc
from laion_clap.training.data import get_audio_features, int16_to_float32, float32_to_int16
from transformers import RobertaTokenizer

def load_clap(model_name: str = "HTSAT-tiny", pretrained: str = "630k-audioset-best", device="cpu", **kwargs):
    fusion = "fusion" in pretrained
    
    model = laion_clap.CLAP_Module(enable_fusion=fusion, amodel=model_name)
    model.load_ckpt(pretrained)
    model = model.to(device)
    model.model.eval() # ensure internal model is in eval mode
    
    clap_wrapper = CLAPWrapper(model)
    clap_tokenizer = CLAPTokenizer()
    clap_transform = CLAPTransform()
    clap_loader = AudioLoader(fusion, model.model_cfg)
    
    return clap_wrapper, clap_transform, clap_tokenizer, clap_loader

class CLAPWrapper(torch.nn.Module):
    """ CLAP wrapper for CLIP benchmark """

    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def encode_text(self, text):
        return self.model.model.get_text_embedding(text)
    
    def encode_audio(self, audio):
        if isinstance(audio, dict):
            device = next(self.model.parameters()).device
            input_dict = {k: v.to(device) for k, v in audio.items()}
            audio_embeds = self.model.model.encode_audio(input_dict, device=device)["embedding"]
            audio_embeds = self.model.model.audio_projection(audio_embeds)
            audio_embeds = torch.nn.functional.normalize(audio_embeds, dim=-1)
            return audio_embeds
        return self.model.model.get_audio_embedding(audio)

class CLAPTokenizer:
    def __init__(self):
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    def __call__(self, texts):
        return self.tokenizer(texts, padding="max_length", truncation=True, max_length=77, return_tensors="pt")

class CLAPTransform:
    def __call__(self, audio):
        return audio

class AudioLoader:
    def __init__(self, enable_fusion, model_cfg):
        self.enable_fusion = enable_fusion
        self.model_cfg = model_cfg
    
    def __call__(self, key, data):
        """
        Decodes audio data using librosa.
        """
        extension = key.split(".")[-1].lower()
        if extension not in ["wav", "flac", "mp3"]:
            return None
        
        try:
            # Load audio using librosa from bytes
            audio_waveform, _ = librosa.load(io.BytesIO(data), sr=48000)
            
            # quantize
            audio_waveform = int16_to_float32(float32_to_int16(audio_waveform))
            audio_waveform = torch.from_numpy(audio_waveform).float()

            temp_dict = {}
            temp_dict = get_audio_features(
                temp_dict, audio_waveform, 480000, 
                data_truncating='fusion' if self.enable_fusion else 'rand_trunc', 
                data_filling='repeatpad',
                audio_cfg=self.model_cfg['audio_cfg'],
                require_grad=audio_waveform.requires_grad
            )
            return temp_dict
        except Exception as e:
            print(f"Error loading {key}: {e}")
            return None