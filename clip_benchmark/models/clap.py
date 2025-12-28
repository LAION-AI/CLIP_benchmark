import laion_clap
import torch

def int16_to_float32(x):
    return (x / 32767.0).to(torch.float32)


def float32_to_int16(x):
    x = torch.clip(x, min=-1., max=1.)
    return (x * 32767.0).to(torch.int16)

class CLAPWrapper(torch.nn.Module):
    """ CLAP wrapper for CLIP benchmark """

    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def encode_text(self, text):
        return self.model.get_text_embedding(text, use_tensor=True)
    
    @torch.amp.autocast('cuda', enabled=False)
    def encode_audio(self, audio):
        assert type(audio) == torch.Tensor, "Audio must be a torch tensor"
        assert len(audio.shape) == 2, "Audio must be 2D"

        # emulate int16 quantization
        audio_data = int16_to_float32(float32_to_int16(audio)).float()
        
        # Encode Audio - Pass as list to avoid tensor path issues
        audio_embed = self.model.get_audio_embedding_from_data(x=audio_data, use_tensor=True)
        assert audio_embed.shape[0] == audio_data.shape[0], "Audio embedding shape mismatch"
        return audio_embed

def load_clap(model_name: str = "HTSAT-base", pretrained: str = "630k-audioset-best", device="cpu", **kwargs):
    """
    Load CLAP by laion (https://github.com/laion-ai/CLAP)
    
    :model_name: name of audio encoder
    :pretrained: path to checkpoint (find checkpoints here: https://huggingface.co/lukewys/laion_clap/tree/main)
    :device: device to load model onto
    """
    fusion = "fusion" in pretrained
    
    model = laion_clap.CLAP_Module(enable_fusion=fusion, amodel=model_name)
    model.load_ckpt(pretrained)
    model = model.to(device)
    
    clap_wrapper = CLAPWrapper(model)
    
    return clap_wrapper, None, None
