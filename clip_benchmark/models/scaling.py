"""Load Scaling laws for synthetic data

References
- Scaling Laws of Synthetic Images for Model Training ... for Now, Fan et al., CVPR 2024
- https://github.com/google-research/syn-rep-learn/tree/main/Scaling#clip
"""
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data


from open_clip import create_model_and_transforms, get_tokenizer
from torch.nn import functional as F


def load_model(model: str = 'ViT-B-16', pretrained: str = None,
               device: str = "cpu", cudnn_benchmark = True, **kwargs):
    if pretrained is None:
        raise FileNotFoundError(f'Failing early, missing: {pretrained}!')

    tokenizer = get_tokenizer(model)
    model, preprocess_train, preprocess_val = create_model_and_transforms(
        model,
        '',
        precision='amp',
        device='cuda',
        jit=False,
        force_quick_gelu=True,
        force_custom_text=False,
        force_patch_dropout=None,
        force_image_size=224,
        pretrained_image=False,
        image_mean=None,
        image_std=None,
        aug_cfg={},
        output_dict=True,
    )
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    state_dict = torch.load(pretrained, map_location=device)
    logit_scale = np.exp(state_dict['logit_scale'].item())
    msg = model.load_state_dict(state_dict, strict=True)
    print(msg)
    model = model.to(device)
    model.eval()
    cudnn.benchmark = cudnn_benchmark
    return model, preprocess_val, tokenizer


if __name__ == '__main__':
    load_model(ckpt='./logs/scaling_syn_real/371M.pt')
