"""
Code adapated from https://github.com/mlfoundations/open_clip/blob/main/src/training/zero_shot.py
Thanks to the authors of OpenCLIP
"""
import logging
from contextlib import suppress

import torch
import torch.nn.functional as F
from tqdm import tqdm



def zero_shot_classifier(model, tokenizer, classnames, templates, device):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template.format(c=classname) for template in templates]  # format with class
            texts = tokenizer(texts).to(device)  # tokenize
            class_embeddings = model.encode_text(texts)
            class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def run_classification(model, classifier, dataloader, device, amp=False):
    autocast = torch.cuda.amp.autocast if amp else suppress
    with torch.no_grad():
        top1, top5, n = 0., 0., 0.
        for images, target in tqdm(dataloader):
            images = images.to(device)
            target = target.to(device)

            with autocast():
                # predict
                image_features = model.encode_image(images)
                image_features = F.normalize(image_features, dim=-1)
                logits = 100. * image_features @ classifier

            # measure accuracy
            if len(dataloader.dataset.classes) >= 5:
                acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            else:
                acc1, = accuracy(logits, target, topk=(1,))
                acc5 = float("nan")
            top1 += acc1
            top5 += acc5
            n += images.size(0)

    top1 = (top1 / n)
    top5 = (top5 / n)
    return {'top1_zeroshot_accuracy': top1, 'top5_zeroshot_accuracy': top5}

def evaluate(model, dataloader, tokenizer, classnames, templates, device, amp=False):
    classifier = zero_shot_classifier(model, tokenizer, classnames, templates, device)
    return run_classification(model, classifier, dataloader, device, amp=amp)