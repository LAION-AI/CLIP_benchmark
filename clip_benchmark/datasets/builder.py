from . import cifar10
from . import  imagenet1k
from . import cifar100
from torchvision.datasets import CIFAR10, CIFAR100, ImageNet, CocoCaptions, Flickr8k, Flickr30k

def build_dataset(args, transform, train=False, download=True, **kwargs):
    dataset_name = args.dataset
    root = args.dataset_root
    available_datasets = ["cifar10", "cifar100", "imagenet1k", "mscoco_captions"] 
    if dataset_name == "cifar10":
        ds = CIFAR10(root=root, train=train, transform=transform, download=download, **kwargs)
        return ds, cifar10.zeroshot_templates, cifar10.classnames
    elif dataset_name == "cifar100":
        ds = CIFAR100(root=root, train=train, transform=transform, download=download, **kwargs)
        return ds, cifar100.zeroshot_templates, cifar100.classnames
    elif dataset_name == "imagenet1k":
        ds = ImageNet(root=root, split="train" if train else "val", transform=transform, **kwargs)
        return ds, imagenet1k.zeroshot_templates, imagenet1k.classnames
    elif dataset_name == "mscoco_captions":
        ds = CocoCaptions(root=root, ann_file=args.annotation_file, transform=transform, **kwargs)
        return ds, None, None
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Supported datasets are: f{available_datasets}")