"""Console script for clip_benchmark."""
import argparse
import sys
import json
import torch
import open_clip

from clip_benchmark.datasets.builder import build_dataset, get_dataset_collate_fn, get_zeroshot_classification_templates
from clip_benchmark.metrics import zeroshot_classification, zeroshot_retrieval, linear_probe

from torch.utils.data import default_collate, Subset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="cifar10", help="Dataset to use for the benchmark")
    parser.add_argument('--split', type=str, default="test", help="Dataset split to use")
    parser.add_argument('--model', type=str, default="ViT-B-32-quickgelu", help="Model architecture to use from OpenCLIP")
    parser.add_argument('--pretrained', type=str, default="laion400m_e32", help="Model checkpoint name to use from OpenCLIP")
    parser.add_argument('--task', type=str, default="zeroshot_classification", choices=["zeroshot_classification", "zeroshot_retrieval", "linear_probe"])
    parser.add_argument('--amp', default=False, action="store_true", help="whether to use mixed precision")
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--subset', default="", type=str)
    parser.add_argument('--not_subset', default="", type=str)
    parser.add_argument('--recall_k', default=[5], type=int, help="for retrieval, select the k for Recall@K metric. ", nargs="+",)
    parser.add_argument('--fewshot_k', default=-1, type=int, help="for linear probe, how many shots. -1 = whole dataset.")
    parser.add_argument('--fewshot_epochs', default=10, type=int, help="for linear probe, how many epochs.")
    parser.add_argument('--fewshot_lr', default=0.1, type=float, help="for linear probe, what is the learning rate.")
    parser.add_argument("--skip_load", action="store_true", help="for linear probes, when everything is cached, no need to load model.")
    parser.add_argument('--seed', default=0, type=int, help="random seed.")
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--dataset_root', default="root", type=str, help="dataset root folder where the datasets are downloaded.")
    parser.add_argument('--feature_root', default="features", type=str, help="feature root folder where the features are stored.")
    parser.add_argument('--annotation_file', default="", type=str, help="text annotation file for retrieval datasets. Only needed  for when `--task` is `zeroshot_retrieval`.")
    parser.add_argument('--output', default="result.json", type=str, help="output file where to dump the metrics")
    parser.add_argument('--verbose', default=False, action="store_true", help="verbose mode")
    args = parser.parse_args()
    run(args)
    
def run(args):
    """Console script for clip_benchmark."""
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    # set seed.
    torch.manual_seed(args.seed)
    if args.skip_load:
        model, transform, collate_fn, dataloader = None, None, None, None
    else:
        model, _, transform = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
        model = model.to(args.device)
        dataset = build_dataset(
            dataset_name=args.dataset, 
            root=args.dataset_root, 
            transform=transform, 
            split=args.split, 
            annotation_file=args.annotation_file,
            download=True,
        )
        import pandas as pd
        if args.subset or args.not_subset:
            fn = args.subset if args.subset else args.not_subset
            N = len(dataset)
            paths = open(fn).readlines()
            paths = ["/p/scratch/ccstdl/cherti1/clip_benchmark_datasets/" + args.dataset + "/" + p.strip() for p in paths]
            print(paths)
            inds = [i for i in range(len(dataset)) if dataset.samples[i][0] in paths]
            if args.not_subset:
                inds = list(set(range(N)) - set(inds))
            print(inds)
            classes = dataset.classes
            d = dataset
            dataset = Subset(dataset, inds)
            dataset.classes = classes
            T = d.transform
            d.transform = None
            dataset[10][0].save(f"{args.dataset}.png")
            d.transform = T

        collate_fn = get_dataset_collate_fn(args.dataset)
        if args.verbose:
            print(f"Dataset size: {len(dataset)}")
            print(f"Dataset split: {args.split}")
            print(f"Dataset classes: {dataset.classes}")
            print(f"Dataset number of classes: {len(dataset.classes)}")

        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, 
            shuffle=False, num_workers=args.num_workers, 
            collate_fn=collate_fn
        )

    if args.task == "zeroshot_classification":
        zeroshot_templates = get_zeroshot_classification_templates(args.dataset)
        if args.verbose:
            print(f"Zero-shot templates: {zeroshot_templates}")
        classnames = dataset.classes if hasattr(dataset, "classes") else None
        assert (zeroshot_templates is not None and classnames is not None), "Dataset does not support classification"
        metrics = zeroshot_classification.evaluate(
            model, 
            dataloader, 
            open_clip.tokenizer.tokenize, 
            classnames, zeroshot_templates, 
            device=args.device, 
            amp=args.amp,
            verbose=args.verbose,
        )
    elif args.task == "zeroshot_retrieval":
        metrics = zeroshot_retrieval.evaluate(
            model, 
            dataloader, 
            open_clip.tokenizer.tokenize, 
            recall_k_list=args.recall_k,
            device=args.device, 
            amp=args.amp
        )
    elif args.task == "linear_probe":
        # we also need the train split for linear probing.
        train_dataset = build_dataset(
            dataset_name=args.dataset, 
            root=args.dataset_root, 
            transform=transform, 
            split='train', 
            annotation_file=args.annotation_file,
            download=True,
        )
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, 
            shuffle=False, num_workers=args.num_workers, 
            collate_fn=collate_fn, pin_memory=True,
        )
        metrics = linear_probe.evaluate(
            model,
            train_dataloader, 
            dataloader, 
            args.fewshot_k,
            args.batch_size,
            args.num_workers,
            args.fewshot_lr,
            args.fewshot_epochs,
            (args.model + '-' + args.pretrained + '-' + args.dataset).replace('/', '_'),
            args.seed,
            args.feature_root,
            device=args.device, 
            amp=args.amp,
            verbose=args.verbose,
        )
    else:
        raise ValueError("Unsupported task: {}. task should `zeroshot_classification` or `zeroshot_retrieval`".format(args.task))
    dump = {
        "dataset": args.dataset,
        "model": args.model,
        "pretrained": args.pretrained,
        "task": args.task,
        "metrics": metrics
    }
    with open(args.output, "w") as f:
        json.dump(dump, f)
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
