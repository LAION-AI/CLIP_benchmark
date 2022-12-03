# Convert CLIP_benchmark datasets to webdataset format

import argparse
import io
import os
import sys

from tqdm import tqdm
import torch.utils.data
import webdataset
from .datasets.builder import build_dataset


def get_parser_args():
    parser = argparse.ArgumentParser(description="""
        Convert a CLIP_benchmark dataset to the webdataset format (TAR files).
        Datasets can be uploaded to the Huggingface Hub to allow CLIP model
        evaluation from anywhere with an Internet connection.
    """)
    parser.add_argument("--dataset", "-d", required=True, type=str,
        help="CLIP_benchmark compatible dataset for conversion")
    parser.add_argument("--split", "-s", default="test", type=str,
        help="Dataset split to use")
    parser.add_argument("--dataset-root", "-r", default="data", type=str,
        help="Root directory for input data")
    parser.add_argument("--output", "-o", required=True, type=str,
        help="Root directory for output data")
    parser.add_argument("--image-format", default="webp", type=str,
        help="Image extension for saving: (lossless) webp, png, or jpg (Default: webp)")
    parser.add_argument("--max-count", default=10_000, type=int,
        help="Maximum number of images per TAR shard (Default: 10_000)")
    parser.add_argument("--max-size", default=1_000_000_000, type=int,
        help="Maximum size in bytes per TAR shard (Default: 1_000_000_000)")
    args = parser.parse_args()
    return args

def main():
    args = get_parser_args()
    run(args)

def run(args):
    # Setup dataset folder
    os.makedirs(os.path.join(args.output, args.split), exist_ok=True)
    # Run conversion
    convert_dataset(
        args.dataset,
        args.dataset_root,
        args.split,
        args.output,
        image_format=args.image_format,
        max_count=args.max_count,
        max_size=args.max_size
    )


def PIL_to_bytes(image_format):
    OPTIONS = {
        "webp": dict(format="webp", lossless=True),
        "png": dict(format="png"),
        "jpg": dict(format="jpeg"),
    }
    def transform(image):
        bytestream = io.BytesIO()
        image.save(bytestream, **OPTIONS[image_format])
        return bytestream.getvalue()
    return transform


def convert_dataset(task_name, data_root, split, folder_name, image_format, max_count, max_size):
    VERBOSE = True
    dataset = build_dataset(
        dataset_name=task_name,
        root=data_root,
        split=split,
        transform=PIL_to_bytes(image_format),
        download=True,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=8,
        collate_fn=lambda batch: batch[0] # No collate, only for multiprocessing
    )
    if VERBOSE:
        try:
            print(f"Dataset size: {len(dataset)}")
        except TypeError:
            print("IterableDataset has no len()")
        print(f"Dataset number of classes: {len(dataset.classes)}")
    # Save classnames
    if dataset.classes:
        classnames_fname = os.path.join(folder_name, "classnames.txt")
        with open(classnames_fname, "w") as classnames_file:
            print(*dataset.classes, sep="\n", end="\n", file=classnames_file)
        if VERBOSE:
            print("Saved class names to '%s'" % classnames_fname)
    elif VERBOSE:
        print("WARNING: No class names found")
    # Save zeroshot templates
    if dataset.templates:
        templates_fname = os.path.join(folder_name, "zeroshot_classification_templates.txt")
        with open(templates_fname, "w") as templates_file:
            print(*dataset.templates, sep="\n", end="\n", file=templates_file)
        if VERBOSE:
            print("Saved class names to '%s'" % templates_fname)
    elif VERBOSE:
        print("WARNING: No zeroshot classification templates found")
    # Write to TAR files
    data_fname = os.path.join(folder_name, split, r"%d.tar")
    sink = webdataset.ShardWriter(
        data_fname,
        maxcount=max_count,
        maxsize=max_size
    )
    for index, (input, output) in enumerate(tqdm(dataloader, desc=task_name)):
        sink.write({
            "__key__": "s%07d" % index,
            image_format: input,
            "cls": output,
        })
    num_shards = sink.shard
    sink.close()
    if VERBOSE:
        print("Saved dataset to '%s'" % data_fname.replace(r"%d", "{0..%d}" % (num_shards - 1)))
    # Save number of shards
    nshards_fname = os.path.join(folder_name, split, "nshards.txt")
    with open(nshards_fname, "w") as nshards_file:
        print(num_shards, end="\n", file=nshards_file)
    if VERBOSE:
        print("Saved number of shards = %d to '%s'" % (num_shards, nshards_fname))


if __name__ == "__main__":
    sys.exit(main())
