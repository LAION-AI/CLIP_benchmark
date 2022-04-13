# CLIP Benchmark


![https://img.shields.io/pypi/v/clip_benchmark.svg](https://pypi.python.org/pypi/clip_benchmark)

![https://img.shields.io/travis/mehdidc/clip_benchmark.svg](https://travis-ci.com/mehdidc/clip_benchmark)


The goal of this repo is to evaluate CLIP-like models on a standard set
of datasets on different tasks such as zero-shot classification and zero-shot
retrieval.

## Features

* Support for zero-shot classification and zero-shot retrieval
* Support for [OpenCLIP](https://github.com/mlfoundations/open_clip) pre-trained models

## How to install?

`pip install clip_benchmark`

In order to use the CLI, you also need to install [OpenCLIP](https://github.com/mlfoundations/open_clip) using
`pip install open_clip_torch`.

## How to use?

### Command line interface (CLI)

The easiest way to to benchmark the models is using the CLI, `clip_benchmark`.
You can specify the model to use, the dataset and the task to evaluate on. Once it is done, evaluation is run and
the results are written into a JSON file.

 Here is an example for CIFAR-10 zero-shot classification using OpenCLIP's pre-trained model on LAION-400m:

 `clip_benchmark --dataset=cifar10 --task=zeroshot_classification --pretrained=laion400m_e32 --model=ViT-B-32-quickgelu --output=result.json --batch_size=64`

Here is the content of `result.json` after the evaluation is done:

```json
{"dataset": "cifar10", "model": "ViT-B-32-quickgelu", "pretrained": "laion400m_e32", "task": "zeroshot_classification", "metrics": {"top1_zeroshot_accuracy": 0.9074, "top5_zeroshot_accuracy": 0.998}}
```

 Here is an example for COCO captions zero-shot retrieval:

 `clip_benchmark --dataset=mscoco_captions --task=zeroshot_rertrieval --pretrained=laion400m_e32 --model=ViT-B-32-quickgelu --output=result.json --root=<PATH_TO_IMAGE_FOLDER> --annotation_file=<PATH_TO_ANNOTATION_FILE> --batch_size=64` 
 
 (see <https://cocodataset.org/#home> for instructions on how to download)

### API

You can also use the API directly. This is especially useful if your model
does not belong to currently supported models.
(TODO)

## Credits

- Thanks to [OpenCLIP](https://github.com/mlfoundations/open_clip) authors, zero-shot accuracy code and pre-trained models in the CLIP are using OpenCLIP.
- Thanks to [SLIP](https://github.com/facebookresearch/SLIP) authors, zero-shot templates and classnames are used from there.
- This package was created with [Cookiecutter]( https://github.com/audreyr/cookiecutter) and the [audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage) project template. Thanks to the author.
