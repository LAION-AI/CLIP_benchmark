## History

### 1.7.0

- Add audio-text (CLAP) evaluation on MAEB, MSEB and HEAR (#154)
  (`pip install "clip_benchmark[audio]"`, MSEB is installed separately from GitHub, see README)
- Migrate packaging from `setup.py` to `pyproject.toml`
- Require Python 3.12 or higher (VTAB datasets need Python 3.11 or lower, use 1.6.2 or earlier for them)
- Remove the `vtab` extra, as `task_adaptation` cannot be installed on Python 3.12 or higher
- Fix pickling error with Python 3.14 (forkserver is the default start method)
- Fix TypeError in zero-shot classification ("only 0-dimensional arrays can be converted to Python scalars")
- `clip_benchmark eval` no longer defaults to `cifar10` and `ViT-B-32-quickgelu`/`laion400m_e32`: `--dataset` and the model (`--model`/`--pretrained` or `--pretrained_model`) are now required

### 1.6.2

- Separate results from clip benchmark repo 
- Update default output file to {dataset}_{pretrained}_{model}_{language}_{task}.json, otherwise same result.json file overwritten
- Fix imagenetv2 dead urls
- Add XTD_10 dataset
- Fix issues (#131, #125, #138)

### 1.6.1

* Fix missing sugar crepe example #119 thanks to @samarth4149

### 1.6.0


* Fix overwritten zeroshot templates issue (https://github.com/LAION-AI/CLIP_benchmark/issues/109)
* Support new multilingual retrieval datasets:  Crossmodal-3600, XTD10, Flickr30k-200, and XTD200
* Support tuning linear probing on validation set

### 1.5.0

* Custom classnames and templates
* support wds for captioning evaluation
* support imagenet-w
* support babel imagenet
* support chinese flickr30k/8k
* support sugar crepe (compositionality)
* support (optional) sharding evaluation based on rank, for parallel runs
* fix many issues

### 1.4.0

* Fix silent webdataset error-handling
* Added support for wds/voc2007_multilabel 
* default to float32 
* add mscoco generative benchmark

### 1.3.0

* update flickr8k results, solve issue #48, thanks to @orchidmajumder
* Evaluate multiple models/datasets/languages using the CLI directly
* Support Japanese CLIP by rinna
* Add arabic imagenet
* updating CuPL prompts with more generated sentences + ensembled with openAI prompts
* put model in eval mode before evaluation
* Webdataset updates
* Make verbose the default

### 1.2.0

* Added support for loading webdatasets

### 1.1.0

* Added better support for multilingual eval
* Added better support for linear probing
* Added support for CuPL prompts

### 1.0.1

* pypi description as markdown

### 1.0.0

* Actual first release on PyPI.


### 0.1.0

* First release on PyPI.
