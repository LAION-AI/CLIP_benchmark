# Benchmark

the benchmark results are available in [benchmark.csv](benchmark.csv).
You can visualize the results in the [notebook](results.ipynb)

# How to reproduce th CLIP benchmark results


## VTAB+ and retrieval datasets (MS-COCO, Flickr30k, Flickr8k)

```bash
clip_benchmark eval --pretrained_model  openai openclip_base  --dataset vtab+ retrieval \
--dataset_root "clip_benchmark_datasets/{dataset}" \
--output "vtab_plus_and_retrieval_{dataset}_{pretrained}_{model}_{language}_{task}.json"
```
(Change `--dataset_root` accordingly)

Once the evaluation finishes, you can construct a CSV with all the results:

```bash
clip_benchmark build vtab_plus_and_retrieval*.json --output=benchmark.csv
```

## Multilingual ImageNet benchmark

To run the multilingual ImageNet benchmark, use:

```bash
clip_benchmark eval --pretrained_model openclip_multilingual openclip_base openai  --dataset imagenet1k --language cn it jp en \
--dataset_root "clip_benchmark_datasets/{dataset}" \
--output "multilingual_{dataset}_{pretrained}_{model}_{language}_{task}.json"
```
(Change `--dataset_root` accordingly)

## Multilingual MS-COCO benchmark

To run the multilingual MS-COCO benchmark, use:

```bash
clip_benchmark eval --pretrained_model openclip_multilingual openclip_base openai --dataset multilingual_mscoco_captions --language es it ko pl ru tr zh en \
--dataset_root "clip_benchmark_datasets/{dataset}" \
--output "multilingual_{dataset}_{pretrained}_{model}_{language}_{task}.json"
```

(Change `--dataset_root` accordingly)
