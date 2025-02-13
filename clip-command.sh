python clip_benchmark/cli.py eval \
--model_type "transformers" \
--model "openai" \
--pretrained "clip-vit-base-patch32" \
--dataset "imagenet1k" \
--seed 40 \
--task "zeroshot_classification" \
--output "result_{model}_{pretrained_full_path}_{task}_{dataset}_{language}.json" \
--dataset_root "{dataset}" \
--model_cache_dir "."