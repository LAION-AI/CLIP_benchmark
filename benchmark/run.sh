RUN=clip_benchmark
DATASETS=$(cat datasets.txt)
MODELS=$(cat models.txt)
BS=128
WORKERS=4
MODEL=$1
PRETRAINED=$2
ROOT=clip_benchmark_datasets
LANGUAGE=en
for ds in $DATASETS;do
    ds_name=`echo $ds|tr '/' '_'`
    for model in $MODELS;do
       arch=$(echo $model|cut -d, -f1)
       pretrained=$(echo $model|cut -d, -f2)
       echo $ds $pretrained $arch
       if [[ "$ds_name" == "flickr30k" || "$ds_name" == "flickr8k" || "$ds_name" == "mscoco_captions" ]]
       then
               TASK=zeroshot_retrieval
       else
               TASK=zeroshot_classification
       fi
       echo "$ds_name"
       echo $TASK
       $RUN --dataset=$ds --language=$LANGUAGE --annotation_file=$ROOT/$ds/captions.txt --dataset_root $ROOT/$ds --task=$TASK --pretrained=$pretrained --model=$arch --output="${ds_name}_${name}_${arch}.json"  --batch_size=$BS --num_workers=$WORKERS
    done
done
python build_csv.py
