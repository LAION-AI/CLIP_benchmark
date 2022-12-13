RUN=clip_benchmark
DATASETS=$(cat datasets.txt)
MODELS=$(cat models.txt)
BS=128
WORKERS=4
MODEL=$1
PRETRAINED=$2
ROOT=clip_benchmark_datasets
for ds in $DATASETS;do
    LANGUAGE=$(echo $ds,|cut -d, -f2)
    ds=$(echo $ds|cut -d, -f1)
    ds_name=`echo $ds|tr '/' '_'`
    if [[ "$LANGUAGE" == "" ]]
    then
	LANGUAGE="en"
    fi
    for model in $MODELS;do
       arch=$(echo $model|cut -d, -f1)
       pretrained=$(echo $model|cut -d, -f2)
       echo $ds $pretrained $arch
       if [[ "$ds_name" == "flickr30k" || "$ds_name" == "flickr8k" || "$ds_name" == "mscoco_captions" || "$ds_name" == "multilingual_mscoco_captions" ]]
       then
               TASK=zeroshot_retrieval
       else
               TASK=zeroshot_classification
       fi
       echo "$ds_name"
       echo "$LANGUAGE"
       echo $TASK
       $RUN --dataset=$ds --language=$LANGUAGE --annotation_file=$ROOT/$ds/captions.txt --dataset_root $ROOT/$ds --task=$TASK --pretrained=$pretrained --model=$arch --output="${ds_name}_${name}_${arch}_${LANGUAGE}.json"  --batch_size=$BS --num_workers=$WORKERS
    done
done
python build_csv.py
