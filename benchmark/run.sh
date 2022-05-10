RUN=clip_benchmark
DATASETS=$(cat ds.txt)
MODELS=$(cat models.txt)
BS=128
WORKERS=4
MODEL=$1
PRETRAINED=$2
ROOT=clip_benchmark_datasets
for ds in $DATASETS;do
    ds_name=`echo $ds|tr '/' '_'`
    for model in $MODELS;do
        arch=$(echo $model|cut -d, -f1)
        pretrained=$(echo $model|cut -d, -f2)
        echo $ds $pretrained $arch
        $RUN --dataset=$ds --dataset_root $ROOT/$ds --task=zeroshot_classification --pretrained=$pretrained --model=$arch --output="${ds_name}_${pretrained}_${model}.json"  --batch_size=$BS --num_workers=$WORKERS
    done
done
python build_csv.py
