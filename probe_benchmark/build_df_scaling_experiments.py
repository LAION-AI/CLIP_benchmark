import os
import pandas as pd
import json

if __name__ == '__main__':

    compute_df = pd.read_csv('probe_benchmark/clip_table_2.csv')
    info = []

    models = ['ViT-B-32-quickgelu,laion400m_e32',
              'ViT-B-32,openai',
              'ViT-B-32,laion2b_s34b_b79k',
              'ViT-B-16,laion400m_e32',
              'ViT-B-16-plus-240,laion400m_e32',
              'ViT-B-16,openai',
              #'ViT-L-14-336,openai',
              'ViT-L-14,openai',
              'ViT-B-32,laion2b_e16',
              'ViT-L-14,laion400m_e32',
              'ViT-L-14,laion2b_s32b_b82k',
              'ViT-H-14,laion2b_s32b_b79k',
            ]

    datasets = ['imagenet1k-unverified', 'cifar100']

    ks = [10, 25, -1]
    lrs = [0.1, 0.01, 0.001]
    epoch_vals = [10, 20, 40]
    batch_sizes = [32 * 8]

    for dataset in datasets:
      dataset_root = '/datasets01/imagenet_full_size/061417' if dataset.startswith('imagenet') else '/private/home/mitchellw/git/forks/CLIP_benchmark'
      for model_info in models:
          model_info_split = model_info.split(',')
          model, pretrained = model_info_split[0], model_info_split[1]
          for epochs in epoch_vals:
            for k in ks:
              for lr in lrs:
                for bs in batch_sizes:
                  pth = '/private/home/mitchellw/git/forks/CLIP_benchmark/probe_benchmark/data/' + f'{model}-{pretrained}-{dataset}-{epochs}-{k}-{lr}-{bs}.json'.replace('/', '_')
                  assert os.path.exists(pth)
                  row = {
                    'k' : k,
                    'lr' : lr,
                    'bs' : bs,
                    'epochs' : epochs,
                    'model' : model,
                    'pretrained' : pretrained,
                    'pretrained_short' : 'laion2b' if 'laion2b' in pretrained else pretrained,
                    'dataset' : dataset,
                    'macts' : compute_df[compute_df.model == model.replace('-quickgelu', '')]['image_macts'].values[0]
                  }
                  with open(pth, 'r') as f:
                    row.update(json.load(f)['metrics'])
                info.append(row)
    
    with open('probe_benchmark/scaling_experiment_data.json', 'w') as f:
      json.dump(info, f)
                  
