import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from clip_benchmark import cli

if __name__ == '__main__':
    cli.main()