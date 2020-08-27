#!/bin/bash
# set the number of nodes
#SBATCH --nodes=1
#SBATCH --job-name=lanczos
#SBATCH --partition=small
#SBATCH --gres=gpu:1
module load python3/anaconda
source activate diegorubin
python3 run_lanczos.py --dir out/ --dataset CIFAR10 --data_path=data/ --model=VGG16 --epochs=100 --save_freq=25 --eval_freq=1 --lr_init 0.0003 --lanczos_beta=1e-4 --wd=0
