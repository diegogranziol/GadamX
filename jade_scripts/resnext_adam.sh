#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1
#SBATCH --time=23:59:59

# set name of job
#SBATCH --job-name=adam

#choose partition
#SBATCH --partition=small

# set number of GPUs
#SBATCH --gres=gpu:1

# ../run the application
module load python3/anaconda
source activate diegorubin

python3 ../run_adam.py --dataset=CIFAR100 --data_path=data/ --model=ResNeXt29CIFAR --save_freq=10 --use_test --lr=0.0003 --wd 0 --eval_freq=1 --dir out/ --epochs 300

python3 ../run_sgd.py --dataset=CIFAR100 --data_path=data/ --model=ResNeXt29CIFAR --save_freq=10 --use_test --lr=0.05 --wd 5e-4 --eval_freq=1 --dir out/ --epochs 300 --swag --swag_start 161 --swag_lr 0.025 --no_covariance

