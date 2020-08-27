#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set name of job
#SBATCH --job-name=gpadamimgnt

#choose partition
#SBATCH --partition=small

# set number of GPUs
#SBATCH --gres=gpu:1

# run the application
module load python3/anaconda
source activate diegorubin

#python3 run_adam.py --dataset=ImageNet32 --data_path=data/ --model=VGG16BN --save_freq=300 --use_test --lr=0.1 --wd 5e-4 --eval_freq=5 --dir out/ --epochs 300 --seed 1 --swag --no_covariance --swag_start 161 --swag_lr 0.05 --use_test

python3 ../run_padam_gap.py --dataset=ImageNet32 --data_path ../../curvature/data/ --model=WideResNet28x10 --save_freq=30 --use_test --lr_init=0.03 --wd 3e-4 --decoupled_wd --eval_freq=1 --dir ../out/ --epochs 50 --seed 5 --linear_annealing --swag_start 30 --swag --no_covariance

#python3 ../run_padam.py --dataset=ImageNet32 --data_path ../../curvature/data/ --model=WideResNet28x10 --save_freq=30 --use_test --lr_init=0.03 --wd 3e-4 --decoupled_wd --eval_freq=1 --dir ../out/ --epochs 50 --seed 6  --linear_annealing --swag_start 30 --swag --no_covariance
