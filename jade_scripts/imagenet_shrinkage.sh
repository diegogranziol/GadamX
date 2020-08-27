#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set name of job
#SBATCH --job-name=shrinimg

#choose partition
#SBATCH --partition=devel

# set number of GPUs
#SBATCH --gres=gpu:1

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=diego@robots.ox.ac.uk

# run the application
module load python3/anaconda
source activate diegorubin

python3 ../shrinkage_sgd.py  --num_curv_samples=8 --stats_batch 1 --wd_freq=10 --dataset=ImageNet32 --dir ../out/ --data_path=../data/ --model=WideResNet28x10 --epochs=50 --save_freq=10 --use_test --lr_init=0.03 --eval_freq=1 --swag --swag_lr 0.03 --swag_start 31


