#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set name of job
#SBATCH --job-name=fullimagenet

#choose partition
#SBATCH --partition=devel

# set number of GPUs
#SBATCH --gres=gpu:8

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=diego@robots.ox.ac.uk

# run the application
module load python3/anaconda
source activate diegorubin

python3 ../run_sgd.py  --dir ../out/ --dataset=ImageNet --data_path=../data/ --model=WideResNet28x10 --epochs=50 --save_freq=10 --use_test --lr_init=0.05  --wd=0 --eval_freq=1 --swag --no_covariance --swag_lr 0.05 --swag_start 31 --batch_size=1


