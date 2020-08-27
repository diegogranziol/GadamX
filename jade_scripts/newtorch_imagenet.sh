#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set name of job
#SBATCH --job-name=newfullimagenet

#choose partition
#SBATCH --partition=big

# set number of GPUs
#SBATCH --gres=gpu:8

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=diego@robots.ox.ac.uk

# run the application
module load python3/anaconda
source activate curvature

python3 ../run_sgd_imagenet.py  --dir ../out/ --dataset=ImageFolder --data_path=kfac-curvature/data --model=resnet50 --epochs=50 --save_freq=4 --lr_init=0.1  --wd=0.0001 --eval_freq=1 --batch_size=256 --save_freq=5 --verbose


