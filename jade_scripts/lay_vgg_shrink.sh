#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set name of job
#SBATCH --job-name=layvggs

#choose partition
#SBATCH --partition=small

# set number of GPUs
#SBATCH --gres=gpu:1

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=diego@robots.ox.ac.uk

# run the application
module load python3/anaconda
source activate diegorubin

# Weight decay = 0
python3 ../shrinkage_swag_layerwise.py --model VGG16BN --stats_batch=128 --lr_init 0.1 --swag_lr 0.1 --dir ../out/ --data_path=../data/  --dataset=CIFAR100 --epochs 300 --swag --swag_start 161 --wd_freq=20

