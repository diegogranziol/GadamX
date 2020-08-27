#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set name of job
#SBATCH --job-name=3_swd

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

python3 ../run_sgd.py --dir=../out/ --dataset=CIFAR100 --data_path=../data/ --model=AllCNN_CIFAR100 --epochs=300 --save_freq=25 --lr_init=0.1 --batch_size 128  --eval_freq=1 --wd 1e-4


