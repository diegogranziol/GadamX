#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set name of job
#SBATCH --job-name=vggsub

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
python3 ../run_sgd_smallsample.py --epochs=3000 --wd=0  --dataset=CIFAR100 --data_path ../data/ --lr_init 0.1 --model VGG16BN --dir ../out/ --sub_sample_size=5000 --sub_sample_seed=4 --save_freq=1000
python3 ../run_sgd_smallsample.py --epochs=3000 --wd=0 --dataset=CIFAR100 --data_path ../data/ --lr_init 0.1 --model VGG16BN --dir ../out/ --sub_sample_size=5000 --sub_sample_seed=5 --save_freq=1000

