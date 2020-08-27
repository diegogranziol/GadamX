#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set name of job
#SBATCH --job-name=vggsubsgd

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
python3 ../run_sgd_smallsample.py --epochs=1000 --wd=5e-4  --dataset=CIFAR100 --data_path ../data/ --lr_init 0.03 --model VGG16 --dir ../out/ --sub_sample_size=5000 --swag --swag_start=500 --swag_lr=0.03 --no_covariance --sub_sample_seed=4 --save_freq=1000
python3 ../run_sgd_smallsample.py --epochs=1000 --wd=5e-4  --dataset=CIFAR100 --data_path ../data/ --lr_init 0.03 --model VGG16 --dir ../out/ --sub_sample_size=5000 --swag --swag_start=500 --swag_lr=0.03 --no_covariance --sub_sample_seed=5 --save_freq=1000

