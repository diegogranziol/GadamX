#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set name of job
#SBATCH --job-name=swawrnimagenet

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

python3 ../run_padam.py  --dataset=ImageNet32 --dir ../out/ --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model=WideResNet28x10 --epochs=50 --save_freq=10 --use_test --lr_init=0.03  --wd=3e-4  --decoupled_wd --eval_freq=1 --swag --no_covariance --swag_lr 0.015 --swag_start 31


