#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set name of job
#SBATCH --job-name=3adwrnimagenet

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

python3 ../run_adam.py --dir=../out/ --dataset=ImageNet32 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model=WideResNet28x10 --epochs=50 --save_freq=10 --use_test --lr_init=0.0028 --batch_size 1024  --eval_freq=1 --wd 1e-3 --decoupled_wd


