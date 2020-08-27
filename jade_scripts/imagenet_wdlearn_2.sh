#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set name of job
#SBATCH --job-name=2imagenetwd

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

python3 ../shrinkage_sgd.py --model WideResNet28x10 --use_test --stats_batch=128 --curvature_matrix hessian --lr_init 0.03 --num_curv_samples=10000 --no_schedule --wd_freq=10 --dir out/ --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/  --dataset=ImageNet32 --epochs=50
