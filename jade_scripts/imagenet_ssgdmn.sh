#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set name of job
#SBATCH --job-name=1ssgdmnimagenet

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
python3 ../run_ssgdmn_fastanddirty.py --model WideResNet28x10 --use_test --curvaturebatchsize=32 --epochfreq=1 --dir out/ --data_path=data/  --dataset=ImageNet32 --epochs=50 --swag --no_covariance --swag_start 30 --swag_lr 0.1
