#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set name of job
#SBATCH --job-name=ssgdmnwres

#choose partition
#SBATCH --partition=big

# set number of GPUs
#SBATCH --gres=gpu:4

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=diego@robots.ox.ac.uk

# run the application
module load python3/anaconda
source activate diegorubin

# Weight decay = 0
python3 ../run_ssgdmn_fastanddirty.py --model WideResNet28x10 --curvaturebatchsize=32 --epochfreq=1 --dir out/ --data_path=/data/  --dataset=CIFAR100 --epochs=300 --wd=5e-4 --swag --swag_start 200 --swag_lr 0.1


