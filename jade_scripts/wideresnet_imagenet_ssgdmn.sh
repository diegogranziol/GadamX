#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set name of job
#SBATCH --job-name=ssgdmn164

#choose partition
#SBATCH --partition=devel

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
python3 ../run_ssgdmn_fastanddirty.py --model WideResNet28x10 --curvaturebatchsize=32 --epochfreq=20 --dir=../out/ --data_path=../data/  --dataset=ImageNet32 --epochs=50 --wd=5e-4


