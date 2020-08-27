#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set name of job
#SBATCH --job-name=32ssgdmnimagenet

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
python3 ../run_ssgdmn_fastanddirty.py --model WideResNet28x10 --use_test --curvaturebatchsize=32 --epochfreq=1 --dir out/ --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --wd_freq=1  --dataset=ImageNet32 --epochs=50 --wd=0.0005
