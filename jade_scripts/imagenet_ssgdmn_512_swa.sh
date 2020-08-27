#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set name of job
#SBATCH --job-name=512ssgdmnimagenet

#choose partition
#SBATCH --partition=big

# set number of GPUs
#SBATCH --gres=gpu:8

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=diego@robots.ox.ac.uk

# run the application
module load python3/anaconda
source activate diegorubin

# Weight decay = 0
python3 ../run_ssgdmn_fastanddirty.py --model WideResNet28x10 --use_test --curvaturebatchsize=512 --epochfreq=1 --dir out/ --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --dataset=ImageNet32 --epochs=100 --wd=0.0005 --swag --swag_lr 0.1 --swag_start 50 --batch_size =512
