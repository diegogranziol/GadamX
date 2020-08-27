#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set name of job
#SBATCH --job-name=swagapwrnimagenet

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

python3 ../run_swa_gap.py --ckpt=/jmain01/home/JAD017/sjr02/dxg49-sjr02/kfac-curvature/out/ImageNet32/WideResNet28x10/SGDSWA/seed=1_lr=0.03_mom=0.9_wd=8e-06_start=10.0_slr=0.03/checkpoint-00040.pt --dataset=ImageNet32 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model=WideResNet28x10 --swa_epochs=40 --save_freq=10 --use_test --lr=0.03  --eval_freq=1


