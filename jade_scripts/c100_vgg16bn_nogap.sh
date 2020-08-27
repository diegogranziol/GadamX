#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set name of job
#SBATCH --job-name=vgwg

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

python3 ../run_swa_gap.py --ckpt=/jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out/CIFAR100/VGG16BN/SGDSWA/seed=1_lr=0.1_swalr=0.05_mom=0.9_wd=0.0005_numepochs=225/checkpoint-00150.pt --dataset=CIFAR100 --data_path=--data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model=VGG16BN --swa_epochs=75 --save_freq=25 --lr=0.05  --eval_freq=1


