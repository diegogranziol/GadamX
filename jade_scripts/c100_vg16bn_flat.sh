#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set name of job
#SBATCH --job-name=vgg16bnflat

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
python3 ../run_sgd.py --model VGG16BN --dir out/ --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --dataset=CIFAR100 --epochs=300 --lr_init 0.05 --no_schedule --wd=5e-4
