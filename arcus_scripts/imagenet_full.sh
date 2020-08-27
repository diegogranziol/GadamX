#!/bin/bash

# set the number of nodes
#SBATCH --gres=gpu:v100:8 --C="gpu,gpu_mem:32GB,gpu_sku:V100-LS"
# set name of job
#SBATCH --job-name=ImageNetbig

# run the application
# module load python3/anaconda
#module load python3/anaconda
source activate pytorch

python3 ../run_sgd.py --use_test  --dir ../out/ --dataset=ImageFolder --data_path=/data/parg/chri3937/data/imagefolder --model=resnet50 --epochs=50 --save_freq=1 --lr_init=0.05  --wd=0.0001 --eval_freq=1 --batch_size=256 --verbose


