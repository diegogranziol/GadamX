#!/bin/bash

# set the number of nodes
sbatch --gres=gpu:v100:8 -C "gpu,gpu_mem:32GB,gpu_sku:V100-LS"
# set name of job
#SBATCH --job-name=Adamw

#choose partition
#SBATCH --partition=small

# set number of GPUs
#SBATCH --gres=gpu:1

# run the application
# module load python3/anaconda
module load python3/anaconda
source activate pytorch

python3 ../run_sgd.py --use_test  --dir ../out/ --dataset=ImageFolder --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/data --model=resnet50 --epochs=50 --save_freq=4 --lr_init=0.05  --wd=0.0001 --eval_freq=1 --batch_size=128


