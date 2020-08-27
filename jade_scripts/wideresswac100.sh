#!/bin/bash
# set the number of nodes
#SBATCH --nodes=1
#SBATCH --job-name=wideswa
#SBATCH --partition=small
#SBATCH --gres=gpu:1
module load python3/anaconda
source activate diegorubin
python3 ../run_sgd.py --dir ../out/ --dataset CIFAR100 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model=WideResNet28x10 --epochs=225 --save_freq=25 --eval_freq=1 --lr_init 0.1 --wd=5e-04 --swag --swag_start 150 --swag_lr 0.05 --no_covariance
