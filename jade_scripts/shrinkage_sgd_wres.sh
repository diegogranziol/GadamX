#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=hessian_preres
#SBATCH --partition=small
#SBATCH --gres=gpu:1

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=diego@robots.ox.ac.uk

python3 ../shrinkage_grad_sgd.py --model WideResNet28x10 --lr_init=0.1 --dataset CIFAR100 --data_path ../data/ --dir ../out/ --wd_freq 1 --num_curv_samples 1000
python3 ../shrinkage_grad_sgd.py --model WideResNet28x10 --lr_init=0.1 --dataset CIFAR100 --data_path ../data/ --dir ../out/ --wd_freq 1 --num_curv_samples 1000 --swag --swag_lr 0.05 --swag_start 150
python3 ../shrinkage_grad_sgd.py --model WideResNet28x10 --lr_init=0.1 --dataset CIFAR100 --data_path ../data/ --dir ../out/ --wd_freq 1 --num_curv_samples 1000 --swag --swag_lr 0.03 --swag_start 150

