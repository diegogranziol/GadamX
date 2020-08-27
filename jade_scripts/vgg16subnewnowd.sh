#!/bin/bash
# set the number of nodes
#SBATCH --nodes=1
#SBATCH --job-name=sgdvbn
#SBATCH --partition=small
#SBATCH --gres=gpu:1
module load python3/anaconda
source activate diegorubin

###WITH WEIGHT DECAY

python3 ../run_sgd.py --epochs=300 --wd=0  --dataset=CIFAR100 --data_path ../data/ --lr_init 0.005 --model VGG16 --dir ../out/ --save_freq=100  --save_freq=100

python3 ../run_sgd.py --epochs=300 --wd=0  --dataset=CIFAR100 --data_path ../data/ --lr_init 0.005 --model VGG16 --dir ../out/ --save_freq=100  --swag --swag_start=500 --swag_lr=0.005 --no_covariance --save_freq=100
python3 ../run_sgd.py --epochs=300 --wd=0 --dataset=CIFAR100 --data_path ../data/ --lr_init 0.005 --model VGG16 --dir ../out/ --save_freq=100  --swag --swag_start=500 --swag_lr=0.001 --no_covariance --save_freq=100
python3 ../run_sgd.py --epochs=300 --wd=0  --dataset=CIFAR100 --data_path ../data/ --lr_init 0.005 --model VGG16 --dir ../out/ --save_freq=100  --swag --swag_start=500 --swag_lr=0.0003 --no_covariance --save_freq=100
