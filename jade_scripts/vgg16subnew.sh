#!/bin/bash
# set the number of nodes
#SBATCH --nodes=1
#SBATCH --job-name=sgdvbn
#SBATCH --partition=small
#SBATCH --gres=gpu:1
module load python3/anaconda
source activate diegorubin

###WITH WEIGHT DECAY

#python3 ../run_sgd.py --epochs=300 --wd=5e-4  --dataset=CIFAR100 --data_path ../data/ --lr_init 0.05 --model VGG16 --dir ../out/ --save_freq=100  --save_freq=100

python3 ../run_sgd.py --epochs=300 --wd=5e-4  --dataset=CIFAR100 --data_path ../data/ --lr_init 0.05 --model VGG16 --dir ../out/ --save_freq=100  --swag --swag_start=161 --swag_lr=0.05 --no_covariance --save_freq=100 
python3 ../run_sgd.py --epochs=300 --wd=5e-4 --dataset=CIFAR100 --data_path ../data/ --lr_init 0.05 --model VGG16 --dir ../out/ --save_freq=100  --swag --swag_start=161 --swag_lr=0.01 --no_covariance --save_freq=100
python3 ../run_sgd.py --epochs=300 --wd=5e-4  --dataset=CIFAR100 --data_path ../data/ --lr_init 0.05 --model VGG16 --dir ../out/ --save_freq=100  --swag --swag_start=161 --swag_lr=0.003 --no_covariance --save_freq=100 --resume /nfs/home/dgranziol/kfac-curvature/out/CIFAR100/VGG16/SGDSWA/seed=1_lr=0.05_swalr=0.003_mom=0.9_wd=0.0005_numepochs=300/checkpoint-00100.pt

python3 ../run_sgd.py --epochs=300 --wd=5e-4 --dataset=CIFAR100 --data_path ../data/ --lr_init 0.05 --model VGG16 --dir ../out/ --save_freq=100  --swag --swag_start=161 --swag_lr=0.03 --no_covariance --save_freq=100
