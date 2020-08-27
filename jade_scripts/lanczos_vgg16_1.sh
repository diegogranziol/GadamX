#!/bin/bash
# set the number of nodes
#SBATCH --nodes=1
#SBATCH --job-name=gnlanc
#SBATCH --partition=small
#SBATCH --gres=gpu:1
module load python3/anaconda
source activate diegorubin

python3 ../run_lanczos.py --dir ../out/  --dataset CIFAR100 --data_path ../data/ --model VGG16 --lr_init 1 --lanczos_beta=0.0001 --epochs 100 --save_freq=10 --eval_freq=1 --lanczos_steps=20 --matrix_type='gn'  --wd=0
python3 ../run_lanczos.py --dir ../out/  --dataset CIFAR100 --data_path ../data/ --model VGG16 --lr_init 1  --lanczos_beta=0.001 --epochs 100 --save_freq=10 --eval_freq=1 --lanczos_steps=20 --matrix_type='gn'  --wd=0
python3 ../run_lanczos.py --dir ../out/  --dataset CIFAR100 --data_path ../data/ --model VGG16 --lr_init 1  --lanczos_beta=0.01 --epochs 100 --save_freq=10 --eval_freq=1 --lanczos_steps=20 --matrix_type='gn'  --wd=0
python3 ../run_lanczos.py --dir ../out/  --dataset CIFAR100 --data_path ../data/ --model VGG16 --lr_init 1  --lanczos_beta=0.1 --epochs 100 --save_freq=10 --eval_freq=1 --lanczos_steps=20 --matrix_type='gn'  --wd=0
