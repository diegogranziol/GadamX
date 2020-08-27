#!/bin/bash
# set the number of nodes
#SBATCH --nodes=1
#SBATCH --job-name=gnlanc2
#SBATCH --partition=small
#SBATCH --gres=gpu:1

# send mail to this address
#SBATCH --mail-user=diego@robots.ox.ac.uk

module load python3/anaconda
source activate diegorubin

python3 ../run_lanczos.py --dir ../out/  --dataset CIFAR100 --data_path ../data/ --model VGG16 --lr_init 1  --lanczos_beta=0.256 --epochs 100 --save_freq=10 --eval_freq=1 --lanczos_steps=20 --matrix_type='gn'  --wd=0 --lanczos_batch=256
python3 ../run_lanczos.py --dir ../out/  --dataset CIFAR100 --data_path ../data/ --model VGG16 --lr_init 1  --lanczos_beta=0.512 --epochs 100 --save_freq=10 --eval_freq=1 --lanczos_steps=20 --matrix_type='gn'  --wd=0 --lanczos_batch=256
python3 ../run_lanczos.py --dir ../out/  --dataset CIFAR100 --data_path ../data/ --model VGG16 --lr_init 1  --lanczos_beta=1.024 --epochs 100 --save_freq=10 --eval_freq=1 --lanczos_steps=20 --matrix_type='gn'  --wd=0 --lanczos_batch=256
python3 ../run_lanczos.py --dir ../out/  --dataset CIFAR100 --data_path ../data/ --model VGG16 --lr_init 1  --lanczos_beta=2.048 --epochs 100 --save_freq=10 --eval_freq=1 --lanczos_steps=20 --matrix_type='gn'  --wd=0 --lanczos_batch=256
python3 ../run_lanczos.py --dir ../out/  --dataset CIFAR100 --data_path ../data/ --model VGG16 --lr_init 1  --lanczos_beta=4.096 --epochs 100 --save_freq=10 --eval_freq=1 --lanczos_steps=20 --matrix_type='gn'  --wd=0 --lanczos_batch=256
