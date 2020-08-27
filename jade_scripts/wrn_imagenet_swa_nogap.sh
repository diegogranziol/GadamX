#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set name of job
#SBATCH --job-name=gadamimgnt

#choose partition
#SBATCH --partition=small

# set number of GPUs
#SBATCH --gres=gpu:1

# run the application
module load python3/anaconda
source activate diegorubin

#python3 run_adam.py --dataset=ImageNet32 --data_path=data/ --model=VGG16BN --save_freq=300 --use_test --lr=0.1 --wd 5e-4 --eval_freq=5 --dir out/ --epochs 300 --seed 1 --swag --no_covariance --swag_start 161 --swag_lr 0.05 --use_test

python3 ../run_swa_nogap.py --ckpt /jmain01/home/JAD017/sjr02/dxg49-sjr02/kfac-curvature/out/ImageNet32/WideResNet28x10/PadamX/seed=1_lr=0.03_wd=0.0003_swa_start=31.0_swa_lr=0.015/checkpoint-00030.pt --dataset=ImageNet32 --data_path ../../curvature/data/ --model=WideResNet28x10 --save_freq=30 --use_test --lr=0.0015 --wd 3e-4 --eval_freq=1 --dir ../out/ --swa_epochs 20

