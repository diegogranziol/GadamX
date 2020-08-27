#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set name of job
#SBATCH --job-name=noisesgd

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

python3 ../loss_stats.py --ckpt=/jmain01/home/JAD017/sjr02/dxg49-sjr02/kfac-curvature/out/ImageNet32/WideResNet28x10/SGD/seed=1_lr=0.03_mom=0.9_wd=8e-06/checkpoint-00000.pt --dataset=ImageNet32 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model=WideResNet28x10 --use_test

python3 ../loss_stats.py --ckpt=/jmain01/home/JAD017/sjr02/dxg49-sjr02/kfac-curvature/out/ImageNet32/WideResNet28x10/SGD/seed=1_lr=0.03_mom=0.9_wd=8e-06/checkpoint-00010.pt --dataset=ImageNet32 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model=WideResNet28x10 --use_test

python3 ../loss_stats.py --ckpt=/jmain01/home/JAD017/sjr02/dxg49-sjr02/kfac-curvature/out/ImageNet32/WideResNet28x10/SGD/seed=1_lr=0.03_mom=0.9_wd=8e-06/checkpoint-00020.pt --dataset=ImageNet32 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model=WideResNet28x10 --use_test

python3 ../loss_stats.py --ckpt=/jmain01/home/JAD017/sjr02/dxg49-sjr02/kfac-curvature/out/ImageNet32/WideResNet28x10/SGD/seed=1_lr=0.03_mom=0.9_wd=8e-06/checkpoint-00030.pt --dataset=ImageNet32 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model=WideResNet28x10 --use_test

python3 ../loss_stats.py --ckpt=/jmain01/home/JAD017/sjr02/dxg49-sjr02/kfac-curvature/out/ImageNet32/WideResNet28x10/SGD/seed=1_lr=0.03_mom=0.9_wd=8e-06/checkpoint-00040.pt --dataset=ImageNet32 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model=WideResNet28x10 --use_test

python3 ../loss_stats.py --ckpt=/jmain01/home/JAD017/sjr02/dxg49-sjr02/kfac-curvature/out/ImageNet32/WideResNet28x10/SGD/seed=1_lr=0.03_mom=0.9_wd=8e-06/checkpoint-00050.pt --dataset=ImageNet32 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model=WideResNet28x10 --use_test


