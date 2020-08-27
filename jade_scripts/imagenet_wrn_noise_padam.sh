#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set name of job
#SBATCH --job-name=noisepadam

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

python3 ../loss_stats.py --ckpt=/jmain01/home/JAD017/sjr02/dxg49-sjr02/kfac-curvature/out/ImageNet32/WideResNet28x10/PadamX/seed=1_lr=0.1_wd=0.0001_swa_start=30.0_swa_lr=0.015/checkpoint-00000.pt --dataset=ImageNet32 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model=WideResNet28x10 --use_test

python3 ../loss_stats.py --ckpt=/jmain01/home/JAD017/sjr02/dxg49-sjr02/kfac-curvature/out/ImageNet32/WideResNet28x10/PadamX/seed=1_lr=0.1_wd=0.0001_swa_start=30.0_swa_lr=0.015/checkpoint-00010.pt --dataset=ImageNet32 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model=WideResNet28x10 --use_test

python3 ../loss_stats.py --ckpt=/jmain01/home/JAD017/sjr02/dxg49-sjr02/kfac-curvature/out/ImageNet32/WideResNet28x10/PadamX/seed=1_lr=0.1_wd=0.0001_swa_start=30.0_swa_lr=0.015/checkpoint-00020.pt --dataset=ImageNet32 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model=WideResNet28x10 --use_test

python3 ../loss_stats.py --ckpt=/jmain01/home/JAD017/sjr02/dxg49-sjr02/kfac-curvature/out/ImageNet32/WideResNet28x10/PadamX/seed=1_lr=0.1_wd=0.0001_swa_start=30.0_swa_lr=0.015/checkpoint-00030.pt --dataset=ImageNet32 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model=WideResNet28x10 --use_test

python3 ../loss_stats.py --ckpt=/jmain01/home/JAD017/sjr02/dxg49-sjr02/kfac-curvature/out/ImageNet32/WideResNet28x10/PadamX/seed=1_lr=0.1_wd=0.0001_swa_start=30.0_swa_lr=0.015/swag-00040.pt --dataset=ImageNet32 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model=WideResNet28x10 --use_test --swag

python3 ../loss_stats.py --ckpt=/jmain01/home/JAD017/sjr02/dxg49-sjr02/kfac-curvature/out/ImageNet32/WideResNet28x10/PadamX/seed=1_lr=0.1_wd=0.0001_swa_start=30.0_swa_lr=0.015/swag-00050.pt --dataset=ImageNet32 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model=WideResNet28x10 --use_test --swag

