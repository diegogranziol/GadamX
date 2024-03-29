#!/bin/bash
# set the number of nodes
#SBATCH --nodes=1
#SBATCH --job-name=gnlanc
#SBATCH --partition=small
#SBATCH --gres=gpu:1
module load python3/anaconda
source activate diegorubin


python3 ../run_lanczos.py --resume /jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out/CIFAR100/VGG16/Lanczos/lr=0.001_matrix=gn_damping=0.01_wd=0.0001_batch_size=128/checkpoint-00040.pt --dir /jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out  --dataset CIFAR100 --data_path /jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model VGG16 --lr_init 0.001 --lanczos_beta=0.01 --epochs 100 --save_freq=5 --eval_freq=1 --lanczos_steps=30 --matrix_type='gn' --wd=0.0001
