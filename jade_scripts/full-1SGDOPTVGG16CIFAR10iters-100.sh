#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=speclanc
#SBATCH --partition=small
#SBATCH --gres=gpu:1
source activate diegorubin
python3 ../spectrum.py --curvature_matrix=hessian  --dataset=CIFAR100 --iters=100 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model=VGG16 --ckpt=/jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out/CIFAR100/VGG16/Lanczos/lr=0.001_matrix=gn_damping=0.01_wd=0.0001_batch_size=128/checkpoint-00300.pt
