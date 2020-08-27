#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=nonggn_SGDOPT_WideResNet28x10_CIFAR100
#SBATCH --partition=small
#SBATCH --gres=gpu:1
python3 spectrum.py --curvature_matrix=nonggn   --dataset=CIFAR100 --iters=20 --data_path=data/ --model=VGG16BN --ckpt=/nfs/home/dgranziol/kfac-curvature/out/CIFAR100/VGG16BN/SGD/seed=1_lr=0.1_mom=0.9_wd=0.0005_batchsize=128_numepochs=300/checkpoint-00300.pt
python3 spectrum.py --curvature_matrix=gn   --dataset=CIFAR100 --iters=20 --data_path=data/ --model=VGG16BN --ckpt=/nfs/home/dgranziol/kfac-curvature/out/CIFAR100/VGG16BN/SGD/seed=1_lr=0.1_mom=0.9_wd=0.0005_batchsize=128_numepochs=300/checkpoint-00300.pt
python3 spectrum.py --curvature_matrix=hessian   --dataset=CIFAR100 --iters=20 --data_path=data/ --model=VGG16BN --ckpt=/nfs/home/dgranziol/kfac-curvature/out/CIFAR100/VGG16BN/SGD/seed=1_lr=0.1_mom=0.9_wd=0.0005_batchsize=128_numepochs=300/checkpoint-00300.pt
