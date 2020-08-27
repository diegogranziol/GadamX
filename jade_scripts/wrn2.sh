#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=hessian_SGDOPT_WideResNet28x10_CIFAR100
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=diego@robots.ox.ac.uk
module load python3/anaconda
source activate diegorubin
python3 ../spectrum.py --curvature_matrix=hessian   --dataset=CIFAR100 --iters=100 --data_path=../data/ --model=WideResNet28x10 --ckpt=/jmain01/home/JAD017/sjr02/dxg49-sjr02/kfac-curvature/out/CIFAR100/WideResNet28x10/SGD/seed=1_lr=0.1_mom=0.9_wd=0.0005_batchsize=128_numepochs=300/checkpoint-00125.pt --basis
python3 ../spectrum.py --curvature_matrix=hessian   --dataset=CIFAR100 --iters=100 --data_path=../data/ --model=WideResNet28x10 --ckpt=/jmain01/home/JAD017/sjr02/dxg49-sjr02/kfac-curvature/out/CIFAR100/WideResNet28x10/SGD/seed=1_lr=0.1_mom=0.9_wd=0.0005_batchsize=128_numepochs=300/checkpoint-00150.pt --basis
python3 ../spectrum.py --curvature_matrix=hessian   --dataset=CIFAR100 --iters=100 --data_path=../data/ --model=WideResNet28x10 --ckpt=/jmain01/home/JAD017/sjr02/dxg49-sjr02/kfac-curvature/out/CIFAR100/WideResNet28x10/SGD/seed=1_lr=0.1_mom=0.9_wd=0.0005_batchsize=128_numepochs=300/checkpoint-00175.pt --basis
python3 ../spectrum.py --curvature_matrix=hessian   --dataset=CIFAR100 --iters=100 --data_path=../data/ --model=WideResNet28x10 --ckpt=/jmain01/home/JAD017/sjr02/dxg49-sjr02/kfac-curvature/out/CIFAR100/WideResNet28x10/SGD/seed=1_lr=0.1_mom=0.9_wd=0.0005_batchsize=128_numepochs=300/checkpoint-00200.pt --basis
