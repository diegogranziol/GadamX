#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=gn_SGDOPT_WideResNet28x10_CIFAR100
#SBATCH --partition=small
#SBATCH --gres=gpu:1

python3 ../spectrum.py --curvature_matrix=gn   --dataset=CIFAR100 --iters=100 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data --model=WideResNet28x10 --ckpt=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/ckpts/c100/WideResNet28x10/SGDOPT/run/checkpoint-00225.pt --basis
python3 ../spectrum.py --curvature_matrix=gn   --dataset=CIFAR100 --iters=100 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data --model=WideResNet28x10 --ckpt=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/ckpts/c100/WideResNet28x10/SGDOPT/run/checkpoint-00225.pt --bn_train_mode_off --basis

python3 ../spectrum.py --curvature_matrix=hessian   --dataset=CIFAR100 --iters=100 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data --model=WideResNet28x10 --ckpt=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/ckpts/c100/WideResNet28x10/SGDOPT/run/checkpoint-00225.pt --basis
python3 ../spectrum.py --curvature_matrix=hessian   --dataset=CIFAR100 --iters=100 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data --model=WideResNet28x10 --ckpt=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/ckpts/c100/WideResNet28x10/SGDOPT/run/checkpoint-00225.pt --bn_train_mode_off --basis


python3 ../spectrum.py --curvature_matrix=nonggn   --dataset=CIFAR100 --iters=100 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data --model=WideResNet28x10 --ckpt=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/ckpts/c100/WideResNet28x10/SGDOPT/run/checkpoint-00225.pt --basis
python3 ../spectrum.py --curvature_matrix=nonggn   --dataset=CIFAR100 --iters=100 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data --model=WideResNet28x10 --ckpt=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/ckpts/c100/WideResNet28x10/SGDOPT/run/checkpoint-00225.pt --bn_train_mode_off --basis
