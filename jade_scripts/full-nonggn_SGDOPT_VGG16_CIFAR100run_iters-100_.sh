#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=nonggn_SGDOPT_VGG16_CIFAR100
#SBATCH --partition=small
#SBATCH --gres=gpu:1
module load python3/anaconda
source activate diegorubin
python3 ../spectrum.py --curvature_matrix=nonggn   --dataset=CIFAR100 --iters=100 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data --model=VGG16 --ckpt=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/ckpts/c100/VGG16/run/checkpoint-00000.pt
python3 ../spectrum.py --curvature_matrix=nonggn   --dataset=CIFAR100 --iters=100 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data --model=VGG16 --ckpt=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/ckpts/c100/VGG16/run/checkpoint-00025.pt
python3 ../spectrum.py --curvature_matrix=nonggn   --dataset=CIFAR100 --iters=100 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data --model=VGG16 --ckpt=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/ckpts/c100/VGG16/run/checkpoint-00050.pt
python3 ../spectrum.py --curvature_matrix=nonggn   --dataset=CIFAR100 --iters=100 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data --model=VGG16 --ckpt=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/ckpts/c100/VGG16/run/checkpoint-00075.pt
python3 ../spectrum.py --curvature_matrix=nonggn   --dataset=CIFAR100 --iters=100 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data --model=VGG16 --ckpt=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/ckpts/c100/VGG16/run/checkpoint-00100.pt
python3 ../spectrum.py --curvature_matrix=nonggn   --dataset=CIFAR100 --iters=100 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data --model=VGG16 --ckpt=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/ckpts/c100/VGG16/run/checkpoint-00125.pt
python3 ../spectrum.py --curvature_matrix=nonggn   --dataset=CIFAR100 --iters=100 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data --model=VGG16 --ckpt=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/ckpts/c100/VGG16/run/checkpoint-00150.pt
python3 ../spectrum.py --curvature_matrix=nonggn   --dataset=CIFAR100 --iters=100 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data --model=VGG16 --ckpt=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/ckpts/c100/VGG16/run/checkpoint-00175.pt
python3 ../spectrum.py --curvature_matrix=nonggn   --dataset=CIFAR100 --iters=100 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data --model=VGG16 --ckpt=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/ckpts/c100/VGG16/run/checkpoint-00200.pt
python3 ../spectrum.py --curvature_matrix=nonggn   --dataset=CIFAR100 --iters=100 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data --model=VGG16 --ckpt=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/ckpts/c100/VGG16/run/checkpoint-00225.pt
python3 ../spectrum.py --curvature_matrix=nonggn   --dataset=CIFAR100 --iters=100 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data --model=VGG16 --ckpt=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/ckpts/c100/VGG16/run/checkpoint-00250.pt
python3 ../spectrum.py --curvature_matrix=nonggn   --dataset=CIFAR100 --iters=100 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data --model=VGG16 --ckpt=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/ckpts/c100/VGG16/run/checkpoint-00275.pt
python3 ../spectrum.py --curvature_matrix=nonggn   --dataset=CIFAR100 --iters=100 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data --model=VGG16 --ckpt=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/ckpts/c100/VGG16/run/checkpoint-00300.pt
