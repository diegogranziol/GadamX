#!/bin/bash
# set the number of nodes
#SBATCH --nodes=1
#SBATCH --job-name=sgd
#SBATCH --partition=small
#SBATCH --gres=gpu:1
module load python3/anaconda
source activate diegorubin
python3 run_lanczos.py --dir /jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/out/ --dataset CIFAR10 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model=VGG16 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 0.003 --wd=1e-05
python3 run_lanczos.py --dir /jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/out/ --dataset CIFAR10 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model=VGG16 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 0.003 --wd=5e-05
python3 run_lanczos.py --dir /jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/out/ --dataset CIFAR10 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model=VGG16 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 0.003 --wd=0.0001
python3 run_lanczos.py --dir /jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/out/ --dataset CIFAR10 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model=VGG16 --epochs=300 --save_freq=25 --eval_freq=1 --lr_init 0.003 --wd=0.0005
