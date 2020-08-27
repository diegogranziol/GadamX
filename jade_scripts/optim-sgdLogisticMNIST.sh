#!/bin/bash
# set the number of nodes
#SBATCH --nodes=1
#SBATCH --job-name=sgd
#SBATCH --partition=small
#SBATCH --gres=gpu:1
module load python3/anaconda
source activate diegorubin
python3 run_sgd.py --dir out/ --dataset MNIST --data_path=data/ --model=Logistic --epochs=100 --save_freq=25 --eval_freq=1 --lr_init 0.003 --wd=0.0001
python3 run_sgd.py --dir out/ --dataset MNIST --data_path=data/ --model=Logistic --epochs=100 --save_freq=25 --eval_freq=1 --lr_init 0.003 --wd=0.0005
python3 run_sgd.py --dir out/ --dataset MNIST --data_path=data/ --model=Logistic --epochs=100 --save_freq=25 --eval_freq=1 --lr_init 0.003 --wd=0
python3 run_sgd.py --dir out/ --dataset MNIST --data_path=data/ --model=Logistic --epochs=100 --save_freq=25 --eval_freq=1 --lr_init 0.003 --wd=0.0001
python3 run_sgd.py --dir out/ --dataset MNIST --data_path=data/ --model=Logistic --epochs=100 --save_freq=25 --eval_freq=1 --lr_init 0.003 --wd=0.0005
python3 run_sgd.py --dir out/ --dataset MNIST --data_path=data/ --model=Logistic --epochs=100 --save_freq=25 --eval_freq=1 --lr_init 0.003 --wd=0
python3 run_sgd.py --dir out/ --dataset MNIST --data_path=data/ --model=Logistic --epochs=100 --save_freq=25 --eval_freq=1 --lr_init 0.01 --wd=0.0001
python3 run_sgd.py --dir out/ --dataset MNIST --data_path=data/ --model=Logistic --epochs=100 --save_freq=25 --eval_freq=1 --lr_init 0.01 --wd=0.0005
python3 run_sgd.py --dir out/ --dataset MNIST --data_path=data/ --model=Logistic --epochs=100 --save_freq=25 --eval_freq=1 --lr_init 0.01 --wd=0
python3 run_sgd.py --dir out/ --dataset MNIST --data_path=data/ --model=Logistic --epochs=100 --save_freq=25 --eval_freq=1 --lr_init 0.03 --wd=0.0001
python3 run_sgd.py --dir out/ --dataset MNIST --data_path=data/ --model=Logistic --epochs=100 --save_freq=25 --eval_freq=1 --lr_init 0.03 --wd=0.0005
python3 run_sgd.py --dir out/ --dataset MNIST --data_path=data/ --model=Logistic --epochs=100 --save_freq=25 --eval_freq=1 --lr_init 0.03 --wd=0
