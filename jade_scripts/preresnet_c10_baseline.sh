#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set name of job
#SBATCH --job-name=ssgdmn164

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

# Weight decay = 0

python3 ../run_sgd.py --model PreResNet110 --lr_init 0.1 --dir out/ --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/  --dataset=CIFAR10 --epochs=300
python3 ../run_sgd.py --model PreResNet110 --swag --no_covariance --lr_init 0.05 --no_schedule --dir out/ --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/  --dataset=CIFAR10 --epochs=300
python3 ../run_sgd.py --model PreResNet110 --swag --no_covariance --lr_init 0.1  --no_schedule --dir out/ --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/  --dataset=CIFAR10 --epochs=300
python3 ../run_sgd.py --model PreResNet110 --swag --no_covariance --lr_init 0.1 --swag_lr 0.05 --dir out/ --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/  --dataset=CIFAR10 --epochs=300

python3 ../run_sgd.py --model PreResNet110 --lr_init 0.1 --dir out/ --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/  --dataset=CIFAR10 --epochs=300 --wd=0
python3 ../run_sgd.py --model PreResNet110 --swag --no_covariance --lr_init 0.05 --no_schedule --dir out/ --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/  --dataset=CIFAR10 --epochs=300 --wd=0
python3 ../run_sgd.py --model PreResNet110 --swag --no_covariance --lr_init 0.1  --no_schedule --dir out/ --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/  --dataset=CIFAR10 --epochs=300 --wd=0
python3 ../run_sgd.py --model PreResNet110 --swag --no_covariance --lr_init 0.1 --swag_lr 0.05 --dir out/ --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/  --dataset=CIFAR10 --epochs=300 --wd=0
