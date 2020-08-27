#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set name of job
#SBATCH --job-name=wideresbase

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
python3 ../run_sgd.py --model WideResNet28x10 --lr_init 0.1 --dir ../out/ --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/kfac-curvature/data/  --dataset=CIFAR10 --epochs=300 --momentum=0 --wd=0
python3 ../run_sgd.py --model WideResNet28x10 --lr_init 1 --dir ../out/ --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/kfac-curvature/data/  --dataset=CIFAR10 --epochs=300 --momentum=0 --wd=0
python3 ../run_sgd.py --model WideResNet28x10 --lr_init 0.1 --dir ../out/ --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/kfac-curvature/data/  --dataset=CIFAR10 --epochs=300 --wd=0

python3 ../run_sgd.py --model WideResNet28x10 --lr_init 0.1 --dir ../out/ --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/kfac-curvature/data/  --dataset=CIFAR10 --epochs=300 --momentum=0 --wd=5e-4
python3 ../run_sgd.py --model WideResNet28x10 --lr_init 1 --dir ../out/ --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/kfac-curvature/data/  --dataset=CIFAR10 --epochs=300 --momentum=0 --wd=5e-4
python3 ../run_sgd.py --model WideResNet28x10 --lr_init 0.1 --dir ../out/ --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/kfac-curvature/data/  --dataset=CIFAR10 --epochs=300 --wd=0 --wd=5e-4


