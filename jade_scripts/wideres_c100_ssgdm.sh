#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set name of job
#SBATCH --job-name=ssgdm164

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
python3 ../run_ssgdm_fastanddirty.py --model WideResNet28x10 --curvaturebatchsize=32 --epochfreq=1 --dir ../out/ --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/  --dataset=CIFAR100 --epochs=300
python3 ../run_ssgdm_fastanddirty.py --model WideResNet28x10 --curvaturebatchsize=32 --epochfreq=1 --dir ../out/ --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/  --dataset=CIFAR100 --epochs=300 --wd=5e-4
python3 ../run_ssgdm_fastanddirty.py --model WideResNet28x10 --curvaturebatchsize=32 --epochfreq=1 --dir ../out/ --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/  --dataset=CIFAR100 --epochs=300 --wd=1e-3
python3 ../run_ssgdm_fastanddirty.py --model WideResNet28x10 --curvaturebatchsize=32 --epochfreq=1 --dir ../out/ --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/  --dataset=CIFAR100 --epochs=300 --wd=0

python3 ../run_ssgdm_fastanddirty.py --model WideResNet28x10 --curvaturebatchsize=32 --epochfreq=1 --dir ../out/ --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/  --dataset=CIFAR100 --epochs=300 --type='medium'
python3 ../run_ssgdm_fastanddirty.py --model WideResNet28x10 --curvaturebatchsize=32 --epochfreq=1 --dir ../out/ --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/  --dataset=CIFAR100 --epochs=300 --type='medium' --wd=5e-4
python3 ../run_ssgdm_fastanddirty.py --model WideResNet28x10 --curvaturebatchsize=32 --epochfreq=1 --dir ../out/ --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/  --dataset=CIFAR100 --epochs=300 --type='medium' --wd=1e-3
python3 ../run_ssgdm_fastanddirty.py --model WideResNet28x10 --curvaturebatchsize=32 --epochfreq=1 --dir ../out/ --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/  --dataset=CIFAR100 --epochs=300 --type='medium' --wd=0






