#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set name of job
#SBATCH --job-name=swa2imgnet32sgd

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

#python3 ../run_swag.py --dir=/jmain01/home/JAD017/sjr02/dxg49-sjr02/kfac-curvature/out --dataset=ImageNet32 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model=PreResNet110 --epochs=225 --save_freq=25 --use_test --lr_init=0.05 --eval_freq=1 --swag_lr=0.005 --wd=8e-6 --swag_start=176 --swag
python3 ../run_swag.py --dir=/jmain01/home/JAD017/sjr02/dxg49-sjr02/kfac-curvature/out --resume=/jmain01/home/JAD017/sjr02/dxg49-sjr02/kfac-curvature/out/ImageNet32/PreResNet110/SWA/lr=0.05_wd=8e-06_swastart=176.0_swalr=0.005/checkpoint-00170.pt --dataset=ImageNet32 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model=PreResNet110 --epochs=225 --save_freq=25 --use_test --lr_init=0.1 --eval_freq=1 --swag_lr=0.025 --wd=8e-6 --swag_start=176 --swag
python3 ../run_swag.py --dir=/jmain01/home/JAD017/sjr02/dxg49-sjr02/kfac-curvature/out --resume=/jmain01/home/JAD017/sjr02/dxg49-sjr02/kfac-curvature/out/ImageNet32/PreResNet110/SWA/lr=0.05_wd=8e-06_swastart=176.0_swalr=0.005/checkpoint-00170.pt --dataset=ImageNet32 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model=PreResNet110 --epochs=225 --save_freq=25 --use_test --lr_init=0.1 --eval_freq=1 --swag_lr=0.005 --wd=0 --swag_start=176 --swag
