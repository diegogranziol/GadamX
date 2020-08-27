#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=imgnthess
#SBATCH --partition=small
#SBATCH --gres=gpu:1
module load python3/anaconda
source activate diegorubin
python3 ../spectrum.py --curvature_matrix=hessian  --iters=30 --use_test --dataset=ImageNet32 --iters=30 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data/ --model=WideResNet28x10 --ckpt=/jmain01/home/JAD017/sjr02/dxg49-sjr02/kfac-curvature/out/ImageNet32/WideResNet28x10/SGD/seed=1_lr=0.03_mom=0.9_wd=8e-06/checkpoint--00050.pt
