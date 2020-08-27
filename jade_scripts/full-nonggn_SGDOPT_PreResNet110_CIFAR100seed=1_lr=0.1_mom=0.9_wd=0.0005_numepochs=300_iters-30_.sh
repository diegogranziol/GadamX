#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=nonggn_SGDOPT_PreResNet110_CIFAR100
#SBATCH --partition=small
#SBATCH --gres=gpu:1

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=xdiego@robots.ox.ac.uk
module load python3/anaconda
source activate diegorubin
python3 ../spectrum.py --curvature_matrix=nonggn   --dataset=CIFAR100 --iters=100 --data_path=/nfs/home/dgranziol/kfac-curvature/data/ --model=PreResNet110 --ckpt=/nfs/home/dgranziol/kfac-curvature/out/CIFAR100/VGG16/SGD/seed=1_lr=0.1_mom=0.9_wd=0.0005_numepochs=1000/checkpoint-00000.pt
python3 ../spectrum.py --curvature_matrix=nonggn   --dataset=CIFAR100 --iters=100 --data_path=/nfs/home/dgranziol/kfac-curvature/data/ --model=PreResNet110 --ckpt=/nfs/home/dgranziol/kfac-curvature/out/CIFAR100/VGG16/SGD/seed=1_lr=0.1_mom=0.9_wd=0.0005_numepochs=1000/checkpoint-00000.pt --bn_train_mode_off
python3 ../spectrum.py --curvature_matrix=nonggn   --dataset=CIFAR100 --iters=100 --data_path=/nfs/home/dgranziol/kfac-curvature/data/ --model=PreResNet110 --ckpt=/nfs/home/dgranziol/kfac-curvature/out/CIFAR100/VGG16/SGD/seed=1_lr=0.1_mom=0.9_wd=0.0005_numepochs=1000/checkpoint-00025.pt
python3 ../spectrum.py --curvature_matrix=nonggn   --dataset=CIFAR100 --iters=100 --data_path=/nfs/home/dgranziol/kfac-curvature/data/ --model=PreResNet110 --ckpt=/nfs/home/dgranziol/kfac-curvature/out/CIFAR100/VGG16/SGD/seed=1_lr=0.1_mom=0.9_wd=0.0005_numepochs=1000/checkpoint-00025.pt --bn_train_mode_off
python3 ../spectrum.py --curvature_matrix=nonggn   --dataset=CIFAR100 --iters=100 --data_path=/nfs/home/dgranziol/kfac-curvature/data/ --model=PreResNet110 --ckpt=/nfs/home/dgranziol/kfac-curvature/out/CIFAR100/VGG16/SGD/seed=1_lr=0.1_mom=0.9_wd=0.0005_numepochs=1000/checkpoint-00050.pt
python3 ../spectrum.py --curvature_matrix=nonggn   --dataset=CIFAR100 --iters=100 --data_path=/nfs/home/dgranziol/kfac-curvature/data/ --model=PreResNet110 --ckpt=/nfs/home/dgranziol/kfac-curvature/out/CIFAR100/VGG16/SGD/seed=1_lr=0.1_mom=0.9_wd=0.0005_numepochs=1000/checkpoint-00050.pt --bn_train_mode_off
python3 ../spectrum.py --curvature_matrix=nonggn   --dataset=CIFAR100 --iters=100 --data_path=/nfs/home/dgranziol/kfac-curvature/data/ --model=PreResNet110 --ckpt=/nfs/home/dgranziol/kfac-curvature/out/CIFAR100/VGG16/SGD/seed=1_lr=0.1_mom=0.9_wd=0.0005_numepochs=1000/checkpoint-00075.pt
python3 ../spectrum.py --curvature_matrix=nonggn   --dataset=CIFAR100 --iters=100 --data_path=/nfs/home/dgranziol/kfac-curvature/data/ --model=PreResNet110 --ckpt=/nfs/home/dgranziol/kfac-curvature/out/CIFAR100/VGG16/SGD/seed=1_lr=0.1_mom=0.9_wd=0.0005_numepochs=1000/checkpoint-00075.pt --bn_train_mode_off
python3 ../spectrum.py --curvature_matrix=nonggn   --dataset=CIFAR100 --iters=100 --data_path=/nfs/home/dgranziol/kfac-curvature/data/ --model=PreResNet110 --ckpt=/nfs/home/dgranziol/kfac-curvature/out/CIFAR100/VGG16/SGD/seed=1_lr=0.1_mom=0.9_wd=0.0005_numepochs=1000/checkpoint-00100.pt
python3 ../spectrum.py --curvature_matrix=nonggn   --dataset=CIFAR100 --iters=100 --data_path=/nfs/home/dgranziol/kfac-curvature/data/ --model=PreResNet110 --ckpt=/nfs/home/dgranziol/kfac-curvature/out/CIFAR100/VGG16/SGD/seed=1_lr=0.1_mom=0.9_wd=0.0005_numepochs=1000/checkpoint-00100.pt --bn_train_mode_off
python3 ../spectrum.py --curvature_matrix=nonggn   --dataset=CIFAR100 --iters=100 --data_path=/nfs/home/dgranziol/kfac-curvature/data/ --model=PreResNet110 --ckpt=/nfs/home/dgranziol/kfac-curvature/out/CIFAR100/VGG16/SGD/seed=1_lr=0.1_mom=0.9_wd=0.0005_numepochs=1000/checkpoint-00125.pt
python3 ../spectrum.py --curvature_matrix=nonggn   --dataset=CIFAR100 --iters=100 --data_path=/nfs/home/dgranziol/kfac-curvature/data/ --model=PreResNet110 --ckpt=/nfs/home/dgranziol/kfac-curvature/out/CIFAR100/VGG16/SGD/seed=1_lr=0.1_mom=0.9_wd=0.0005_numepochs=1000/checkpoint-00125.pt --bn_train_mode_off
python3 ../spectrum.py --curvature_matrix=nonggn   --dataset=CIFAR100 --iters=100 --data_path=/nfs/home/dgranziol/kfac-curvature/data/ --model=PreResNet110 --ckpt=/nfs/home/dgranziol/kfac-curvature/out/CIFAR100/VGG16/SGD/seed=1_lr=0.1_mom=0.9_wd=0.0005_numepochs=1000/checkpoint-00150.pt
python3 ../spectrum.py --curvature_matrix=nonggn   --dataset=CIFAR100 --iters=100 --data_path=/nfs/home/dgranziol/kfac-curvature/data/ --model=PreResNet110 --ckpt=/nfs/home/dgranziol/kfac-curvature/out/CIFAR100/VGG16/SGD/seed=1_lr=0.1_mom=0.9_wd=0.0005_numepochs=1000/checkpoint-00150.pt --bn_train_mode_off
python3 ../spectrum.py --curvature_matrix=nonggn   --dataset=CIFAR100 --iters=100 --data_path=/nfs/home/dgranziol/kfac-curvature/data/ --model=PreResNet110 --ckpt=/nfs/home/dgranziol/kfac-curvature/out/CIFAR100/VGG16/SGD/seed=1_lr=0.1_mom=0.9_wd=0.0005_numepochs=1000/checkpoint-00175.pt
python3 ../spectrum.py --curvature_matrix=nonggn   --dataset=CIFAR100 --iters=100 --data_path=/nfs/home/dgranziol/kfac-curvature/data/ --model=PreResNet110 --ckpt=/nfs/home/dgranziol/kfac-curvature/out/CIFAR100/VGG16/SGD/seed=1_lr=0.1_mom=0.9_wd=0.0005_numepochs=1000/checkpoint-00175.pt --bn_train_mode_off
python3 ../spectrum.py --curvature_matrix=nonggn   --dataset=CIFAR100 --iters=100 --data_path=/nfs/home/dgranziol/kfac-curvature/data/ --model=PreResNet110 --ckpt=/nfs/home/dgranziol/kfac-curvature/out/CIFAR100/VGG16/SGD/seed=1_lr=0.1_mom=0.9_wd=0.0005_numepochs=1000/checkpoint-00200.pt
python3 ../spectrum.py --curvature_matrix=nonggn   --dataset=CIFAR100 --iters=100 --data_path=/nfs/home/dgranziol/kfac-curvature/data/ --model=PreResNet110 --ckpt=/nfs/home/dgranziol/kfac-curvature/out/CIFAR100/VGG16/SGD/seed=1_lr=0.1_mom=0.9_wd=0.0005_numepochs=1000/checkpoint-00200.pt --bn_train_mode_off
python3 ../spectrum.py --curvature_matrix=nonggn   --dataset=CIFAR100 --iters=100 --data_path=/nfs/home/dgranziol/kfac-curvature/data/ --model=PreResNet110 --ckpt=/nfs/home/dgranziol/kfac-curvature/out/CIFAR100/VGG16/SGD/seed=1_lr=0.1_mom=0.9_wd=0.0005_numepochs=1000/checkpoint-00225.pt
python3 ../spectrum.py --curvature_matrix=nonggn   --dataset=CIFAR100 --iters=100 --data_path=/nfs/home/dgranziol/kfac-curvature/data/ --model=PreResNet110 --ckpt=/nfs/home/dgranziol/kfac-curvature/out/CIFAR100/VGG16/SGD/seed=1_lr=0.1_mom=0.9_wd=0.0005_numepochs=1000/checkpoint-00225.pt --bn_train_mode_off
