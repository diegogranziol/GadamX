#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=hessian_SGDOPT_VGG16_CIFAR100
#SBATCH --partition=small
#SBATCH --gres=gpu:1
module load python3/anaconda
source activate diegorubin
python3 ../spectrum.py --curvature_matrix=hessian   --dataset=CIFAR100 --iters=100 --data_path=../data/ --model=VGG16 --ckpt=/jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out/CIFAR100/VGG16BN/SGD/seed=1_lr=1.0_mom=0.0_wd=0.0_batchsize=128_numepochs=300/checkpoint-00000.pt
python3 ../spectrum.py --curvature_matrix=hessian   --dataset=CIFAR100 --iters=100 --data_path=../data/ --model=VGG16 --ckpt=/jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out/CIFAR100/VGG16BN/SGD/seed=1_lr=1.0_mom=0.0_wd=0.0_batchsize=128_numepochs=300/checkpoint-00025.pt
python3 ../spectrum.py --curvature_matrix=hessian   --dataset=CIFAR100 --iters=100 --data_path=../data/ --model=VGG16 --ckpt=/jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out/CIFAR100/VGG16BN/SGD/seed=1_lr=1.0_mom=0.0_wd=0.0_batchsize=128_numepochs=300/checkpoint-00050.pt
python3 ../spectrum.py --curvature_matrix=hessian   --dataset=CIFAR100 --iters=100 --data_path=../data/ --model=VGG16 --ckpt=/jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out/CIFAR100/VGG16BN/SGD/seed=1_lr=1.0_mom=0.0_wd=0.0_batchsize=128_numepochs=300/checkpoint-00075.pt
python3 ../spectrum.py --curvature_matrix=hessian   --dataset=CIFAR100 --iters=100 --data_path=../data/ --model=VGG16 --ckpt=/jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out/CIFAR100/VGG16BN/SGD/seed=1_lr=1.0_mom=0.0_wd=0.0_batchsize=128_numepochs=300/checkpoint-00100.pt
python3 ../spectrum.py --curvature_matrix=hessian   --dataset=CIFAR100 --iters=100 --data_path=../data/ --model=VGG16 --ckpt=/jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out/CIFAR100/VGG16BN/SGD/seed=1_lr=1.0_mom=0.0_wd=0.0_batchsize=128_numepochs=300/checkpoint-00125.pt
python3 ../spectrum.py --curvature_matrix=hessian   --dataset=CIFAR100 --iters=100 --data_path=../data/ --model=VGG16 --ckpt=/jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out/CIFAR100/VGG16BN/SGD/seed=1_lr=1.0_mom=0.0_wd=0.0_batchsize=128_numepochs=300/checkpoint-00150.pt
python3 ../spectrum.py --curvature_matrix=hessian   --dataset=CIFAR100 --iters=100 --data_path=../data/ --model=VGG16 --ckpt=/jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out/CIFAR100/VGG16BN/SGD/seed=1_lr=1.0_mom=0.0_wd=0.0_batchsize=128_numepochs=300/checkpoint-00175.pt
python3 ../spectrum.py --curvature_matrix=hessian   --dataset=CIFAR100 --iters=100 --data_path=../data/ --model=VGG16 --ckpt=/jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out/CIFAR100/VGG16BN/SGD/seed=1_lr=1.0_mom=0.0_wd=0.0_batchsize=128_numepochs=300/checkpoint-00200.pt
python3 ../spectrum.py --curvature_matrix=hessian   --dataset=CIFAR100 --iters=100 --data_path=../data/ --model=VGG16 --ckpt=/jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out/CIFAR100/VGG16BN/SGD/seed=1_lr=1.0_mom=0.0_wd=0.0_batchsize=128_numepochs=300/checkpoint-00225.pt
python3 ../spectrum.py --curvature_matrix=hessian   --dataset=CIFAR100 --iters=100 --data_path=../data/ --model=VGG16 --ckpt=/jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out/CIFAR100/VGG16BN/SGD/seed=1_lr=1.0_mom=0.0_wd=0.0_batchsize=128_numepochs=300/checkpoint-00250.pt
python3 ../spectrum.py --curvature_matrix=hessian   --dataset=CIFAR100 --iters=100 --data_path=../data/ --model=VGG16 --ckpt=/jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out/CIFAR100/VGG16BN/SGD/seed=1_lr=1.0_mom=0.0_wd=0.0_batchsize=128_numepochs=300/checkpoint-00275.pt
python3 ../spectrum.py --curvature_matrix=hessian   --dataset=CIFAR100 --iters=100 --data_path=../data/ --model=VGG16 --ckpt=/jmain01/home/JAD017/sjr01/dxg49-sjr01/kfac-curvature/out/CIFAR100/VGG16BN/SGD/seed=1_lr=1.0_mom=0.0_wd=0.0_batchsize=128_numepochs=300/checkpoint-00300.pt
