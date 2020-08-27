#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=hessian_SGDOPT_PreResNet110_CIFAR10
#SBATCH --partition=small
#SBATCH --gres=gpu:1
# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=xingchen.wan@outlook.com
module load python3/anaconda
source activate diegorubin
python3 ../spectrum.py --curvature_matrix=hessian   --dataset=CIFAR10 --iters=100 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data --model=PreResNet110 --ckpt=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/ckpts/c10/PreResNet110/SGDOPT/runeval1/checkpoint-00000.pt  
python3 ../spectrum.py --curvature_matrix=hessian   --dataset=CIFAR10 --iters=100 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data --model=PreResNet110 --ckpt=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/ckpts/c10/PreResNet110/SGDOPT/runeval1/checkpoint-00000.pt --bn_train_mode_off  
python3 ../spectrum.py --curvature_matrix=hessian   --dataset=CIFAR10 --iters=100 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data --model=PreResNet110 --ckpt=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/ckpts/c10/PreResNet110/SGDOPT/runeval1/checkpoint-00025.pt  
python3 ../spectrum.py --curvature_matrix=hessian   --dataset=CIFAR10 --iters=100 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data --model=PreResNet110 --ckpt=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/ckpts/c10/PreResNet110/SGDOPT/runeval1/checkpoint-00025.pt --bn_train_mode_off  
python3 ../spectrum.py --curvature_matrix=hessian   --dataset=CIFAR10 --iters=100 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data --model=PreResNet110 --ckpt=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/ckpts/c10/PreResNet110/SGDOPT/runeval1/checkpoint-00050.pt  
python3 ../spectrum.py --curvature_matrix=hessian   --dataset=CIFAR10 --iters=100 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data --model=PreResNet110 --ckpt=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/ckpts/c10/PreResNet110/SGDOPT/runeval1/checkpoint-00050.pt --bn_train_mode_off  
python3 ../spectrum.py --curvature_matrix=hessian   --dataset=CIFAR10 --iters=100 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data --model=PreResNet110 --ckpt=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/ckpts/c10/PreResNet110/SGDOPT/runeval1/checkpoint-00075.pt  
python3 ../spectrum.py --curvature_matrix=hessian   --dataset=CIFAR10 --iters=100 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data --model=PreResNet110 --ckpt=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/ckpts/c10/PreResNet110/SGDOPT/runeval1/checkpoint-00075.pt --bn_train_mode_off  
python3 ../spectrum.py --curvature_matrix=hessian   --dataset=CIFAR10 --iters=100 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data --model=PreResNet110 --ckpt=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/ckpts/c10/PreResNet110/SGDOPT/runeval1/checkpoint-00100.pt  
python3 ../spectrum.py --curvature_matrix=hessian   --dataset=CIFAR10 --iters=100 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data --model=PreResNet110 --ckpt=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/ckpts/c10/PreResNet110/SGDOPT/runeval1/checkpoint-00100.pt --bn_train_mode_off  
python3 ../spectrum.py --curvature_matrix=hessian   --dataset=CIFAR10 --iters=100 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data --model=PreResNet110 --ckpt=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/ckpts/c10/PreResNet110/SGDOPT/runeval1/checkpoint-00125.pt  
python3 ../spectrum.py --curvature_matrix=hessian   --dataset=CIFAR10 --iters=100 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data --model=PreResNet110 --ckpt=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/ckpts/c10/PreResNet110/SGDOPT/runeval1/checkpoint-00125.pt --bn_train_mode_off  
python3 ../spectrum.py --curvature_matrix=hessian   --dataset=CIFAR10 --iters=100 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data --model=PreResNet110 --ckpt=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/ckpts/c10/PreResNet110/SGDOPT/runeval1/checkpoint-00150.pt  
python3 ../spectrum.py --curvature_matrix=hessian   --dataset=CIFAR10 --iters=100 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data --model=PreResNet110 --ckpt=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/ckpts/c10/PreResNet110/SGDOPT/runeval1/checkpoint-00150.pt --bn_train_mode_off  
python3 ../spectrum.py --curvature_matrix=hessian   --dataset=CIFAR10 --iters=100 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data --model=PreResNet110 --ckpt=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/ckpts/c10/PreResNet110/SGDOPT/runeval1/checkpoint-00175.pt  
python3 ../spectrum.py --curvature_matrix=hessian   --dataset=CIFAR10 --iters=100 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data --model=PreResNet110 --ckpt=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/ckpts/c10/PreResNet110/SGDOPT/runeval1/checkpoint-00175.pt --bn_train_mode_off  
python3 ../spectrum.py --curvature_matrix=hessian   --dataset=CIFAR10 --iters=100 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data --model=PreResNet110 --ckpt=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/ckpts/c10/PreResNet110/SGDOPT/runeval1/checkpoint-00200.pt  
python3 ../spectrum.py --curvature_matrix=hessian   --dataset=CIFAR10 --iters=100 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data --model=PreResNet110 --ckpt=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/ckpts/c10/PreResNet110/SGDOPT/runeval1/checkpoint-00200.pt --bn_train_mode_off  
python3 ../spectrum.py --curvature_matrix=hessian   --dataset=CIFAR10 --iters=100 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data --model=PreResNet110 --ckpt=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/ckpts/c10/PreResNet110/SGDOPT/runeval1/checkpoint-00225.pt  
python3 ../spectrum.py --curvature_matrix=hessian   --dataset=CIFAR10 --iters=100 --data_path=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/data --model=PreResNet110 --ckpt=/jmain01/home/JAD017/sjr02/dxg49-sjr02/curvature/ckpts/c10/PreResNet110/SGDOPT/runeval1/checkpoint-00225.pt --bn_train_mode_off  
