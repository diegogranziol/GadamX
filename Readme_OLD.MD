# Curvature

```
#SETUP
python3 setup.py develop
```
```
For Running on HSE servers
source /home/tgaripov/env/bin/activate
nvidia-smi -l (choose 1-3 which is free, where l is the number of seconds)
export CUDA_VISIBLE_DEVICES=1
```

## Examples

```bash
#SGD
python3 run_sgd.py --dir=out/ --dataset=CIFAR100 --data_path=data/ --use_test --model=VGG16 --epochs=300 --save_freq=50 --lr_init=0.05 --wd=5e-4 --seed=5123

#SGDSWA
python3 run_sgd.py --dir=out/ --dataset=CIFAR100 --data_path=data/ --use_test --model=VGG16 --epochs=300 --save_freq=50 --lr_init=0.05 --wd=5e-4 --seed=5123 --swag --no_covariance --swag_start=161 --swag_lr=0.01 

#SSGDMN
python3 run_ssgdmnn_fastanddirty.py --curvaturebatchsize=32 --dataset=CIFAR100 --data_path=data/ --use_test --model=PreResNet110 --epochs=300 --save_freq=50 --lr_init=0.05 --wd=5e-4 --seed=5123 --swag --no_covariance --swag_start=161 --swag_lr=0.01 

#SSGDM-min
python3 run_ssgdmn_fastanddirty.py --type='minimum' --curvaturebatchsize=32 --dataset=CIFAR100 --data_path=data/ --use_test --model=PreResNet110 --epochs=300 --save_freq=50 --lr_init=0.05 --wd=5e-4 --seed=5123 --swag --no_covariance --swag_start=161 --swag_lr=0.01 

#SSGDM-med
python3 run_ssgdmn_fastanddirty.py --type='med' --curvaturebatchsize=32 --dataset=CIFAR100 --data_path=data/ --use_test --model=PreResNet110 --epochs=300 --save_freq=50 --lr_init=0.05 --wd=5e-4 --seed=5123 --swag --no_covariance --swag_start=161 --swag_lr=0.01 


#Spectrum
python3 spectrum.py --dataset=CIFAR100 --use_test --data_path=/home/tgaripov/projects/data/ --model=VGG16 --iters=5 --ckpt=./ckpts/c100/vgg16/sgd/run1/checkpoint-00050.pt --spectrum_path=../ckpts/c100/vgg16/sgd/run1/spectrum_50.npz --basis_path=./ckpts/c100/vgg16/sgd/run1/basis_50.pt

#Loss statistics (including hessian variance etc..)
python3 loss_stats.py --dataset=CIFAR100  --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model=PreResNet110 --ckpt=./ckpts/c100/PreResNet110/OFIT/runshrink/checkpoint-00000.pt --stats_batch=256 --save_path=./ckpts/c100/PreResNet110/OFIT/runshrink/checkpoint-00000.npz

#Shrinkage Learner
python3 experiments/swag/shrinkage_swag.py --dataset=CIFAR100 --lr_init=0.1 --epochs=225 --data_path=/jmain01/home/JAD017/sjr01/dxg49-sjr01/curvature/data/ --model=PreResNet110 --epochs=225 --eval_freq=1 --save_freq=50 --num_samples=1024 --wd_freq=20 --subsample_seed=1 --seed=1 --wd_mode_off --dir=./ckpts/c100/PreResNet110/shrinkage_sgd_wd/


```

## SGD 

```bash
python3 experiments/swag/run_swag.py \
                 --dir=<DIR> \
                 --dataset=<DATASET> \
                 --data_path=<PATH> \
                 --model=<MODEL> \
                 --epochs=<EPOCHS> \
                 --lr_init=<LR_INIT> \
                 --momentum=<MOM> \
                 --wd=<WD>                 
```

Parameters:

* ```DIR``` &mdash; path to training directory where checkpoints will be stored
* ```DATASET``` &mdash; dataset name [CIFAR10/CIFAR100] (default: CIFAR10)
* ```PATH``` &mdash; path to the data directory
* ```MODEL``` &mdash; DNN model name:
    - VGG16/VGG16BN/VGG19/VGG19BN
    - PreResNet110/PreResNet164
    - WideResNet28x10
* ```EPOCHS``` &mdash; number of training epochs (default: 200)
* ```LR_INIT``` &mdash; initial learning rate (default: 0.1)
* ```MOM``` &mdash; SGD momentum (default: 0.9)
* ```WD``` &mdash; weight decay (default: 1e-4)

## SWAG

```bash
python3 experiments/swag/run_swag.py \
                 --dir=<DIR> \
                 --dataset=<DATASET> \
                 --data_path=<PATH> \
                 --model=<MODEL> \
                 --epochs=<EPOCHS> \
                 --lr_init=<LR_INIT> \
                 --momentum=<MOM> \
                 --wd=<WD> \
                 --swag \
                 --swag_subspace=<SWAG_SUBSPACE> \
                 --swag_rank=<SWAG_RANK> \
                 --swag_start=<SWAG_START> \
                 --swag_lr=<SWAG_LR> \                 
```

Parameters:

* ```DIR``` &mdash; path to training directory where checkpoints will be stored
* ```DATASET``` &mdash; dataset name [CIFAR10/CIFAR100] (default: CIFAR10)
* ```PATH``` &mdash; path to the data directory
* ```MODEL``` &mdash; DNN model name:
    - VGG16/VGG16BN/VGG19/VGG19BN
    - PreResNet110/PreResNet164
    - WideResNet28x10
* ```EPOCHS``` &mdash; number of training epochs (default: 200)
* ```LR_INIT``` &mdash; initial learning rate (default: 0.1)
* ```MOM``` &mdash; SGD momentum (default: 0.9)
* ```WD``` &mdash; weight decay (default: 1e-4)
* ```SWAG_SUBSPACE``` &mdash; subspace to use in swag:
    - pca (default)
    - covariance
    - freq_dir
* ```SWAG_RANK``` &mdash; rank of the SWAG subspace (default: 20)
* ```SWAG_START``` &mdash; the number of epoch starting from which SWAG will start to collect models (default: 161)
* ```SWAG_LR``` &mdash; SWA learning rate (default: 0.02)

## Spectrum

```bash
python3 experiments/lanczos/spectrum.py \
                 --dir=<DIR> \
                 --dataset=<DATASET> \
                 --data_path=<PATH> \
                 --model=<MODEL> \
                 --ckpt=<CKPT> \
                 [--swag] \
                 --iters=<ITERS> \
                 --spectrum_path=<SPECTRUM_PATH> \
                 --basis_path=<BASIS_PATH>
```

* ```CKPT``` &mdash; path to model checkpoint (use ```--swag``` to load SWAG checkpoint)
* ```ITERS``` &mdash; number of Lanczos iterations
* ```SPECTRUM_PATH``` &mdash; path to use to save spectrum ```.npz``` file
* ```BASIS_PATH``` &mdash; path to use to save basis (using ```torch.save```)

# Curvature

```
#SETUP
python3 setup.py develop
```
```
For Running on HSE servers
source /home/tgaripov/env/bin/activate
nvidia-smi -l (choose 1-3 which is free, where l is the number of seconds)
export CUDA_VISIBLE_DEVICES=1
```
