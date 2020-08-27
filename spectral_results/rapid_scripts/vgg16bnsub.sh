python3 ../run_sgd_smallsample.py --epochs=3000 --wd=0  --dataset=CIFAR100 --data_path ../data/ --lr_init 0.1 --model VGG16BN --dir out --sub_sample_size=5000 --sub_sample_seed=1
python3 ../run_sgd_smallsample.py --epochs=3000 --wd=0  --dataset=CIFAR100 --data_path ../data/ --lr_init 0.1 --model VGG16BN --dir out --sub_sample_size=5000 --sub_sample_seed=2
python3 ../run_sgd_smallsample.py --epochs=3000 --wd=0  --dataset=CIFAR100 --data_path ../data/ --lr_init 0.1 --model VGG16BN --dir out --sub_sample_size=5000 --sub_sample_seed=3

python3 ../run_sgd_smallsample.py --epochs=3000 --wd=0  --dataset=CIFAR100 --data_path ../data/ --lr_init 0.1 --model VGG16BN --dir out --sub_sample_size=5000 --swag --swag_start=161 --swag_lr=0.01 --no_covariance --sub_sample_seed=1
python3 ../run_sgd_smallsample.py --epochs=3000 --wd=0  --dataset=CIFAR100 --data_path ../data/ --lr_init 0.1 --model VGG16BN --dir out --sub_sample_size=5000 --swag --swag_start=161 --swag_lr=0.01 --no_covariance --sub_sample_seed=2
python3 ../run_sgd_smallsample.py --epochs=3000 --wd=0  --dataset=CIFAR100 --data_path ../data/ --lr_init 0.1 --model VGG16BN --dir out --sub_sample_size=5000 --swag --swag_start=161 --swag_lr=0.01 --no_covariance --sub_sample_seed=3

###WITH WEIGHT DECAY

python3 ../run_sgd_smallsample.py --epochs=3000 --wd=5e-4  --dataset=CIFAR100 --data_path ../data/ --lr_init 0.1 --model VGG16BN --dir out --sub_sample_size=5000 --sub_sample_seed=1
python3 ../run_sgd_smallsample.py --epochs=3000 --wd=5e-4 --dataset=CIFAR100 --data_path ../data/ --lr_init 0.1 --model VGG16BN --dir out --sub_sample_size=5000 --sub_sample_seed=2
python3 ../run_sgd_smallsample.py --epochs=3000 --wd=5e-4  --dataset=CIFAR100 --data_path ../data/ --lr_init 0.1 --model VGG16BN --dir out --sub_sample_size=5000 --sub_sample_seed=3

python3 ../run_sgd_smallsample.py --epochs=3000 --wd=5e-4  --dataset=CIFAR100 --data_path ../data/ --lr_init 0.1 --model VGG16BN --dir out --sub_sample_size=5000 --swag --swag_start=161 --swag_lr=0.01 --no_covariance --sub_sample_seed=1
python3 ../run_sgd_smallsample.py --epochs=3000 --wd=5e-4 --dataset=CIFAR100 --data_path ../data/ --lr_init 0.1 --model VGG16BN --dir out --sub_sample_size=5000 --swag --swag_start=161 --swag_lr=0.01 --no_covariance --sub_sample_seed=2
python3 ../run_sgd_smallsample.py --epochs=3000 --wd=5e-4  --dataset=CIFAR100 --data_path ../data/ --lr_init 0.1 --model VGG16BN --dir out --sub_sample_size=5000 --swag --swag_start=161 --swag_lr=0.01 --no_covariance --sub_sample_seed=3
