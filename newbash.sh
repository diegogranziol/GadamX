for d in */ ; do
    echo "$d"
    if test -f $d*"hessian"*; then
        echo "Hessian computed"
    else
        if [[ $d == *"1000"* ]]; then
            python3 ../../../../spectrum.py --iters 100 --model Logistic --dataset MNIST --data_path ../../../../data/ --ckpt "$d"checkpoint-01000.pt
        else
            python3 ../../../../spectrum.py --iters 100 --model VGG16 --dataset CIFAR100 --data_path ../../../../data/ --ckpt "$d"checkpoint-00300.pt
    fi
    fi
done
