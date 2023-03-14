#  ACCELERATING DIFFUSION SAMPLING WITH CLASSIFIER-BASED FEATURE DISTILLATION
## Environment
Python 3.6.13, torch 1.9.0



## Training

### Train the base model
```
python -m torch.distributed.launch --nproc_per_node=4 train_base.py \
    --flagfile ./config/CIFAR10_BASE.txt \
    --gpu-id 0,1,2,3 --logdir ./logs/CIFAR10/1024
```

### Distill using PD
```
python -m torch.distributed.launch --nproc_per_node=4 PD.py \
    --flagfile ./config/CIFAR10_PD.txt --gpu-id 0,1,2,3 \
    --logdir ./logs/CIFAR10/512 --base_ckpt ./logs/CIFAR10/1024

...

python -m torch.distributed.launch --nproc_per_node=4 PD.py \
    --flagfile ./config/CIFAR10_PD.txt \
    --logdir ./logs/CIFAR10/4 --base_ckpt ./logs/CIFAR10/8
```

### To use RCFD, train the classifier using classifier/train.py first

```
python train.py --model densenet201
```

### Distill using RCFD

```
python -m torch.distributed.launch --nproc_per_node=4 RCFD.py \
    --flagfile ./config/CIFAR10_RCFD.txt --gpu-id 0,1,2,3 \
    --logdir ./logs/CIFAR10/4_densenet201 --base_ckpt ./logs/CIFAR10/8 \
    --classifier densenet201 --classifier_path ./classifier/result/cifar10/densenet201 \
    --temp 0.95 --alpha 0.003 --beta 0.75
```



## Evaluation

### To eval, run score/get_npz.py first

```
python get_npz.py --dataset cifar10
```

### Eval 8-step DDIM

```
python ddim_eval.py --flagfile ./config/CIFAR10_EVAL.txt \
										--logdir ./logs/CIFAR10/1024 --stride 128
```

### Eval 4-step PD

```
python ddim_eval.py --flagfile ./config/CIFAR10_EVAL.txt \
										--logdir ./logs/CIFAR10/4
```

### Eval 4-step RCFD

```
python ddim_eval.py --flagfile ./config/CIFAR10_EVAL.txt \
										--logdir ./logs/CIFAR10/4_densenet201
```

