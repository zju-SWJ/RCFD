#  ACCELERATING DIFFUSION SAMPLING WITH CLASSIFIER-BASED FEATURE DISTILLATION
## Paper
[https://arxiv.org/abs/2211.12039](https://arxiv.org/abs/2211.12039)

## Environment
Python 3.6.13, torch 1.9.0



## Training

### Train the base model
```
python -m torch.distributed.launch --nproc_per_node=4 train_base.py \
    --flagfile ./config/CIFAR10_BASE.txt \
    --gpu_id 0,1,2,3 --logdir ./logs/CIFAR10/1024
```

### Distill using PD
```
python -m torch.distributed.launch --nproc_per_node=4 PD.py \
    --flagfile ./config/CIFAR10_PD.txt --gpu_id 0,1,2,3 \
    --logdir ./logs/CIFAR10/512 --base_ckpt ./logs/CIFAR10/1024

...

python -m torch.distributed.launch --nproc_per_node=4 PD.py \
    --flagfile ./config/CIFAR10_PD.txt --gpu_id 0,1,2,3 \
    --logdir ./logs/CIFAR10/4 --base_ckpt ./logs/CIFAR10/8
```

### To use RCFD, train the classifier using classifier/train.py first

```
python train.py --model densenet201
python train.py --model resnet18
```

### Distill using RCFD

```
python -m torch.distributed.launch --nproc_per_node=4 RCFD.py \
    --flagfile ./config/CIFAR10_RCFD.txt --gpu_id 0,1,2,3 \
    --logdir ./logs/CIFAR10/4_resnet18 --base_ckpt ./logs/CIFAR10/8 \
    --classifier resnet18 --classifier_path ./classifier/result/cifar10/resnet18 \
    --temp 0.95 --alpha 0.003 --beta 0.75
    # alpha here is actually the beta in the paper, and beta here is actually the gamma in the paper

python -m torch.distributed.launch --nproc_per_node=4 RCFD.py \
    --flagfile ./config/CIFAR10_RCFD.txt --gpu_id 0,1,2,3 \
    --logdir ./logs/CIFAR10/4_densenet201 --base_ckpt ./logs/CIFAR10/8 \
    --classifier densenet201 --classifier_path ./classifier/result/cifar10/densenet201 \
    --temp 0.9 --alpha 0
```



## Evaluation

### To eval, run score/get_npz.py first or download from [google drive](https://drive.google.com/drive/folders/1UBdzl6GtNMwNQ5U-4ESlIer43tNjiGJC?usp=sharing)

```
python get_npz.py --dataset cifar10
```

### Eval
```
# 8-step DDIM
python ddim_eval.py --flagfile ./config/CIFAR10_EVAL.txt --logdir ./logs/CIFAR10/1024 --stride 128
# 4-step PD
python ddim_eval.py --flagfile ./config/CIFAR10_EVAL.txt --logdir ./logs/CIFAR10/4
# 4-step RCFD
python ddim_eval.py --flagfile ./config/CIFAR10_EVAL.txt --logdir ./logs/CIFAR10/4_densenet201/temp0.9/alpha0
python ddim_eval.py --flagfile ./config/CIFAR10_EVAL.txt --logdir ./logs/CIFAR10/4_resnet18/temp0.95/alpha0.003/beta0.75
```

## Citation
If you find this repository useful, please consider citing the following paper:
```
@article{sun2022accelerating,
  title={Accelerating Diffusion Sampling with Classifier-based Feature Distillation},
  author={Sun, Wujie and Chen, Defang and Wang, Can and Ye, Deshi and Feng, Yan and Chen, Chun},
  journal={arXiv preprint arXiv:2211.12039},
  year={2022}
}
```

## Acknowledgment
This codebase is heavily borrowed from [pytorch-ddpm](https://github.com/w86763777/pytorch-ddpm) and [diffusion_distiller](https://github.com/Hramchenko/diffusion_distiller).
