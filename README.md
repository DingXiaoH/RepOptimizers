# RepOptimizers

This is the official repository of [Re-parameterizing Your Optimizers rather than Architectures](https://arxiv.org/abs/2205.15242).

Will update with the model weights, more use cases and a detailed README in two days.

You may reproduce RepOpt-VGG-B1 by

```
1. mkdir output
```
```
2. export CUDA_VISIBLE_DEVICES=0
```

Hyper-Search. The trained model will be saved to output/RepOpt-VGG-B1-hs/hyper-search/latest.pth
```
3. python3 -m torch.distributed.launch --nproc_per_node 1 --master_port 12349 main_repopt.py --arch RepOpt-VGG-B1-hs --batch-size 256 --tag hyper-search --opts TRAIN.EPOCHS 240 TRAIN.BASE_LR 0.1 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 5 MODEL.LABEL_SMOOTHING 0.0 DATA.DATASET cf100 DATA.DATA_PATH /path/to/CIFAR100
```

Use 8 GPUs
```
4. export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

Train on ImageNet
```
5. python3 -m torch.distributed.launch --nproc_per_node 8 --master_port 12349 main_repopt.py --arch RepOpt-VGG-B1-target --batch-size 32 --tag target --scales-path output/RepOpt-VGG-B1-hs/hyper-search/latest.pth --opts TRAIN.EPOCHS 120 TRAIN.BASE_LR 0.1 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 5 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET imagenet DATA.DATA_PATH /path/to/ImageNet
```
