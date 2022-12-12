# RepOptimizers

(Dec 23rd: code for reproducing RepOpt-GhostNet)

This is the official repository of [Re-parameterizing Your Optimizers rather than Architectures](https://arxiv.org/abs/2205.15242).

If you find the paper or this repository helpful, please consider citing

        @article{ding2022re,
        title={Re-parameterizing Your Optimizers rather than Architectures},
        author={Ding, Xiaohan and Chen, Honghao and Zhang, Xiangyu and Huang, Kaiqi and Han, Jungong and Ding, Guiguang},
        journal={arXiv preprint arXiv:2205.15242},
        year={2022}
        }


# Highlights

RepOptimizer and RepOpt-VGG have been used in **YOLOv6** ([paper](https://arxiv.org/abs/2209.02976), [code](https://github.com/meituan/YOLOv6)) and **deployed in business**. The methodology of Structural Re-parameterization also plays a critical role in **YOLOv7** ([paper](https://arxiv.org/abs/2207.02696), [code](https://github.com/WongKinYiu/yolov7)).

# Catalog
- [x] Code
- [x] PyTorch pretrained models
- [x] PyTorch training code

<!-- ✅ ⬜️  -->

# Verify the equivalency GR = CSLA

Tired of reading proof? We provide a script demonstrating the equivalency of GR = CSLA in **both SGD and AdamW** cases.

You may run the following script to verify the equivalency (GR = CSLA) on an experimental model. You can run without a GPU.

```
python check_equivalency.py
```


# Design

RepOptimizers currently support two update rules (SGD with momentum and AdamW) and two models (RepOpt-VGG and [RepOpt-GhostNet](https://https://arxiv.org/abs/2211.06088)). While re-designing the code of RepOptimizer, I decided to separate the update-rule-related behaviors and model-specific behaviors.

The key components of the new implementation (please see ```repoptimizer/```) include

**Model**: ```repoptvgg_model.py``` and ```repoptghostnet_model.py``` define the model architecutres, including the target and search structures.

**Model-specific Handler**: a ```RepOptimizerHandler``` defines the model-specific behavior of RepOptimizer given the searched scales, which include 1) re-initializing the model (i.e., Rule of Initialization) and 2) generating the Grad Mults (i.e., Rule of Iteration).

For example, ```RepOptVGGHandler``` (see ```repoptvgg_impl.py```) implements the formulas presented in the paper.

**Update rule**: ```repoptimizer_sgd.py``` and ```repoptimizer_adamw.py``` define the behavior of RepOptimizers based on different update rules. The differences between a RepOptimizer and its regular counterpart (```torch.optim.SGD``` or ```torch.optim.AdamW```) include 

1. RepOptimizers take one more argument, ```grad_mult_map```, which is the output from RepOptimizerHandler and will be stored in memory. It is a dict where the key is the parameter (```torch.nn.Parameter```) and the value is the corresponding Grad Mult (```torch.Tensor```).

2. In the ```step``` function, RepOptimizers will use the Grad Mults properly. For SGD, please see [here](https://github.com/DingXiaoH/RepOptimizers/blob/main/repoptimizer/repoptimizer_sgd.py#L38). For AdamW, please see [here](https://github.com/DingXiaoH/RepOptimizers/blob/main/repoptimizer/repoptimizer_adamw.py#L145) and [here](https://github.com/DingXiaoH/RepOptimizers/blob/main/repoptimizer/repoptimizer_adamw.py#L274).


# Pre-trained RepOpt-VGG Models

We have released the models pre-trained with this codebase.

| name | ImageNet-1K acc | #params | download |
|:---:|:---:|:---:|:---:|
|RepOpt-VGG-B1|  78.62  |  51.8M  | [Google Drive](https://drive.google.com/file/d/1kBDue-19AG0Rm2NaS5h6aTl-ZHvdPe5K/view?usp=sharing), [Baidu Cloud](https://pan.baidu.com/s/1Zs-eStqDEQIfymGFnIwR-A?pwd=rvgg) |
|RepOpt-VGG-B2|  79.68  |  80.3M  | [Google Drive](https://drive.google.com/file/d/1o4enC7tFr0nORJHEE9vaHh8mHpusFvRE/view?usp=sharing), [Baidu Cloud](https://pan.baidu.com/s/1kPKFvi-BNL_GOdbnYlCBzg?pwd=rvgg)   |
|RepOpt-VGG-L1|  79.82  |  76.0M   | [Google Drive](https://drive.google.com/file/d/19wd13WgBK6LtyLVA_N9ZjFYaLwduywnm/view?usp=sharing), [Baidu Cloud](https://pan.baidu.com/s/1CsbNRqGZIPuejavxaeClGQ?pwd=rvgg) |
|RepOpt-VGG-L2|  80.47  |  118.1M  | [Google Drive](https://drive.google.com/file/d/1PG0sSqOTRdnVoBS_ZBKPShcOIyHNLze6/view?usp=sharing), [Baidu Cloud](https://pan.baidu.com/s/1D5KuqjcXGW-CsdNm9UZzvQ?pwd=rvgg) |


# Use cases

The following cases use RepOpt-VGG-B1 as an example. You may replace ```RepOpt-VGG-B1``` by ```RepOpt-VGG-B2```, ```RepOpt-VGG-L1```, or ```RepOpt-VGG-L2``` as you wish.

## Evaluation

You may test our released models by
```
python -m torch.distributed.launch --nproc_per_node {your_num_gpus} --master_port 12349 main_repopt.py --arch RepOpt-VGG-B1-target --tag test --eval --resume RepOpt-VGG-B1-acc78.62.pth --data-path /path/to/imagenet --batch-size 32 --opts DATA.DATASET imagenet
```

## Training

To reproduce RepOpt-VGG-B1, you may build a RepOptimizer with our released constants ```RepOpt-VGG-B1-scales.pth``` and train a RepOpt-VGG-B1 with it.
```
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12349 main_repopt.py --data-path /path/to/imagenet --arch RepOpt-VGG-B1-target --batch-size 32 --tag experiment --scales-path RepOpt-VGG-B1-scales.pth --opts TRAIN.EPOCHS 120 TRAIN.BASE_LR 0.1 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 5 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET imagenet
```
The log and weights will be saved to ```output/RepOpt-VGG-B1-target/experiment/```

## Hyper-Search

Besides using our released scales, you may Hyper-Search by
```
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12349 main_repopt.py --data-path /path/to/search/dataset --arch RepOpt-VGG-B1-hs --batch-size 32 --tag search --opts TRAIN.EPOCHS 240 TRAIN.BASE_LR 0.1 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 10 MODEL.LABEL_SMOOTHING 0.1 DATA.DATASET cf100 TRAIN.CLIP_GRAD 5.0
```

(Note that since the model seems too big for such a small dataset, we use grad clipping to stablize the training. But do not use grad clipping while training with RepOptimizer! That would break the equivalency.)

The weights of the search model will be saved to ```output/RepOpt-VGG-B1-hs/search/latest.pth```

Then you may train with it by
```
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12349 main_repopt.py --data-path /path/to/imagenet --arch RepOpt-VGG-B1-target --batch-size 32 --tag experiment --scales-path output/RepOpt-VGG-B1-hs/search/latest.pth --opts TRAIN.EPOCHS 120 TRAIN.BASE_LR 0.1 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 5 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET imagenet
```

## Use RepOptimizer and RepOpt-VGG in your code

Given the searched scales (saved in a ```.pth``` file), you may conveniently build a RepOptimizer and a RepOpt-VGG model and use them just like you use the common optimizers and models.

Please see ```build_RepOptVGG_and_SGD_optimizer_from_pth``` [here](https://github.com/DingXiaoH/RepOptimizers/blob/main/repoptimizer/repoptvgg_impl.py).

## Another example: RepOpt-GhostNet

RepGhostNet is a recently proposed lightweight model built with Structural Re-parameterization. The training-time forward function of a block can be formulated as ```output=batch_norm(depthwise_convolution(x)) + batch_norm(x)```. With RepOptimizer, the parallel batch norm (referred to as "fusion layer" in the RepGhostNet paper) can be removed even during training. Similar to RepVGG and RepOpt-VGG, we design the CSLA model by replacing the batch norm layers with constant or trainable scaling layers and the Grad Mults of RepOptimizer accordingly. 


| name | ImageNet-1K acc | download |
|:---:|:---:|:---:|
|RepGhostNet-0.5x (our implementation)|  66.51  | [Google Drive](https://drive.google.com/file/d/1Ok5Wy1rGtg6havxtVDMlLVorabq1Q58r/view?usp=sharing), [Baidu Cloud](https://pan.baidu.com/s/1iDjmaXudw7fSXalwetOLSg?pwd=rvgg) |
|RepOpt-GhostNet-0.5x |  66.50  | [Google Drive](https://drive.google.com/file/d/1lwzG1zHXqNS5qA-35N0M-8sAE4teTKdr/view?usp=sharing), [Baidu Cloud](https://pan.baidu.com/s/1rU6tJyuMpPY8v2iZz3gSrw?pwd=rvgg) |


We trained the original RepGhostNet-0.5x with this codebase and got a top-1 accuracy of 66.51%.
```
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12349 main_repopt.py --data-path /path/to/imagenet --arch ghost-rep --batch-size 128 --tag reproduce --opts TRAIN.EPOCHS 300 TRAIN.BASE_LR 0.6 TRAIN.WEIGHT_DECAY 1e-5 TRAIN.WARMUP_EPOCHS 5 MODEL.LABEL_SMOOTHING 0.1 DATA.DATASET imagenet TRAIN.OPTIMIZER.NAME sgd TRAIN.WARMUP_LR 1e-4
```
The log and weights will be saved to ```output/ghost-rep/reproduce/```

You may reproduce RepOpt-GhostNet with our released scales
```
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12349 main_repopt.py --data-path /path/to/imagenet --arch ghost-target --batch-size 128 --tag reproduce --scales-path RepOptGhostNet_0_5x_scales.pth --opts TRAIN.EPOCHS 300 TRAIN.BASE_LR 0.6 TRAIN.WEIGHT_DECAY 1e-5 TRAIN.WARMUP_EPOCHS 5 MODEL.LABEL_SMOOTHING 0.1 DATA.DATASET imagenet TRAIN.OPTIMIZER.NAME sgd TRAIN.WARMUP_LR 1e-4
```

Or first Hyper-Search and then use the searched scales
```
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12349 main_repopt.py --data-path /path/to/cifar100 --arch ghost-hs --batch-size 128 --tag reproduce --opts TRAIN.EPOCHS 600 TRAIN.BASE_LR 0.6 TRAIN.WEIGHT_DECAY 1e-5 TRAIN.WARMUP_EPOCHS 10 MODEL.LABEL_SMOOTHING 0.1 DATA.DATASET cf100 TRAIN.CLIP_GRAD 5.0

python -m torch.distributed.launch --nproc_per_node 8 --master_port 12349 main_repopt.py --data-path /path/to/imagenet --arch ghost-target --batch-size 128 --tag reproduce --scales-path output/ghost-hs/reproduce/latest.pth --opts TRAIN.EPOCHS 300 TRAIN.BASE_LR 0.6 TRAIN.WEIGHT_DECAY 1e-5 TRAIN.WARMUP_EPOCHS 5 MODEL.LABEL_SMOOTHING 0.1 DATA.DATASET imagenet TRAIN.OPTIMIZER.NAME sgd TRAIN.WARMUP_LR 1e-4
```



## License
This project is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.

## Contact

**xiaohding@gmail.com** (The original Tsinghua mailbox dxh17@mails.tsinghua.edu.cn will expire in several months)

Google Scholar Profile: https://scholar.google.com/citations?user=CIjw0KoAAAAJ&hl=en

Homepage: https://dingxiaohan.xyz/

My open-sourced papers and repos: 

The **Structural Re-parameterization Universe**:

1. RepLKNet (CVPR 2022) **Powerful efficient architecture with very large kernels (31x31) and guidelines for using large kernels in model CNNs**\
[Scaling Up Your Kernels to 31x31: Revisiting Large Kernel Design in CNNs](https://arxiv.org/abs/2203.06717)\
[code](https://github.com/DingXiaoH/RepLKNet-pytorch).

2. **RepOptimizer** uses **Gradient Re-parameterization** to train powerful models efficiently. The training-time model is as simple as the inference-time. It also addresses the problem of quantization.\
[Re-parameterizing Your Optimizers rather than Architectures](https://arxiv.org/pdf/2205.15242.pdf)\
[code](https://github.com/DingXiaoH/RepOptimizers).

3. RepVGG (CVPR 2021) **A super simple and powerful VGG-style ConvNet architecture**. Up to **84.16%** ImageNet top-1 accuracy!\
[RepVGG: Making VGG-style ConvNets Great Again](https://arxiv.org/abs/2101.03697)\
[code](https://github.com/DingXiaoH/RepVGG).

4. RepMLP (CVPR 2022) **MLP-style building block and Architecture**\
[RepMLPNet: Hierarchical Vision MLP with Re-parameterized Locality](https://arxiv.org/abs/2112.11081)\
[code](https://github.com/DingXiaoH/RepMLP).

5. ResRep (ICCV 2021) **State-of-the-art** channel pruning (Res50, 55\% FLOPs reduction, 76.15\% acc)\
[ResRep: Lossless CNN Pruning via Decoupling Remembering and Forgetting](https://openaccess.thecvf.com/content/ICCV2021/papers/Ding_ResRep_Lossless_CNN_Pruning_via_Decoupling_Remembering_and_Forgetting_ICCV_2021_paper.pdf)\
[code](https://github.com/DingXiaoH/ResRep).

6. ACB (ICCV 2019) is a CNN component without any inference-time costs. The first work of our Structural Re-parameterization Universe.\
[ACNet: Strengthening the Kernel Skeletons for Powerful CNN via Asymmetric Convolution Blocks](http://openaccess.thecvf.com/content_ICCV_2019/papers/Ding_ACNet_Strengthening_the_Kernel_Skeletons_for_Powerful_CNN_via_Asymmetric_ICCV_2019_paper.pdf).\
[code](https://github.com/DingXiaoH/ACNet). 

7. DBB (CVPR 2021) is a CNN component with higher performance than ACB and still no inference-time costs. Sometimes I call it ACNet v2 because "DBB" is 2 bits larger than "ACB" in ASCII (lol).\
[Diverse Branch Block: Building a Convolution as an Inception-like Unit](https://arxiv.org/abs/2103.13425)\
[code](https://github.com/DingXiaoH/DiverseBranchBlock).

**Model compression and acceleration**:

1. (CVPR 2019) Channel pruning: [Centripetal SGD for Pruning Very Deep Convolutional Networks with Complicated Structure](http://openaccess.thecvf.com/content_CVPR_2019/html/Ding_Centripetal_SGD_for_Pruning_Very_Deep_Convolutional_Networks_With_Complicated_CVPR_2019_paper.html)\
[code](https://github.com/DingXiaoH/Centripetal-SGD)

2. (ICML 2019) Channel pruning: [Approximated Oracle Filter Pruning for Destructive CNN Width Optimization](http://proceedings.mlr.press/v97/ding19a.html)\
[code](https://github.com/DingXiaoH/AOFP)

3. (NeurIPS 2019) Unstructured pruning: [Global Sparse Momentum SGD for Pruning Very Deep Neural Networks](http://papers.nips.cc/paper/8867-global-sparse-momentum-sgd-for-pruning-very-deep-neural-networks.pdf)\
[code](https://github.com/DingXiaoH/GSM-SGD)


