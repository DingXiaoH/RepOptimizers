# RepOptimizers

(Sep 23rd: Refactoring. Will finish in two days. Will release models in one weak.)

This is the official repository of [Re-parameterizing Your Optimizers rather than Architectures](https://arxiv.org/abs/2205.15242).

If you find the paper or this repository helpful, please consider citing

        @article{ding2022re,
        title={Re-parameterizing Your Optimizers rather than Architectures},
        author={Ding, Xiaohan and Chen, Honghao and Zhang, Xiangyu and Huang, Kaiqi and Han, Jungong and Ding, Guiguang},
        journal={arXiv preprint arXiv:2205.15242},
        year={2022}
        }


## Highlights

RepOptimizer and RepOpt-VGG have been used in **YOLOv6** ([paper](https://arxiv.org/abs/2209.02976), [code](https://github.com/meituan/YOLOv6)) and **deployed in business**. The methodology of Structural Re-parameterization also plays a critical role in **YOLOv7** ([paper](https://arxiv.org/abs/2207.02696), [code](https://github.com/WongKinYiu/yolov7)).

## Catalog
- [x] Model code
- [ ] PyTorch pretrained models
- [ ] PyTorch training code

<!-- ✅ ⬜️  -->

## Pre-trained Models

Uploading.


## Evaluation


## Training

To reproduce RepOpt-VGG-B1, you may build a RepOptimizer with our released constants ```RepOpt-VGG-B1-scales.pth``` and train a RepOpt-VGG-B1 with it.
```
python3 -m torch.distributed.launch --nproc_per_node 8 --master_port 12349 main_repopt.py --data-path /path/to/imagenet --arch RepOpt-VGG-B1-target --batch-size 32 --tag experiment --scales-path RepOpt-VGG-B1-scales.pth --opts TRAIN.EPOCHS 120 TRAIN.BASE_LR 0.1 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 5 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET imagenet
```
The log and weights will be saved to ```output/RepOpt-VGG-B1-target/experiment/```

Will update with more use cases in several days.



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


