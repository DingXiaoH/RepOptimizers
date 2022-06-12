# RepOptimizers

This is the official repository of [Re-parameterizing Your Optimizers rather than Architectures](https://arxiv.org/abs/2205.15242).

If you find the paper or this repository helpful, please consider citing

        @article{ding2022re,
        title={Re-parameterizing Your Optimizers rather than Architectures},
        author={Ding, Xiaohan and Chen, Honghao and Zhang, Xiangyu and Huang, Kaiqi and Han, Jungong and Ding, Guiguang},
        journal={arXiv preprint arXiv:2205.15242},
        year={2022}
        }

## Catalog
- [x] Model code
- [ ] PyTorch pretrained models
- [ ] PyTorch training code

<!-- ✅ ⬜️  -->

## Pre-trained Models

Uploading.


## Evaluation


## Training

To reproduce RepOpt-VGG-B1, you may build a RepOptimizer with our released constants ```RepOpt-VGG-B1-scales.pth```
```
python3 -m torch.distributed.launch --nproc_per_node 8 --master_port 12349 main_repopt.py --data-path /path/to/imagenet --arch RepOpt-VGG-B1-target --batch-size 32 --tag experiment --scales-path RepOpt-VGG-B1-scales.pth --opts TRAIN.EPOCHS 120 TRAIN.BASE_LR 0.1 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 5 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET imagenet
```
The log and weights will be saved to ```output/RepOpt-VGG-B1-target/experiment/```

Will update with more use cases in several days.



## License
This project is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.

