## PSCFNet: Prototype Learning and Spatial Consistent Feature Fusion for Efficient Salient Object Detection

This is the code for the IJCNN 2024 Submission "PSCFNet: Prototype Learning and Spatial Consistent Feature Fusion for Efficient Salient Object Detection".


## Requirements

- Python 3.8
- PyTorch 1.10.0
- cuda 11.4


## Download dataset

Download the following datasets and unzip them into `GT` folder

- [PASCAL-S](http://cbi.gatech.edu/salobj/)
- [ECSSD](http://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html)
- [HKU-IS](https://i.cs.hku.hk/~gbli/deep_saliency.html)
- [DUT-OMRON](http://saliencydetection.net/dut-omron/)
- [DUTS](http://saliencydetection.net/duts/)

## Prepare backbones
[PVT-V2-B0]([https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_a1h-35c100f8.pth](https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b0.pth))
Create a directory 'pretrain_models' to place the above backbone.

## Training and Testing
You can train the model as followed.

```shell
    python train.py
```

For testing, you have to prepare a pretrained model. You can train one by yourself. We will upload the pretrained model if the paper is accepted.
```shell
    python test.py
```
