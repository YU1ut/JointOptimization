# Joint Optimization Framework for Learning with Noisy Labels
This is an unofficial PyTorch implementation of [Joint Optimization Framework for Learning with Noisy Labels](https://arxiv.org/abs/1803.11364). 
The official Chainer implementation is [here](https://github.com/DaikiTanaka-UT/JointOptimization).


## Requirements
- Python 3.6
- PyTorch 0.4
- torchvision
- progress
- matplotlib
- numpy

## Usage
Train the network on the Symmmetric Noise CIFAR-10 dataset (noise rate = 0.7):

First, 
```
python train.py --gpu 0 --out first_sn07 --lr 0.08 --alpha 1.2 --beta 0.8 --percent 0.7
```
to train and relabel the dataset.

Secondly,
```
python retrain.py --gpu 0 --out second_sn07 --label first_sn07
```
to retrain on the relabeled dataset.

Train the network on the Asymmmetric Noise CIFAR-10 dataset (noise rate = 0.4):

First, 
```
python train.py --gpu 0 --out first_an04 --lr 0.03 --alpha 0.8 --beta 0.4 --percent 0.4 --asym
```
to train and relabel the dataset.

Secondly,
```
python retrain.py --gpu 0 --out second_an04 --label first_an04
```
to retrain on the relabeled dataset.


## References
- D. Tanaka, D. Ikami, T. Yamasaki and K. Aizawa. "Joint Optimization Framework for Learning with Noisy Labels", in CVPR, 2018.