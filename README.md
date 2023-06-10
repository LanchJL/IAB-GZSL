# Estimation of Near-Instance-Level Attribute Bottleneck for Zero-Shot Learning (IAB)
Codes of Estimation of Near-Instance-Level Attribute Bottleneck for Zero-Shot Learning

## Installation
```shell
$ cd repository
$ pip install -r requirements.txt
```

## Datasets
The splits of dataset and its attributes can be found in [data]([https://github.com/wenjiaXu/APN-ZSL](https://drive.google.com/file/d/1bCZ28zJZNzsRjlHxH_vh2-9d7Ln1GgjE/view)). Please download the data folder and place it in ./data/.\\

Set the --root in opt.py as your code path.

Please download CUB, AWA2, SUN, FLO datasets, and set the --image_root in opt.py to the datasets.\\

Please download pretrained resnet weights and place it in ./pretrained_models/

## Train
Please run IAB.py, for example:
```shell
python IAB.py --att_size 85 --image_size 224 --t 8 --Lp1 10 --gamma 2 --delta 2.0 --calibrated_stacking 2.0 --seen_classes 40 --nclasses 50
```

##Acknowledgment

We are very grateful to the following repos for their great help in constructing our work:
[1] [APN] https://github.com/wenjiaXu/APN-ZSL



