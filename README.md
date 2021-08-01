# RaNet: Relation-aware Network for Moment Localization via Language

## Introduction

This is an implementation repository for RaNet: Relation-aware Network for Moment Localization via Language. 

![](https://github.com/Anonymous970/RaNet/blob/master/img/framework.png)

## Note:
The repository contains the development code. This preview is intended for the reviewers of our AAAI2022 submission.
The code provided allows for evaluating our pretrained models. We will release the final version of the code on our official GitHub repo soon.
We discourage the reviewers from distributing this repository to third party users. Please follow the instructions below for the installation and download of necessary data. 

# Installation

Clone the repository and move to folder:
```bash
git clone https://github.com/Anonymous970/RaNet.git

cd RaNet
```

To use this source code, you need Python3.7+ and a few python3 packages:
- pytorch 1.1.0
- torchvision 0.3.0
- torchtext
- easydict
- terminaltables
- tqdm

# Data
We use the data offered by [2D-TAN](https://rochester.app.box.com/s/8znalh6y5e82oml2lr7to8s6ntab6mav).

</br>

The folder structure should be as follows:
```
.
├── checkpoints
│   ├── checkpoints-paper
│   │    ├── TACoS
│   │    ├── ActivityNet
│   │    └── Charades
├── data
│   ├── TACoS
│   │    ├── tall_c3d_features.hdf5
│   │    └── ...
│   ├── ActivityNet
│   │    ├── sub_activitynet_v1-3.c3d.hdf5
│   │    └── ...
│   ├── Charades-STA
│   │    ├── charades_vgg_rgb.hdf5
│   │    └── ...
│
├── experiments
│
├── lib
│   ├── core
│   ├── datasets
│   └── models
│
└── moment_localization
```

# Train and Test
Please download the visual features from [box drive](https://rochester.app.box.com/s/8znalh6y5e82oml2lr7to8s6ntab6mav) and save it to the `data/` folder.

#### Training
Use the following commands for training:
- For TACoS dataset, run: 
```bash
    sh run_tacos.sh
```
- For ActivityNet-Captions dataset, run:
```bash
    sh run_activitynet.sh
```
- For Charades-STA dataset, run:
```bash
    sh run_charades.sh
```

#### Testing
Our trained model are provided in [box drive](https://). Please download them to the `checkpoints/checkpoints-paper/` folder.
Use the following commands for testing:
- For TACoS dataset, run: 
```bash
    sh test_tacos.sh
```
- For ActivityNet-Captions dataset, run:
```bash
    sh test_activitynet.sh
```
- For Charades-STA dataset, run:
```bash
    sh test_charades.sh
```

# Main results:

| **TACoS** | Rank1@0.3 | Rank1@0.5 | Rank5@0.3 | Rank5@0.5 |
| ---- |:-------------:| :-----:|:-----:|:-----:|
| **RaNet** |  43.09 | 32.24 |  68.71 | 55.51 |
</br>

| **ActivityNet** | Rank1@0.5 | Rank1@0.7 | Rank5@0.6 | Rank5@0.7 |
| ---- |:-------------:| :-----:|:-----:|:-----:|
| **RaNet** | 46.23 | 29.37 | 76.09 | 62.33 |
</br>

| **Charades-STA**  | Rank1@0.5 | Rank1@0.7 | Rank5@0.5 | Rank5@0.7 |
| ---- |:-------------:| :-----:|:-----:|:-----:|
| **RaNet** | 44.33 | 26.59 | 87.23 | 53.47 |
