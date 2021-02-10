## SSS-AE: Anomaly Detection using Self-Attention based Sequence-to-Sequence Auto-Encoder and Multi-Products Learning in SMD Assembly Machine Sound

[SSS-AE: Anomaly Detection using Self-Attention based Sequence-to-Sequence Auto-Encoder and Multi-Products Learning in SMD Assembly Machine Sound]

[Kihyun Nam](https://github.com/DevKiHyun)<sup>1*</sup>, YoungJong Song<sup>1*</sup>, IlDong Yun<sup>1,2</sup> 

A Surface-Mount Device assembly machine continuously assembles various products in the real field. Unwanted situations such as device failure can occur at any time during the assembly process. Detecting these situations may be regarded as an anomaly detection problem. Existing SMD assembly machine sound anomaly detection models show good performance for a small dataset, but are unstable on a large dataset with various products. Beside, the number of models to be trained increases linearly as new products increase day by day. In this paper, we propose a Self-Attention based Sequence-to-Sequence Auto-Encoder (SSS-AE) with Temporal-Adaptive-Average-Pooling to solve these problems. The proposed model is a powerful model that uses self-attention so that Encoder attend to relationships between inner states of input sequential feature and applies sequence-to-sequence structure to pass temporal information well from Encoder to Decoder. In addition, we introduce an effective preprocessing method, Temporal-Adaptive-Average Pooling, to significantly improve the performance of anomaly detection models including previous model. For more accurate evaluation compared with the previous SMD anomaly detection studies, a new large-scale SMD dataset containing observed real abnormal products were collected and evaluated. Experimental results show that SSS-AE is a robust and practical model for both single-product learning and multi-product learning.

<sup>1</sup> Department of Computer Engineering, Hankuk University on Foreign Studies <p>
<sup>2</sup> Corresponding Author <p>
<sup>*</sup> Both authors equally contributed to this work.
  
  
## Table of contents 
* [1. Dataset](#1-dataset)
    + [SMD2020](#smd2020)
    + [Setting](#setting)
* [2. Dependency](#2-dependency)
* [3. Training and Evaluation](#3-training-and-evaluation)
    + [Run train](#run-train)
    + [Run test](#run-test)

## 1. Dataset

### SMD2020
```
To download our SMD dataset(2020 version) used in this paper, please contact the corresponding author by email.
Please keep in mind that it can only be used for academic purposes and not for commercial purposes.

Corresponding Auth - IlDong Yun(yun@hufs.ac.kr) 
```

#### Audio examples

| Name | Normal 1 | Normal 2 | Error level 1 | Error level 2|
| :---: | :-----: | :------: | :------------: | :-----------: |
| GT-4118 | [wav](https://github.com/HUFS-VLab/Tensorflow-SSAE/blob/master/dataset/SMD_2020/GT-4118/001.wav) | [wav](https://github.com/HUFS-VLab/Tensorflow-SSAE/blob/master/dataset/SMD_2020/GT-4118/002.wav) | [wav](https://github.com/HUFS-VLab/Tensorflow-SSAE/blob/master/dataset/SMD_2020/GT-4118-1/001.wav) | [wav](https://github.com/HUFS-VLab/Tensorflow-SSAE/blob/master/dataset/SMD_2020/GT-4118-2/001.wav) |
| ST-3214 | [wav](https://github.com/HUFS-VLab/Tensorflow-SSAE/blob/master/dataset/SMD_2020/ST-3214/001.wav) | [wav](https://github.com/HUFS-VLab/Tensorflow-SSAE/blob/master/dataset/SMD_2020/ST-3214/002.wav) | [wav](https://github.com/HUFS-VLab/Tensorflow-SSAE/blob/master/dataset/SMD_2020/ST-3214-1/001.wav) | [wav](https://github.com/HUFS-VLab/Tensorflow-SSAE/blob/master/dataset/SMD_2020/ST-3214-2/001.wav) |
| ST-3708 | [wav](https://github.com/HUFS-VLab/Tensorflow-SSAE/blob/master/dataset/SMD_2020/ST-3708/001.wav) | [wav](https://github.com/HUFS-VLab/Tensorflow-SSAE/blob/master/dataset/SMD_2020/ST-3708/002.wav) | [wav](https://github.com/HUFS-VLab/Tensorflow-SSAE/blob/master/dataset/SMD_2020/ST-3708-1/001.wav) | [wav](https://github.com/HUFS-VLab/Tensorflow-SSAE/blob/master/dataset/SMD_2020/ST-3708-2/001.wav) |


### Setting
We should follow the structure of the directory and manifests of the dataset as below:

```
Tensorflow-SSAE/
└──run.py
...
└──dataset/ # Important!!
   └──SMD_2020/
      └──GT-4118/
         └──001.wav
         ...
         └──00N.wav
      └──ST-3214/
         └──001.wav
         ...
         └──00N.wav
   └──other_dataset/ # Just Example 
      └──class1/
         └──sample1.wav
         ...
         └──sample2.wav
└──manifets/ # Manifests of target data. Also Important!!
   └──GT-4118.json
   └──GT-4118-1.json
   └──GT-4118-2.json
   └──ST-3214.json
   └──ST-3214-1.json
   └──ST-3214-2.json
   └──ST-3708.json
   └──ST-3708-1.json
   └──ST-3708-2.json
```

We should be make manfiests of the target data(to train and test) into `manifests/` (e.g. GT-4118, ST-3214)
```
GT-4118.json
[
    {
        "wav": "SMD_2020/GT-4118/001",
        "sr": 192000,
        "item": "GT-4118",
        "type": "NORMAL"
    },
    {
        "wav": "SMD_2020/GT-4118/002",
        "sr": 192000,
        "item": "GT-4118",
        "type": "NORMAL"
    }
]
---------------------
ST-3214.json
[
    {
        "wav": "SMD_2020/ST-3214/001",
        "sr": 192000,
        "item": "ST-3214",
        "type": "NORMAL"
    },
    {
        "wav": "SMD_2020/ST-3214/002",
        "sr": 192000,
        "item": "ST-3214",
        "type": "NORMAL"
    }
]
```

## 2. Dependency
```
numpy==1.16.4
matplotlib==3.2.1
librosa==0.7.0
scipy==1.3.1
tensorflow==1.14 (Available= 1.10 <= x <=1.14)
```

## 3. Training and Evaluation

### Run train

```
cd script/

./run_ssae_trainer.sh
```

### Run test
```
cd script/

./run_ssae_test.sh
```
