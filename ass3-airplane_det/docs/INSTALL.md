## Installation

### Requirements

- Linux
- Python 3.5+ (Python 2 is not supported)
- PyTorch **1.3** or higher
- CUDA 9.0 or higher
- NCCL 2
- GCC(G++) **4.9** or higher
- [mmcv](https://github.com/open-mmlab/mmcv)==**0.2.14**

Note some cuda extensions, e.g., ```box_iou_rotated``` and ```nms_rotated``` require pytorch>=1.3 and gcc>=4.9.

We have tested the following versions of OS and softwares:

- OS:  CentOS 7.2
- CUDA: 10.0-10.1
- NCCL: 2.1.15/2.2.13/2.3.7/2.4.2
- GCC(G++): 4.9
- pytorch: 1.3.1

### Install

a. Create a conda virtual environment and activate it.

```shell
conda create -n AirplaneDet python=3.7 -y
conda activate AirplaneDet
```

b. Install PyTorch stable or nightly and torchvision following the [official instructions](https://pytorch.org/), e.g.,

```shell
conda install pytorch=1.3 torchvision cudatoolkit=10.0 -c pytorch
```

c. Install AirplaneDet

```shell
# optional
pip install -r requirements.txt

python setup.py develop
# or "pip install -v -e ."
```

### Install DOTA_devkit
```
sudo apt-get install swig
cd DOTA_devkit/polyiou
swig -c++ -python csrc/polyiou.i
python setup.py build_ext --inplace
```

### Prepare datasets

You need convert annotations to mmdet's format. Please refer to [DOTA_devkit/prepare_fair1m.py](../DOTA_devkit/prepare_fair1m.py).

It is recommended to symlink the dataset root to `$MMDETECTION/data`.
If your folder structure is different, you may need to change the corresponding paths in config files.

```
mmdetection
├── mmdet
├── tools
├── configs
├── data
│   ├── dota_1024
│   │   ├── trainval_split
│   │   │    │─── images
│   │   │    │─── labelTxt
│   │   │    │─── trainval_s2anet.pkl
│   │   ├── test_split
│   │   │    │─── images
│   │   │    │─── test_s2anet.pkl
│   ├── HRSC2016 (optional)
│   │   ├── Train
│   │   │    │─── AllImages
│   │   │    │─── Annotations
│   │   │    │─── train.txt
│   │   ├── Test
│   │   │    │─── AllImages
│   │   │    │─── Annotations
│   │   │    │─── test.txt
```

Note `train.txt` and `test.txt` are `.txt` files recording image names without extension.

For example:
```
1033__1__0___0
1076__1__0___0
1113__1__0___0
1116__1__0___0
1120__1__0___0
1125__1__0___0
1131__1__0___0
1133__1__0___0
...
```
