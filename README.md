This is the official code for "Diffusion-Based Continuous Feature Representation for Infrared Small-Dim Target Detection"
![3-3_DCHFR-Net](https://github.com/flyannie/DCFR-Net/assets/162861421/00e27e9f-b37c-4fba-88b7-59b8e4ec9248)

# 1. Data preparation

This experiment includes multiple public datasets, which are introduced below one by one. The quadruple bicubic interpolation required in the DCHFR branch is included in the code. Additionally, since DCHFR incorporates a diffusion model, its testing process is relatively slower. To avoid affecting the training progress, you may consider extending the testing interval or reducing the number of test images.

## 1.1 "BigReal" and "BigSim"

* dataset: https://xzbai.buaa.edu.cn/datasets.html

* paper doi: 10.1109/TGRS.2023.3235150

* description: IRDST dataset consists of 142,727 real and simulation frames (40,650 real frames in 85 scenes and 102,077 simulation frames in 317 scenes).

* path format
  
  images: "your root path/images/xxx.png", where "xxx" is included in the corresponding TXT file.

  labels: "your root path/masks/xxx.png", where "xxx" is included in the corresponding TXT file.

  TXT files: "your root path/train.txt", "your root path/test_hr.txt"

## 1.2 "NUAA"

* dataset: https://github.com/YimianDai/sirst

* paper doi: 10.1109/WACV48630.2021.00099

* description: SIRST is a dataset specially constructed for single-frame infrared small target detection, in which the images are selected from hundreds of infrared sequences for different scenarios.

* path format
  
  images: "your root path/train_imgs/xxx.png", "your root path/test_imgs/xxx.png", where "xxx" will be automatically retrieved in the code.

  labels: "your root path/train_labels/xxx.png", "your root path/test_labels/xxx.png", where "xxx" will be automatically retrieved in the code.

## 1.3 "NUDT"

* dataset: https://github.com/YeRen123455/Infrared-Small-Target-Detection

* paper doi: 10.1109/TIP.2022.3199107

* description: NUDT-SIRST dataset is a synthesized dataset, which contains 1327 images with resolution of 256x256.

* path format
  
  images: "your root path/train_imgs/xxx.png", "your root path/test_imgs/xxx.png", where "xxx" will be automatically retrieved in the code.

  labels: "your root path/train_labels/xxx.png", "your root path/test_labels/xxx.png", where "xxx" will be automatically retrieved in the code.

## 1.4 "IRSTD"

* dataset: https://github.com/RuiZhang97/ISNet

* paper doi: 10.1109/CVPR52688.2022.00095

* description: IRSTD-1k dataset is the realistic infrared small target detection dataset, which consists of 1,001 manually labeled realistic images with various target shapes, different target sizes, and rich clutter back-grounds from diverse scenes.

* path format
  
  images: "your root path/train_imgs/xxx.png", "your root path/test_imgs/xxx.png", where "xxx" will be automatically retrieved in the code.

  labels: "your root path/train_labels/xxx.png", "your root path/test_labels/xxx.png", where "xxx" will be automatically retrieved in the code.
  
# 2.Training DCHFR branch

If resources are insufficient to train this branch, you can directly initialize the encoder weights in "train_ISDTD.py" randomly. This approach can still achieve results close to the state-of-the-art (SOTA). Of course, with sufficient resources, more targeted results can be obtained.

# 3.Training ISDTD branch
