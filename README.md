This is the official code for "Diffusion-Based Continuous Feature Representation for Infrared Small-Dim Target Detection"
![3-3_DCHFR-Net](https://github.com/flyannie/DCFR-Net/assets/162861421/00e27e9f-b37c-4fba-88b7-59b8e4ec9248)

1. Data preparation
This experiment includes multiple public datasets, which are introduced below one by one. The quadruple bicubic interpolation required in the DCHFR branch is included in the code. Additionally, since DCHFR incorporates a diffusion model, its testing process is relatively slower. To avoid affecting the training progress, you may consider extending the testing interval or reducing the number of test images.
1.1 "BigReal" and "BigSim"
dataset:https://xzbai.buaa.edu.cn/datasets.html
paper doi: 10.1109/TGRS.2023.3235150
description: IRDST dataset consists of 142,727 real and simulation frames (40,650 real frames in 85 scenes and 102,077 simulation frames in 317 scenes). 
path format:
images: "your root path/images/xxx.png", where "xxx" is included in the corresponding TXT file.
labels: "your root path/masks/xxx.png", where "xxx" is included in the corresponding TXT file.
TXT files: "your root path/train.txt", "your root path/test_hr.txt"
1.2 "NUAA"
1.3 "NUDT"
1.4 "IRSTD"
2.Training DCHFR branch
If resources are insufficient to train this branch, you can directly initialize the encoder weights in "train_ISDTD.py" randomly. This approach can still achieve results close to the state-of-the-art (SOTA). Of course, with sufficient resources, more targeted results can be obtained.
3.Training ISDTD branch
