# Water Level Detection for Ship Hull
Conducted for ENPM673 Spring 2022

Sri Sai Charan V
Vaishanth Ramaraj
Pranav Limbekar
Yash Kulkarni
Jerry Pittman, Jr.

-------------
## RAFT
This repository contains the source code and paper for RAFT:

[RAFT: Recurrent All Pairs Field Transforms for Optical Flow](https://arxiv.org/pdf/2003.12039.pdf)<br/>
ECCV 2020 <br/>
Zachary Teed and Jia Deng<br/>

## Requirements
The code has been tested with PyTorch 1.6 and Cuda 10.1 as well as with CPU computer.
```Shell
conda create --name raft
conda activate raft
conda install pytorch=1.6.0 torchvision=0.7.0 cudatoolkit=10.1 matplotlib tensorboard scipy opencv -c pytorch
```
## Folder Structure:
```
📦Draft-Marker-Reading-Water-Level-Detection
 ┣ 📂Results
 ┃ ┣ 📜Output1.png
 ┃ ┣ 📜Output_screenshot_03.05.2022.png
 ┃ ┣ 📜RAFT.png
 ┃ ┣ 📜Water_Level_Detection_Result_Frame.png
 ┃ ┣ 📜gray1.png
 ┃ ┗ 📜gray_screenshot_03.05.2022.png
 ┣ 📂alt_cuda_corr
 ┃ ┣ 📜correlation.cpp
 ┃ ┣ 📜correlation_kernel.cu
 ┃ ┗ 📜setup.py
 ┣ 📂core
 ┃ ┣ 📂utils
 ┃ ┃ ┣ 📜__init__.py
 ┃ ┃ ┣ 📜augmentor.py
 ┃ ┃ ┣ 📜flow_viz.py
 ┃ ┃ ┣ 📜frame_utils.py
 ┃ ┃ ┗ 📜utils.py
 ┃ ┣ 📜__init__.py
 ┃ ┣ 📜corr.py
 ┃ ┣ 📜datasets.py
 ┃ ┣ 📜extractor.py
 ┃ ┣ 📜raft.py
 ┃ ┗ 📜update.py
 ┣ 📂input
 ┃ ┗ 📜water_level.mp4
 ┣ 📂models
 ┃ ┗ 📜raft-things.pth
 ┣ 📜ENPM673 Project 4 presentation (Group13).pdf
 ┣ 📜README.md
 ┣ 📜cnn_training.py
 ┣ 📜main.py
 ┗ 📜waterdetection.py
 ```
## To Run our Code on the Vessel Movie to detect and mark water levels
If have GPU, run code as is:
```Shell
python main.py
```
if don't have a GPU:
1) Comment out line 20
2) Uncomment line 23
3) Run Code:
```Shell
python main.py
```

CNN Training:
```Shell
python cnn_training.py
```

Final Output video GDrive: https://drive.google.com/drive/folders/1S9JEo39-vkEtpP1TtJBmfUNx4ZZ1hCPv?usp=sharing

## Hardware Implementation using Raspberry Pi

```Shell
python3 waterdetection.py
```

Sadly, for the the hardware implemention that due to slow processing onboard the robot with raspberry pi the number detected in the video doesn’t change. Level detection with water especially with flow takes a great deal amount of processing.

