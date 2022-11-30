# Car Orientation Detection

[English](https://github.com/Fan-Treasure/Auto-Parking/blob/Car-Orientation-Detection/README_en.md) | [简体中文](https://github.com/Fan-Treasure/Auto-Parking/blob/Car-Orientation-Detection/README_en.md)

## Introduction

Shandong University (Weihai) 2020 Data Science and Artificial Intelligence Experimental Class Computer Vision (1) Big Assignment Part I: Car Recognition and Orientation Detection.

We use PP-YOLOv2 for car recognition and orientation detection, and our own network for orientation detection, both of which are trained in Baidu AI Studio and deployed locally.

## Reproduction

### Car Detection

We trained the car recognition model based on PP-YOLOv2 in Baidu AI Studio, as detailed in [Car Detection Model Training  based on PP-YOLOv2 in Baidu AI Studio](https://aistudio.baidu.com/aistudio/projectdetail/4918206).

### Orientation Detection

We trained the orientation detection model in Baidu AI Studio based on the network we built, as detailed in [Orientation Detection Model Training in Baidu AI Studio](https://aistudio.baidu.com/aistudio/projectdetail/4885428).

## Local Deployment

### Environments

- Python＞=3.7
- paddlepaddle-gpu==2.3.2
- PaddleDetection==2.5
- CUDA >= 10.2
- cuDNN >= 7.6
- opencv-python==3.4

For general information about how to install PaddleDetection, please see [PaddleDetection Installation](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.5/docs/tutorials/INSTALL.md).

### Install Dependencies

- Clone this project

```
git clone https://github.com/Xujialu/Auto-Parking/Car-Orientation-Detection
```

- Go to the project directory

```
cd Car-Orientation-Detection
```

- Install environments

```
pip install -r requirements.txt
```

### Download Models

You can download the models provided in AI Studio projects, or you can fork the project in AI Studio and train your own models.

### Run Models

```
python Car_Orientation_Dtection.py
```

## Results

<center>
 <img src=".\images\Orientation Detection.png">
</center>

<center>
 <img src=".\images\Orientation Detection and Automatic Driving.png">
</center>


## Other Links

[[Baidu AI Studio]Car Detection Model Training  based on PP-YOLOv2](https://aistudio.baidu.com/aistudio/projectdetail/4918206) 

[[Baidu AI Studio]Orientation Detection Model Training](https://aistudio.baidu.com/aistudio/projectdetail/4885428) 

[[Github]Road Segmentation and Autonomous Driving](https://github.com/xujialuu/self-driving-car)

[[Bilibili]Explanation Video of Car Orientation Detection](https://www.bilibili.com/video/BV1YM41167Dy) 
