# FoundationPose-ROS
This repository provides the code and Docker containers for **FoundationPose**, **segmentation model**, and a **ROS bridge**.

I tested on my computer that has ubuntu 20.04, ROS Noetic, realsense camera (d435i) in host.

The setup has been tested on a system running Ubuntu 20.04 with ROS Noetic and a RealSense D435i camera.

Below are the official implementations of the pose estimation and segmentation models used:
- **FoundationPose**: [[GitHub]](https://github.com/NVlabs/FoundationPose) | [[Paper]](https://arxiv.org/abs/2312.08344)
- **Grounded SAM**: [[GitHub]](https://github.com/IDEA-Research/Grounded-Segment-Anything) | [[Paper]](https://arxiv.org/abs/2401.14159)

This project was inspired by [this repository](https://github.com/shubho-upenn/FoundationPose/tree/ROS_FoundationPose), which integrates both models with ROS.

https://github.com/user-attachments/assets/8df3949e-8bcf-43f5-a007-1de1bcbcf3fb

# License
The code and data are released under the NVIDIA Source Code License. Copyright © 2024, NVIDIA Corporation. All rights reserved.

# ROS (Ubuntu 20.04, ROS Noetic)

   
> **🔴 Important Note** \
> **Grounded SAM starts running immediately upon launch.**  
> Ensure that **FoundationPose** is already running before starting **Grounded SAM**.

## RGB Segmentation: GSA + ROS (Docker Container)
The Docker image `soom1017/ros_gsa:latest` is available and mirrors `ghcr.io/shubho-upenn/gsa_ros:latest`.

To use it:
1. Pull the Docker image.
2. Download the model weights.
3. Run the scripts below.
   ```sh
   $ cd Grounded-Segment-Anything
   $ ./run_gsa_ros.sh
   $ cd Grounded-Segment-Anything && python ros_gsam.py
   ```

All other steps are the same as in the original repository.

## Pose Estimation: FoundationPose (in Docker container)
The Docker image `soom1017/ros_fp_light_new:fp_only` is available for this component.

To use it:
1. Pull the Docker image.
2. Download the model weights.
3. Run the scripts below.
   ```sh
   $ cd docker
   $ ./run_custom.sh
   $ python run_ROS.py
   ```
