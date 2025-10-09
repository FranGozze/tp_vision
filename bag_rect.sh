#!/bin/bash

ros2 bag play V1_01_easy --remap /cam0/image_raw:=/stereo/left/image_raw \
  /cam1/image_raw:=/stereo/right/image_raw \
  # /cam0/camera_info:=/stereo/left/camera_info \
  # /cam1/camera_info:=/stereo/right/camera_info
