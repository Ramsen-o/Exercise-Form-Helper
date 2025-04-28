# Fitness Virtual Assistant

## Overview
This Python program uses MoveNet to track **17 body keypoints** and counts the number of **bicep curl reps & squat reps** only if the form is proper. The validation is based on the angle of the biceps/legs, ensuring correct movement.

## Features
- **Tracks 17 body keypoints** using MediaPipe's pose estimation.
- **Example of Thresholds for proper form:**
  - **Top position:** Bicep angle must be **less than 30 degrees**.
  - **Bottom position:** Bicep angle must be **greater than 160 degrees**.

## Requirements
- Python 3.x
- MoveNet
- OpenCV
- NumPy
- Google.generativeai
- Speech_recognition

## Author
Developed by [Ramsen Oraha]

