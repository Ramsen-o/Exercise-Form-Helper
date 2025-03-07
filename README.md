# Bicep Curl Counter

## Overview
This Python program uses MediaPipe to track **33 body keypoints** and counts the number of **bicep curl repetitions** only if the form is proper. The validation is based on the angle of the bicep, ensuring correct movement.

## Features
- **Tracks 33 body keypoints** using MediaPipe's pose estimation.
- **Counts bicep curl reps** based on precise form validation.
- **Vector-based angle calculation** ensures accuracy.
- **Thresholds for proper form:**
  - **Top position:** Bicep angle must be **less than 30 degrees**.
  - **Bottom position:** Bicep angle must be **greater than 160 degrees**.

## Requirements
- Python 3.x
- MediaPipe
- OpenCV
- NumPy

### Install Dependencies
Run the following command to install the required dependencies:
```sh
pip install mediapipe opencv-python numpy
```

## How It Works
1. **Capture Video Input:** The program reads from a webcam.
2. **Pose Detection:** Uses MediaPipe's pose estimation model to extract **33 body keypoints**.
3. **Angle Calculation:** Computes the elbow angle using vector mathematics.
4. **Repetition Counting:** A rep is counted only when the arm extends fully (>160°) and then contracts properly (<30°).
5. **Real-time Display:** The current rep count is shown on the screen.

## Usage
Run the script using:
```sh
python bicep_curl_counter.py
```
Ensure your **arm movement follows the required form** for reps to be counted.

## Vector Calculation for Angle Measurement
The elbow angle is determined using vector calculations:
- **Vectors are created** using shoulder, elbow, and wrist coordinates.
- **Dot product and arctan function** are used to calculate the angle between vectors.

## Future Improvements
- Add support for other exercises.
- Improve form via comparison to pre-selected exercise poses.

## Author
Developed by [Ramsen Oraha]

