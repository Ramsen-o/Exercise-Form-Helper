'''
 * Author: [Ramsen Oraha]
 * Date: [2025-02-28]
 * Description: [Exercise form helper to track bicep curl reps which have acceptable form.]
'''

import cv2
import mediapipe as mp
import numpy as np
import math

mp_drawing = mp.solutions.drawing_utils  # grabbing drawing utilities
mp_pose = mp.solutions.pose  # grabbing pose estimation model

# VIDEO FEED
cap = cv2.VideoCapture(0)  # setting video capture device (webcam)

# curl counter variables
counter = 0
stage = None

# calculate the angle between 3 joint landmarks
def calculate_angle(a, b, c):
    a = np.array(a)  # first joint
    b = np.array(b)  # middle joint
    c = np.array(c)  # end joint

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(
        a[1] - b[1], a[0] - b[0]
    )
    angle = math.ceil(np.abs(radians * 180.0 / np.pi))

    if angle > 180.0:
        angle = 360 - angle

    return angle


## setup mediapipe instance
with mp_pose.Pose(
    min_detection_confidence=0.5, min_tracking_confidence=0.5
) as pose:  # work with 'pose'
    while cap.isOpened():
        ret, frame = cap.read()  # return a frame of the capture reading

        # recoloring each image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # make detection
        results = pose.process(image)

        # recoloring each image back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # extract landmarks, try to read landmarks if visible, if not, pass through
        try:
            landmarks = results.pose_landmarks.landmark

            # get coordinates of joint landmarks
            shoulder = [
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
            ]
            elbow = [
                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
            ]
            wrist = [
                landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y,
            ]

            # calculate angle
            angle = calculate_angle(shoulder, elbow, wrist)

            # visualize angle, puts text to string, cv2 expects tuple, multiply coords by image resolution
            cv2.putText(
                image,
                str(angle),
                tuple(np.multiply(elbow, [640, 480]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # curl counter logic
            if angle > 160:
                stage = "down"
            if angle < 30 and stage == "down":
                stage = "up"
                counter = counter + 1
                print(counter)
            
        except:
            pass

        # render curl counter
        # setup status box
        cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
        # add rep counter to the box
        cv2.putText(image, 'REPS', (15,12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter), (10,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        # add stage up/down to the box
        cv2.putText(image, 'STAGE', (65,12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, stage, (60,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        
        # render detections / mp1 = landmark, mp2 = lines
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
        )

        cv2.imshow(
            "Bicep Curl Counter", image
        )  # displays window named MediaPipe Feed, displays us captured frames

        if cv2.waitKey(10) & 0xFF == ord(
            "q"
        ):  # wait for 10ms, 0xFF reads keyboard input, if q is pressed close all windows.
            break

cap.release()
cv2.destroyAllWindows()



shoulder = [
    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
]
elbow = [
    landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
    landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
]
wrist = [
    landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
    landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y,
]

