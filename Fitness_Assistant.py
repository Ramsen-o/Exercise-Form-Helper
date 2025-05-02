import cv2 #
import numpy as np
import math
import threading
import time
from PIL import Image #
from pycoral.adapters import common #
from pycoral.utils.edgetpu import make_interpreter #
from sense_hat import SenseHat
import speech_recognition as sr
import sounddevice
import google.generativeai as genai

# setting API key for gemini
genai.configure(api_key = 'API_KEY')

# select the specific AI model to use
gemini_model = genai.GenerativeModel('gemini-1.5-pro')

# initialize the recognizer
recognizer = sr.Recognizer()

# COLOR CONSTANTS
BLUE = [0, 0, 255]	# squat color
RED = [255, 0, 0]	# bicep curl color
GREEN = [0, 255, 0]
WHITE = [255, 255, 255]	# reset counter color
ORANGE = [255, 120, 0]	# temperature color
BROWN = [150, 75, 0]

# load MoveNet model for Edge TPU
model_path = "movenet_single_pose_lightning_ptq_edgetpu.tflite"
interpreter = make_interpreter(model_path)
interpreter.allocate_tensors()

# initialize sensehat, thread locking variable
sense = SenseHat()
display_lock = threading.Lock()

# video feed
cap = cv2.VideoCapture(0)

# counter variables
counter = 0
stage_legs = None
stage_left_arm = None
stage_right_arm = None
exercise_mode = "biceps"	# default to counting bicep curls

# Define colors
ON = (255, 0, 0)	# red
OFF = (0, 0, 0)	# black

# 7-segment layout defining the pixel coords for A-G
segment_coords = {
    'A': [(0, 0), (1, 0), (2, 0)],
    'B': [(2, 0), (2, 1), (2, 2)],
    'C': [(2, 2), (2, 3), (2, 4)],
    'D': [(0, 4), (1, 4), (2, 4)],
    'E': [(0, 2), (0, 3), (0, 4)],
    'F': [(0, 0), (0, 1), (0, 2)],
    'G': [(0, 2), (1, 2), (2, 2)],
}

# mapping segments for each digit
digit_segments = {
    0: ['A', 'B', 'C', 'D', 'E', 'F'],
    1: ['B', 'C'],
    2: ['A', 'B', 'G', 'E', 'D'],
    3: ['A', 'B', 'G', 'C', 'D'],
    4: ['F', 'G', 'B', 'C'],
    5: ['A', 'F', 'G', 'C', 'D'],
    6: ['A', 'F', 'G', 'E', 'C', 'D'],
    7: ['A', 'B', 'C'],
    8: ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
    9: ['A', 'B', 'C', 'D', 'F', 'G'],
}

# Positions of left and right digits
digit_positions = [(0, 1), (5, 1)]

def draw_digit(digit, origin_x, origin_y, color=ON):
    for segment in digit_segments[digit]:
        for dx, dy in segment_coords[segment]:
            x = origin_x + dx
            y = origin_y + dy
            if 0 <= x < 8 and 0 <= y < 8:
                sense.set_pixel(x, y, color)

def update_led_display(num, color=ON):
    if num < 0 or num > 99:
        return
    sense.clear()
    left_digit = num // 10
    right_digit = num % 10
    draw_digit(left_digit, *digit_positions[0], color=color)
    draw_digit(right_digit, *digit_positions[1], color=color)

# function to calculate angle
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    # getting length of each side of "triangle"
    ab = np.linalg.norm(a - b)
    bc = np.linalg.norm(b - c)
    ac = np.linalg.norm(a - c)
    
    if bc * ab == 0: # avoid division by zero
        return 0
    
    cosine_angle = (ab**2 + bc**2 - ac**2) / (2 * ab * bc) # getting angle using cosine rule
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0) # clipping off ranges out of bounds to avoid bugs
    
    angle = math.degrees(math.acos(cosine_angle)) # convert result to degrees
    
    return round(angle)


# function to listen for joystick button press and directional movement
def joystick_listener():
    global counter, exercise_mode   # global counter variables
    while True:
        for event in sense.stick.get_events(): # when there is a joystick interaction...
            if event.direction in ["up", "down", "left", "right"]:
                time.sleep(0.5)
                if exercise_mode == "biceps": # if moved directionally change exercises depending on the current exercise
                    exercise_mode = "squats"
                    counter = 0
                    print("Switched to squats.")
                elif exercise_mode == "squats":
                    exercise_mode = "biceps"
                    counter = 0
                    print("Switched to bicep curls.")
            elif event.action == "pressed" and event.direction == "middle":   # if pressed straight down reset the counter and display zero
                counter = 0
                update_led_display(counter, WHITE)
                print("Counter Reset.")

                

# run joystick listener in a separate thread to avoid blocking main loop
threading.Thread(target=joystick_listener, daemon=True).start()

def bicep_counter(frame, keypoints):
    global counter, stage_left_arm, stage_right_arm
        # extract landmarks
    try:
        left_shoulder = keypoints[5][:2] 
        left_elbow = keypoints[7][:2]  
        left_wrist = keypoints[9][:2] 
        right_shoulder = keypoints[6][:2]
        right_elbow = keypoints[8][:2]
        right_wrist = keypoints[10][:2]
        
        left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_arm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
        
        frame_height, frame_width, _ = frame.shape
        
        # scale keypoints to frame size
        left_shoulder = (int(left_shoulder[1] * frame_width), int(left_shoulder[0] * frame_height))
        left_elbow = (int(left_elbow[1] * frame_width), int(left_elbow[0] * frame_height))
        left_wrist = (int(left_wrist[1] * frame_width), int(left_wrist[0] * frame_height))
        right_shoulder = (int(right_shoulder[1] * frame_width), int(right_shoulder[0] * frame_height))
        right_elbow = (int(right_elbow[1] * frame_width), int(right_elbow[0] * frame_height))
        right_wrist = (int(right_wrist[1] * frame_width), int(right_wrist[0] * frame_height))

        
        # draw landmarks
        cv2.circle(frame, left_shoulder, 5, (0, 255, 0), -1)  # Green
        cv2.circle(frame, left_elbow, 5, (0, 0, 255), -1)  # Red
        cv2.circle(frame, left_wrist, 5, (255, 0, 0), -1)  # Blue
        cv2.circle(frame, right_shoulder, 5, (0, 255, 0), -1)  # Green
        cv2.circle(frame, right_elbow, 5, (0, 0, 255), -1)  # Red
        cv2.circle(frame, right_wrist, 5, (255, 0, 0), -1)  # Blue
        
        # label landmarks
        cv2.putText(frame, "L Shoulder", (left_shoulder[0] - 20, left_shoulder[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, "L Elbow", (left_elbow[0] - 20, left_elbow[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, "L Wrist", (left_wrist[0] - 20, left_wrist[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(frame, "R Shoulder", (right_shoulder[0] - 20, right_shoulder[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, "R Elbow", (right_elbow[0] - 20, right_elbow[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, "R Wrist", (right_wrist[0] - 20, right_wrist[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # display angle
        # cv2.putText(frame, str(angle), elbow, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # curl counter logic
        if (left_arm_angle > 160 or right_arm_angle > 160):
            if left_arm_angle > 160:
                stage_left_arm = "down"
            elif right_arm_angle > 160:
                stage_right_arm = "down"
            
        if (left_arm_angle < 30 or right_arm_angle < 30):
            if (left_arm_angle < 30 and stage_left_arm == "down"):
                stage_left_arm = "up"
                counter += 1
                update_led_display(counter, RED)
                print("Bicep Curl Reps: ", counter)
            elif (right_arm_angle < 30 and stage_right_arm == "down"):
                stage_right_arm = "up"
                counter += 1
                update_led_display(counter, RED)
                print("Bicep Curl Reps: ", counter)
        
    except:
        pass
    

def squat_counter(frame, keypoints):
    global counter, stage_legs
        # extract landmarks
    try:
        left_hip = keypoints[11][:2]
        right_hip = keypoints[12][:2]
        left_knee = keypoints[13][:2]
        right_knee = keypoints[14][:2]
        left_ankle = keypoints[15][:2]
        right_ankle = keypoints[16][:2]
        
        # calculate knee angles
        left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
        
        frame_height, frame_width, _ = frame.shape
        
        # scale keypoints to frame size
        left_hip = (int(left_hip[1] * frame_width), int(left_hip[0] * frame_height))
        left_knee = (int(left_knee[1] * frame_width), int(left_knee[0] * frame_height))
        left_ankle = (int(left_ankle[1] * frame_width), int(left_ankle[0] * frame_height))
        right_hip = (int(right_hip[1] * frame_width), int(right_hip[0] * frame_height))
        right_knee = (int(right_knee[1] * frame_width), int(right_knee[0] * frame_height))
        right_ankle = (int(right_ankle[1] * frame_width), int(right_ankle[0] * frame_height))
        
        # draw landmarks
        cv2.circle(frame, left_hip, 5, (0, 255, 0), -1)  # Green
        cv2.circle(frame, left_knee, 5, (0, 0, 255), -1)  # Red
        cv2.circle(frame, left_ankle, 5, (255, 0, 0), -1)  # Blue
        cv2.circle(frame, right_hip, 5, (0, 255, 0), -1)  # Green
        cv2.circle(frame, right_knee, 5, (0, 0, 255), -1)  # Red
        cv2.circle(frame, right_ankle, 5, (255, 0, 0), -1)  # Blue
        
                # label landmarks
        cv2.putText(frame, "L Hip", (left_hip[0] - 20, left_hip[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, "L Knee", (left_knee[0] - 20, left_knee[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, "L Ankle", (left_ankle[0] - 20, left_ankle[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(frame, "R Hip", (right_hip[0] - 20, right_hip[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, "R Knee", (right_knee[0] - 20, right_knee[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, "R Ankle", (right_ankle[0] - 20, right_ankle[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # squat counter logic (checks both knees)
        if left_knee_angle > 140 and right_knee_angle > 140:
            stage_legs = "up"
        if left_knee_angle < 100 and right_knee_angle < 100 and stage_legs == "up":
            stage_legs = "down"
            counter += 1
            print("Squat Reps:", counter)
            update_led_display(counter, BLUE)
        
    except:
        pass
    
def summarize_command(text):
    prompt = f"""
You are an exercise assistant. The user may speak casually or with extra words.
Based on the spoken sentence, you will summarize their sentence to just one
command word: it will be either "biceps", "squats", "reset", "temperature",
"humidity", or if the none of the keywords fit or the user isnt saying anything,
respond with "none".

You said: "{text}"
Respond with only one word.
"""
    try:
        response = gemini_model.generate_content(prompt)
        command = response.text.strip().lower()
        return command
    except Exception as e:
        print(f"Gemini API error: {e}")
        return "none"

    
def capture_audio():
    global counter, exercise_mode
    with sr.Microphone(device_index=1) as source:
        while True:
            
            recognizer.adjust_for_ambient_noise(source, duration = 1)
            # capture the audio from the mic
            
            try:
                audio_data = recognizer.listen(source)
                text = recognizer.recognize_google(audio_data)
                print(f"You said: {text}")
                
                # call gemini to summarize
                command = summarize_command(text)
                print(f"Gemini command: {command}")
                
                if command == "reset":
                    counter = 0
                    update_led_display(counter, WHITE)
                    print("Counter Reset.")
                    
                elif command == "squats":
                    counter = 0
                    update_led_display(counter, BLUE)
                    exercise_mode = "squats"
                    
                elif command == "biceps":
                    counter = 0
                    update_led_display(counter, RED)
                    exercise_mode = "biceps"
                    
                elif command == "temperature":
                    temperature = sense.get_temperature()
                    print(f"Current Temperature: {temperature:.1f}°C")
                    temp_int = int(temperature)
                    update_led_display(temp_int, ORANGE)
                    
                elif command == "humidity":
                    humidity = sense.get_humidity()
                    print(f"Current Humidity: {humidity:.1f}%")
                    humidity_int = int(humidity)
                    update_led_display(humidity_int, BROWN)
                    
                else:
                    print("Command invalid!")
                    
            except sr.UnknownValueError:
                # if the speech is unintelligible
                print("The audio is unintelligible.")
            except sr.RequestError as e:
                # request any other errors
                print(f"Could not request results from Google Speech API; {e}")


# run joystick listener in a separate thread to avoid blocking main loop
threading.Thread(target=capture_audio, daemon=True).start()

# main loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((192, 192))
    common.set_input(interpreter, image)
    interpreter.invoke()
    unf_keypoints = np.array(common.output_tensor(interpreter, 0)[0][0], copy=True)
    
    min_confidence = 0.1
    filtered_keypoints = {
            i: keypoint for i, keypoint in enumerate(unf_keypoints) if keypoint[2] >= min_confidence
        }

    # check our exercise mode, run respective function
    if exercise_mode == "biceps":
        bicep_counter(frame, filtered_keypoints)
    else:
        squat_counter(frame, filtered_keypoints)
        
    cv2.imshow("Exercise Assistant", frame)
    
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break
    
cap.release()
cv2.destroyAllWindows()

