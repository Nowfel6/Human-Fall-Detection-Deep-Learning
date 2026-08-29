import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

def mediapipe_detection(image, model):
    image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results


def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections

def draw_styled_landmarks(image, results):
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 

# একদম শুরুতে একটি মেমরি ভেরিয়েবল তৈরি করে নিন
last_known_pose = np.zeros(132) 

def extract_keypoints(results):
    global last_known_pose # গ্লোবাল মেমরি ব্যবহার করছি
    
    if results.pose_landmarks:
        # যদি মানুষ দেখতে পায়, তবে নতুন ডেটা নেবে এবং মেমরিতে সেভ করে রাখবে
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten()
        last_known_pose = pose 
        return pose
    else:
        # যদি মানুষ হারিয়ে যায় বা মেঝেতে শুয়ে পড়ে, তবে জিরো না বসিয়ে আগের পজিশনটাই কপি করে দেবে!
        return last_known_pose


cap = cv2.VideoCapture('Raw_Videos/Normal/38_n.mp4')
# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()
        if not ret:
            print("Ended")
            break
        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        # print(results)
        
        # Draw landmarks
        draw_styled_landmarks(image, results)

        # Show to screen
        cv2.imshow('OpenCV Feed', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# # Main Folder Name
# DATA_PATH = os.path.join('Fall_Detection_Data') 
# actions = np.array(['Normal', 'Fall'])

# # for every action the number of videos
# no_sequences = 50

# # No of Frame in every video
# sequence_length = 60

# # Logic to make folder 
# for action in actions: 
#     for sequence in range(no_sequences):
#         try: 
#             # Fall_Detection_Data/Normal/0, Fall_Detection_Data/Normal/1... 
#             os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
#         except:
#             pass

