import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model


print("এআই মডেল লোড হচ্ছে, দয়া করে অপেক্ষা করুন...")
model = load_model('Fall_Detection_Model.h5')
actions = np.array(['Normal', 'Fall'])

# ==========================================
#  মিডিয়াপাইপ এবং ভেরিয়েবল সেটআপ
# ==========================================
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

sequence =[]       #  স্লাইডিং উইন্ডো (Sliding Window)
predictions =[]    # মডেলের উত্তরের হিস্ট্রি
threshold = 0.7     # কনফিডেন্স লেভেল 

# ফাংশন: মিডিয়াপাইপ ডিটেকশন
def mediapipe_detection(image, model):
    image = cv2.resize(image, (640, 480)) 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False                  
    results = model.process(image)                 
    image.flags.writeable = True                   
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
    return image, results

# একদম শুরুতে একটি মেমরি ভেরিয়েবল তৈরি 
last_known_pose = np.zeros(132) 

def extract_keypoints(results):
    global last_known_pose # গ্লোবাল মেমরি
    
    if results.pose_landmarks:
        
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten()
        last_known_pose = pose 
        return pose
    else:
        
        return last_known_pose

# ==========================================
# লাইভ ওয়েবক্যাম এবং প্রেডিকশন
# ==========================================

cap = cv2.VideoCapture(0)

print("ক্যামেরা চালু হচ্ছে... বন্ধ করতে 'q' চাপুন।")

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("ক্যামেরা থেকে ফ্রেম পাওয়া যাচ্ছে না!")
            break
            
        # ১. ডিটেকশন এবং পয়েন্ট এক্সট্রাক্ট
        image, results = mediapipe_detection(frame, holistic)
        keypoints = extract_keypoints(results)
        
        # ২. স্লাইডিং উইন্ডো লজিক 
        sequence.append(keypoints)      # নতুন ফ্রেম 
        sequence = sequence[-60:]       # শুধুমাত্র শেষের ৬০টি ফ্রেম
        
        # ৩. প্রেডিকশন লজিক
        
        if len(sequence) == 60:
            # মডেলের ইনপুট শেপ (1, 60, 132) 
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(f"Probabilities -> Normal: {res[0]:.2f}, Fall: {res[1]:.2f}")
            
            predicted_class_index = np.argmax(res)
            
            
            if res[predicted_class_index] > threshold:
                action_name = actions[predicted_class_index]
                
                ো
                if action_name == 'Fall':
                    
                    cv2.rectangle(image, (0, 0), (640, 60), (0, 0, 255), -1)
                    cv2.putText(image, '!!! FALL DETECTED !!!', (120, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
                else:
                    স
                    cv2.rectangle(image, (0, 0), (640, 60), (0, 255, 0), -1)
                    cv2.putText(image, 'NORMAL / SAFE', (200, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)

        
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        
        cv2.imshow('Human Fall Detection - AI Live', image)

        ে
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
