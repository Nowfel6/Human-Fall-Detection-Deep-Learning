import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# ==========================================
# পার্ট ১: এআই ব্রেইন (মডেল) ঘুম থেকে তোলা
# ==========================================
print("এআই মডেল লোড হচ্ছে, দয়া করে অপেক্ষা করুন...")
model = load_model('Fall_Detection_Model.h5')
actions = np.array(['Normal', 'Fall'])

# ==========================================
# পার্ট ২: মিডিয়াপাইপ এবং ভেরিয়েবল সেটআপ
# ==========================================
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

sequence =[]       # এটি আমাদের স্লাইডিং উইন্ডো (Sliding Window)
predictions =[]    # মডেলের উত্তরের হিস্ট্রি
threshold = 0.7     # কনফিডেন্স লেভেল (৮০% এর ওপর শিউর হলে তবেই অ্যালার্ম দেবে)

# ফাংশন: মিডিয়াপাইপ ডিটেকশন
def mediapipe_detection(image, model):
    image = cv2.resize(image, (640, 480)) # ট্রেনিংয়ের সময় এই সাইজেই ছিল
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False                  
    results = model.process(image)                 
    image.flags.writeable = True                   
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
    return image, results

# ফাংশন: জয়েন্ট বা পয়েন্ট বের করা
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

# ==========================================
# পার্ট ৩: লাইভ ওয়েবক্যাম এবং প্রেডিকশন
# ==========================================
# ০ মানে ওয়েবক্যাম। আপনি চাইলে এখানে কোনো ভিডিওর পাথ ('video.mp4') দিতে পারেন।
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
        
        # ২. স্লাইডিং উইন্ডো লজিক (The Magic Box)
        sequence.append(keypoints)      # নতুন ফ্রেম লিস্টে ঢোকালাম
        sequence = sequence[-60:]       # শুধুমাত্র শেষের ৬০টি ফ্রেমই রাখলাম (কারণ আমাদের মডেল ৬০ ফ্রেমের)
        
        # ৩. প্রেডিকশন লজিক
        # যখনই আমাদের বাক্সে ৬০টি ফ্রেম জমে যাবে, তখনই মডেলকে ডাকব
        if len(sequence) == 60:
            # মডেলের ইনপুট শেপ (1, 60, 132) বানানোর জন্য expand_dims করা হলো
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(f"Probabilities -> Normal: {res[0]:.2f}, Fall: {res[1]:.2f}")
            # মডেলের উত্তর কোন ক্লাসের (০ নাকি ১)?
            predicted_class_index = np.argmax(res)
            
            # যদি মডেলের কনফিডেন্স threshold (৮০%) এর বেশি হয়
            if res[predicted_class_index] > threshold:
                action_name = actions[predicted_class_index]
                
                # ৪. স্ক্রিনে অ্যালার্ম বা টেক্সট দেখানো
                if action_name == 'Fall':
                    # স্ক্রিনের ওপর একটি লাল রঙের বক্স
                    cv2.rectangle(image, (0, 0), (640, 60), (0, 0, 255), -1)
                    cv2.putText(image, '!!! FALL DETECTED !!!', (120, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
                else:
                    # স্ক্রিনের ওপর একটি সবুজ রঙের বক্স
                    cv2.rectangle(image, (0, 0), (640, 60), (0, 255, 0), -1)
                    cv2.putText(image, 'NORMAL / SAFE', (200, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)

        # ৫. মানুষের কঙ্কাল (Skeleton) আঁকা (যাতে দেখতে সুন্দর লাগে)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        # ৬. স্ক্রিনে ভিডিও দেখানো
        cv2.imshow('Human Fall Detection - AI Live', image)

        # 'q' চাপলে বন্ধ হবে
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()