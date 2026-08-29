import cv2
import numpy as np
import os
import mediapipe as mp

# ১. কনফিগারেশন (Settings)
DATA_PATH = os.path.join('MP_Data_Final') # সেভ করা ডেটা এখানে জমা হবে
actions = np.array(['Normal', 'Fall'])    # আমাদের দুটি ক্লাস
no_sequences = 28                         # প্রতিটি অ্যাকশনের জন্য ৫০টি ভিডিও ক্লিপ
sequence_length = 60                      # প্রতিটি ক্লিপে ৬০টি ফ্রেম (২ সেকেন্ড)

# ২. মিডিয়াপাইপ সেটআপ
mp_holistic = mp.solutions.holistic 
mp_drawing = mp.solutions.drawing_utils 

def mediapipe_detection(image, model):
    image = cv2.resize(image, (640, 480)) # একটি স্ট্যান্ডার্ড সাইজে রিসাইজ করা
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False                  
    results = model.process(image)                 
    image.flags.writeable = True                   
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
    return image, results

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

# ৩. ফোল্ডার তৈরির লজিক (একবারই রান হবে)
for action in actions: 
    for sequence in range(no_sequences):
        os.makedirs(os.path.join(DATA_PATH, action, str(sequence)), exist_ok=True)

# ৪. মেইন ডেটা কালেকশন লুপ (ভিডিও ফাইল থেকে)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    for action in actions:
        # আপনার ভিডিওগুলো যেখানে আছে সেই ফোল্ডার পাথ
        video_folder = os.path.join('Raw_Videos', action) 
        # ফোল্ডারের সব ভিডিও ফাইলের লিস্ট
        video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi'))]
        
        print(f"প্রসেসিং শুরু হচ্ছে: {action} ফোল্ডার। মোট ভিডিও পাওয়া গেছে: {len(video_files)}")

        for sequence in range(no_sequences):
            # যদি ফোল্ডারে ভিডিওর সংখ্যা কম থাকে, তবে লুপ থামিয়ে দাও
            if sequence >= len(video_files):
                print(f"সতর্কতা: {action} এর জন্য আর ভিডিও ফাইল নেই।")
                break
                
            video_path = os.path.join(video_folder, video_files[sequence])
            cap = cv2.VideoCapture(video_path)
            
            print(f"ভিডিও প্রসেস হচ্ছে: {video_files[sequence]} -> ফোল্ডার {sequence}")

            for frame_num in range(sequence_length):
                ret, frame = cap.read()

                if not ret:
                    # যদি ভিডিও ৬০ ফ্রেমের আগে শেষ হয়, তবে জিরো ডেটা সেভ করো যাতে লজিক না ভাঙে
                    keypoints = np.zeros(33*4)
                else:
                    # ডিটেকশন করা
                    image, results = mediapipe_detection(frame, holistic)
                    # ডেটা বের করা
                    keypoints = extract_keypoints(results)

                # ৫. ফাইলে সেভ করা (.npy ফরমেটে)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                # # স্ক্রিনে দেখানো (জাস্ট দেখার জন্য, কিবোর্ডে q চাপলে পরের ভিডিওতে যাবে)
                # if ret:
                #     cv2.imshow('Collecting Data...', image)
                #     if cv2.waitKey(1) & 0xFF == ord('q'):
                #         break
            
            cap.release()

    cv2.destroyAllWindows()

