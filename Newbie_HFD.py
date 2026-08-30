import cv2
import numpy as np
import os
import mediapipe as mp


DATA_PATH = os.path.join('MP_Data_Final') 
actions = np.array(['Normal', 'Fall'])   
no_sequences = 28                        
sequence_length = 60                     


mp_holistic = mp.solutions.holistic 
mp_drawing = mp.solutions.drawing_utils 

def mediapipe_detection(image, model):
    image = cv2.resize(image, (640, 480)) 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False                  
    results = model.process(image)                 
    image.flags.writeable = True                   
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
    return image, results


last_known_pose = np.zeros(132) 

def extract_keypoints(results):
    global last_known_pose 
    
    if results.pose_landmarks:
        
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten()
        last_known_pose = pose 
        return pose
    else:
        
        return last_known_pose


for action in actions: 
    for sequence in range(no_sequences):
        os.makedirs(os.path.join(DATA_PATH, action, str(sequence)), exist_ok=True)


with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    for action in actions:
        
        video_folder = os.path.join('Raw_Videos', action) 
       
        video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi'))]
        
        print(f"প্রসেসিং শুরু হচ্ছে: {action} ফোল্ডার। মোট ভিডিও পাওয়া গেছে: {len(video_files)}")

        for sequence in range(no_sequences):
          
            if sequence >= len(video_files):
                print(f"সতর্কতা: {action} এর জন্য আর ভিডিও ফাইল নেই।")
                break
                
            video_path = os.path.join(video_folder, video_files[sequence])
            cap = cv2.VideoCapture(video_path)
            
            print(f"ভিডিও প্রসেস হচ্ছে: {video_files[sequence]} -> ফোল্ডার {sequence}")

            for frame_num in range(sequence_length):
                ret, frame = cap.read()

                if not ret:
         
                    keypoints = np.zeros(33*4)
                else:
                   
                    image, results = mediapipe_detection(frame, holistic)
                  
                    keypoints = extract_keypoints(results)

              
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

               
                # if ret:
                #     cv2.imshow('Collecting Data...', image)
                #     if cv2.waitKey(1) & 0xFF == ord('q'):
                #         break
            
            cap.release()

    cv2.destroyAllWindows()

