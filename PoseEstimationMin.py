import cv2
import mediapipe as mp
import time

mpDraw=mp.solutions.drawing_utils
mpPose=mp.solutions.pose
pose=mpPose.Pose()
path='picsandvid/1.mp4'

cap= cv2.VideoCapture(path)
pTime=0
while True:
    success, img = cap.read()
    if not success:
        print("Ended")
        break
    img = cv2.resize(img, (512, 600), interpolation=cv2.INTER_AREA)
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results=pose.process(imgRGB)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img,results.pose_landmarks,mpPose.POSE_CONNECTIONS)
        for id , lm in enumerate(results.pose_landmarks.landmark):
            h , w , c = img.shape
            cx,cy=int(lm.x*w),int(lm.y*h)
            cv2.circle(img,(cx,cy),5,(255,0,0),cv2.FILLED)
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.putText(img,str(int(fps)),(0,40), cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)
    
    cv2.imshow("Image",img)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()