import cv2
import numpy as np
# img=cv2.imread("picsandvid/scene_2.png")

# cv2.imshow("output",img)

# cv2.waitKey(0)

# cap=cv2.VideoCapture("picsandvid/scene_2.mp4")

# while True:
#     success, img = cap.read()
#     cv2.imshow("vid",img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap=cv2.VideoCapture(0)
# cap.set(3,640)
# cap.set(4,480)
# cap.set(10,100)
# while True:
#     success, img = cap.read()
#     cv2.imshow("vid",img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# img=cv2.imread("picsandvid/scene_2.png")
# print(img.shape)

# imgResize=cv2.resize(img,(400,600))
# imgCropped=img[200:300,100:200]
# # cv2.imshow("image",img)
# # cv2.imshow("image2",imgResize)
# cv2.imshow("i",imgCropped)
# cv2.waitKey(0)

img = np.zeros((512,512,3),np.uint8)
# print(img.shape)
# img[:]=255,0,0
# cv2.line(img,(0,0),(300,300),(0,250,0),3)
cv2.line(img,(0,0),(img.shape[1],img.shape[0]),(0,250,0),3)
cv2.rectangle(img,(0,0),(250,350),(0,0,250),3)
cv2.circle(img,(300,300),50,(250,250,0),1)
cv2.putText(img,"OpenCv",(300,500),cv2.FONT_HERSHEY_DUPLEX,2,(0,150,0),2)
cv2.imshow("image",img)

cv2.waitKey(0)