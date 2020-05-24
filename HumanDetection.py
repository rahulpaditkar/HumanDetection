import cv2
import cv2utils


face_cascade=cv2.CascadeClassifier("C:\\Users\\Rahul\\PycharmProjects\\FaceDetection\\haarcascade_frontalface_default.xml")

img=cv2.imread("C:\\Users\\Rahul\\PycharmProjects\\FaceDetection\\animal1.jpg",1)

gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faces=face_cascade.detectMultiScale(gray_img,scaleFactor=1.05,minNeighbors=5)


for x,y,w,h in faces:
    img=cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
    print("Human is Detected")
else:
    print("Human is not Detected")

cv2.imshow("Image",img)

cv2.waitKey(0)
cv2.destroyAllWindows()