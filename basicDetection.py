import cv2
import cv2utils
import imutils


hog=cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

img=cv2.imread("C:\\Users\\Rahul\\PycharmProjects\\FaceDetection\\animal1.jpg",1)


img = imutils.resize(img,width=min(400, img.shape[1]))
(regions, _) = hog.detectMultiScale(img,winStride=(4, 4),padding=(4, 4),scale=1.05)

for x,y,w,h in regions:
    img=cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
    print("Human is Detected")
else:
    print("Human is not detected")

cv2.imshow("Image",img)

cv2.waitKey(0)
cv2.destroyAllWindows()