import cv2

face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#img=cv2.imread("photo.jpg") #no 0 or 1 or -1 means the orignal color
img=cv2.imread("group.jpg")
gray_img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert to grayScale

#instead of these two commands, we can directly store grayScaleimage but i want to display colored image

faces= face_cascade.detectMultiScale(gray_img, scaleFactor=1.15, minNeighbors=5)
print(faces) #numpy ndarray

for x,y,w,h in faces:
    img=cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,255),5)


#new_img= cv2.resize(img, (int(img.shape[1]/3), int(img.shape[0]/3)))

new_img= cv2.resize(img, (1000,500))

cv2.imshow("Image",new_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
