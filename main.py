import pyfirmata
import time
import cv2

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
address = "http://10.5.238.23:8080/video"
cap = cv2.VideoCapture(0)
cap.open(address)
while True: 
    check,frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        img = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    cv2.imshow("Blindr", frame)
    key = cv2.waitKey(30)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


