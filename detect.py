import cv2
import time
# Enable Notifications
# from plyer import notification


cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 420)

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalcatface_extended.xml")



while True:
    success, img = cap.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    faces = faceCascade.detectMultiScale(imgGray, 1.1, 1) 

    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
        
        if success:
            cv2.imwrite(time.strftime("%Y%m%d-%H%M%S.jpg"), img)

    cv2.imshow('face_detect', img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break


    



cap.release()
cv2.destroyWindow('face_detect')

#Enable notifications

# notification.notify(
#     title = 'testing',
#     message = 'message',
#     app_icon = None,
#     timeout = 10,
# )