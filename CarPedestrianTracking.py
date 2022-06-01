import cv2
from random import randrange

classifier_file = 'cars.xml'
img_file = 'cars2.jpg'


# load a image
img_color = cv2.imread('cars2.jpg')
img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
# cv2.imshow('test', img_gray)
# cv2.waitKey()
classifier = cv2.CascadeClassifier('cars.xml')
tracker = classifier.detectMultiScale(img_gray)
for (x, y, w, h) in tracker:
    cv2.rectangle(img_color, (x, y), (x + w, y + h), (randrange(128, 256), randrange(256), randrange(256)), 2)
cv2.imshow('cars', img_color)
cv2.waitKey()

# load a video
cap = cv2.VideoCapture('car1.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("can't receive frames (stream end?). Exciting")
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tracker = classifier.detectMultiScale(gray)
    for (x, y, w, h) in tracker:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (randrange(128, 256), randrange(256), randrange(256)), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(25) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


print('Code Completed')


