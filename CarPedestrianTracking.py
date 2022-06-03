import cv2
from random import randrange

car_track_file = 'cars.xml'
pedestrian_track_file = 'haarcascade_fullbody.xml'
img_file = ['cars.jpg', 'cars2.jpg', 'car_ped_1.jpg', 'car_ped_2.jpg', 'car_ped_3.jpg', 'car_ped_4.jpg', 'car_ped_5.jpg']

car_classifier = cv2.CascadeClassifier(car_track_file)
pedestrian_classifier = cv2.CascadeClassifier(pedestrian_track_file)

# for img in img_file:
#     img_color = cv2.imread(img)
#     img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
#     car_tracker = car_classifier.detectMultiScale(img_gray)
#     ped_tracker = pedestrian_classifier.detectMultiScale(img_gray)
#     for (x, y, w, h) in car_tracker:
#         cv2.rectangle(img_color, (x, y), (x + w, y + h), (0, 255, 0), 2)
#     for (x, y, w, h) in ped_tracker:
#         cv2.rectangle(img_color, (x, y), (x + w, y + h), (255, 0, 0), 2)
#     cv2.imshow('cars', img_color)
#     cv2.waitKey()

# load a video
cap = cv2.VideoCapture('car1.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("can't receive frames (stream end?). Exciting")
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    car_tracker = car_classifier.detectMultiScale(gray)
    for (x, y, w, h) in car_tracker:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    ped_tracker = pedestrian_classifier.detectMultiScale(gray)
    for (x, y, w, h) in ped_tracker:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.imshow('frame', frame)
    if cv2.waitKey(50) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

print('Code Completed')


