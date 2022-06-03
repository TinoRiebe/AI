import cv2
from random import randrange

trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Choose an image and convert it to greyscale
# img = cv2.imread('RDJ.jpg')
# gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow('test', gray_img)
# cv2.waitKey()

# Connect to webcam
# vid = cv2.VideoCapture(0)

# load a video
cap = cv2.VideoCapture('11.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    # if frame is read creectly ret is True
    if not ret:
        print("can't receive frames (stream end?). Exciting")
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_coordinates = trained_face_data.detectMultiScale(gray)
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (randrange(128, 256), randrange(256), randrange(256)), 2)
        roi_grey = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_grey)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(25) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()





# face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
#
#
# for (x, y, w, h) in face_coordinates:
#     cv2.rectangle(img, (x, y), (x+w, y+h), (randrange(128, 256), randrange(256), randrange(256)), 2)
#     roi_grey = grayscaled_img[y:y+h, x:x+w]
#     roi_color = img[y:y+h, x:x+w]
#     eyes = eye_cascade.detectMultiScale(roi_grey)
#     for (ex, ey, ew, eh) in eyes:
#         cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
#
#
# cv2.imshow('test', img)
# cv2.waitKey()
# # print(face_coordinates)

print('Code Completed')