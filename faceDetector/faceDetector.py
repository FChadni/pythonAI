import cv2
from random import randrange

# importing the pre-trained data on face frontals from opencv
front_face_data = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
smile_face_data = cv2.CascadeClassifier("haarcascade_smile.xml")
# use the image read funtion from opencv to detect faces in image
img = cv2.imread("friends.jpeg")
def face_detection_image():
    # convert image to grayscale
    grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # detectMultiScale = detects object with different sizes
    face_coords = front_face_data.detectMultiScale(img)
    # print(face_coords)
    # get the face coordanates [[262,100,330,330]]
    # (x, y, w, h) = face_coords[0]
    for (x, y, w, h) in face_coords:
        # Draw the rectangle cv2.rectangle(object, (tuple), (tuple), (color code), (thickness))
        cv2.rectangle(img, (x, y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 3)
        # Get the subframe of the image (using nympy)
        face_frame = img[y:y+h, x:x+w]
        grayscaled_face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2GRAY)
        smile_coords = smile_face_data.detectMultiScale(grayscaled_face_frame, scaleFactor=1.7, minNeighbors=20)
        for (x1, y1, w1, h1) in smile_coords:
            cv2.rectangle(face_frame, (x1, y1), (x1+w1, y1+h1), (0, 0, 255), 2)
        if len(smile_coords) > 0:
            cv2.putText(img, "Smiling", (x, y+h+40), fontScale=3, fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255,255,255))
    cv2.imshow('Img', img)
    cv2.waitKey()
face_detection_image()

def face_detection_webcam():
    # use the videoCapture funtion from opencv to detect faces in video
    webcam = cv2.VideoCapture(0)

    while True:
        # read current frame, returns bool and current frame
        frame_read, frame = webcam.read()
        grayScale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_coords = front_face_data.detectMultiScale(grayScale_frame)
        for (x, y, w, h) in face_coords:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 3)
        cv2.imshow('webcam', frame)
        # 1 becasue it detects real time
        key = cv2.waitKey(1)
        # stop program if Q pressed
        if key == 81 or key == 113: break
    # Relsase the video capture object
    webcam.relase()
# face_detection_webcam()

print("tesing python")