import cv2
from random import randrange

# importing the pre-trained data on face frontals from opencv
front_face_data = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


# # use the image read funtion from opencv to detect faces in image
# # img = cv2.imread("AI Face Detector/Elon.png")
# img = cv2.imread("AI Face Detector/friends.jpeg")

# # convert image to grayscale
# grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # detectMultiScale = detects object with different sizes
# face_coords = front_face_data.detectMultiScale(img)
# # print(face_coords)

# # get the face coordanates [[262,100,330,330]]
# # (x, y, w, h) = face_coords[0]
# for (x, y, w, h) in face_coords:
#     # Draw the rectangle cv2.rectangle(object, (tuple), (tuple), (color code), (thickness))
#     cv2.rectangle(img, (x, y), (x+w, y+h), (randrange(256),
#                   randrange(256), randrange(256)), 3)

# cv2.imshow('Img', img)
# key = cv2.waitKey()
# # stop program if Q pressed
#     if key == 81 or key == 113:
#         break


# use the videoCapture funtion from opencv to detect faces in video
webcam = cv2.VideoCapture(0)

# loop over video frames
while True:
    # read current frame, returns bool and current frame
    frame_read, frame = webcam.read()
    grayScale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_coords = front_face_data.detectMultiScale(grayScale_frame)
    for (x, y, w, h) in face_coords:
        cv2.rectangle(frame, (x, y), (x+w, y+h),
                      (randrange(256), randrange(256), randrange(256)), 3)
    cv2.imshow('webcam', frame)
    # 1 becasue it detects real time
    key = cv2.waitKey(1)
    # stop program if Q pressed
    if key == 81 or key == 113:
        break
# Relsase the video capture object
# webcam.relase()

print("tesing python")