import cv2

# Image of a car
img_file = "cars.webp"
# video_file = cv2.VideoCapture("video1.mp4")
video_file = cv2.VideoCapture("video2.mp4")
# video_file = cv2.VideoCapture("video3.mp4")
# Pre-trained car classifier
car_classifier_file = "cars.xml"
fullbody_classifier_file = "haarcascade_fullbody.xml"

# opencv reads image
# img = cv2.imread(img_file)
# grayScaled = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# car_classifer = cv2.CascadeClassifier(car_classifier_file)
# cars_coords = car_classifer.detectMultiScale(grayScaled)
# for (x, y, w, h) in cars_coords:
#     cv2.rectangle(img, (x, y), (x+w, y+h),(0,0,255), 2)

while True:
    (read_success, frame) = video_file.read()
    if read_success:
        grayScaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break
    car_classifer = cv2.CascadeClassifier(car_classifier_file)
    people_classifer = cv2.CascadeClassifier(fullbody_classifier_file)
    cars_coords = car_classifer.detectMultiScale(grayScaled_frame)
    people_coords = people_classifer.detectMultiScale(grayScaled_frame)
    print(cars_coords)
    for (x, y, w, h) in cars_coords:
        cv2.rectangle(frame, (x, y), (x+w, y+h),(0,0,255), 2)
    for (x, y, w, h) in people_coords:
        cv2.rectangle(frame, (x, y), (x+w, y+h),(255,0,0), 2)
    cv2.imshow("frame", frame)
    key = cv2.waitKey(1)
    if key == 81 or key == 113:
        break
video_file.release()

# cv2.imshow("img", img)
# cv2.waitKey()

print("Code Completed")