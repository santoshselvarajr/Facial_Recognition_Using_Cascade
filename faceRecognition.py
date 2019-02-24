#Working in the virtual_platform
#Required libraries are installed in the virtual platform

#Importing libraries
import cv2
import os

#Set the working directory
os.chdir("C:\\Users\\Santosh Selvaraj\\Documents\\Working Directory\\Computer_Vision_A_Z_Template_Folder\\Module 1 - Face Recognition")

#Loading the cascades
#Apply filters to detect face using cascading
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
#Apply filter to detect eye using cascading
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
#Detect smile
smile_cascade = cv2.CascadeClassifier("haarcascade_smile.xml")

#Function to detect the face and eye
def detect(gray,frame):
    #Detect coordinates of the faces (x,y,w,h) on the gray image and superimpose it back on the color image
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    #Draw a rectangle around each face detected
    for (x,y,w,h) in faces:
        #Give the frame on which to apply, coordinates, color, width of rectangle
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0), 2)
        #Zone of interest where the face is present - Efficient
        #Detect eyes in grayscale
        roi_gray = gray[y:y+h,x:x+w]
        #draw rectangle on color image
        roi_color = frame[y:y+h,x:x+w]
        #Detect eyes in the face region
        eyes = eye_cascade.detectMultiScale(roi_gray,1.1,20)
        for (ex,ey,ew,eh) in eyes:
            #Draw rectangles around eyes
            cv2.rectangle(roi_color,(ex,ey), (ex+ew,ey+eh),(0,255,0), 2)
        #Detect the smiles
        smiles = smile_cascade.detectMultiScale(roi_gray,1.7,20)
        for (sx,sy,sw,sh) in smiles:
            #Draw rectangles around eyes
            cv2.rectangle(roi_color,(sx,sy), (sx+sw,sy+sh),(0,0,255), 2)
    return frame

#Doing recognition using Webcam
#0 for internal cameras and 1 for external cameras
video_capture = cv2.VideoCapture(0)

while True:
    #Read gives 2 outputs but we need only the second one
    _, frame = video_capture.read()
    #Convert the colored image to grayscale
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #Returns the frame after applying the detect function
    canvas = detect(gray,frame)
    #Open a window named 'Video' with the returned images
    cv2.imshow('Video', canvas)
    #Break the webcam if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#Exit the webcam
video_capture.release()
#Destroy the windows with the images
cv2.destroyAllWindows()