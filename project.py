from __future__ import division
from picamera.array import PiRGBArray
from picamera import PiCamera
import imutils
from imutils.video import FPS
from imutils.video import VideoStream
#threading camera fies
from threading import Thread
import cv2
import numpy as np
import time

def hitTest(x1,y1,x2,y2,w1,h1,w2,h2):
    if x1+w1 >= x2:
        if y1+h1 >= y2:
            

            if x1 <= x2+w2:
                if y1 <= y2+h2:
                    return True
    return False

class WebcamVideoStream:
    def __init__(self, src=0):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
 
        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

def nothing(*arg):
        pass

FRAME_WIDTH = 320
FRAME_HEIGHT = 240

# Initial HSV GUI slider values to load on program start.
icol = (0, 0, 0, 255, 255, 255)
# Set HSV values for blue folder.
blueLower = (20, 0, 143)
blueUpper = (170, 114, 255)

cv2.namedWindow('Tracking')
# Lower range colour sliders.
cv2.createTrackbar('lowHue', 'Tracking', icol[0], 255, nothing)
cv2.createTrackbar('lowSat', 'Tracking', icol[1], 255, nothing)
cv2.createTrackbar('lowVal', 'Tracking', icol[2], 255, nothing)
# Higher range colour sliders.
cv2.createTrackbar('highHue', 'Tracking', icol[3], 255, nothing)
cv2.createTrackbar('highSat', 'Tracking', icol[4], 255, nothing)
cv2.createTrackbar('highVal', 'Tracking', icol[5], 255, nothing)

# Initialize webcam. Webcam 0 
vidCapture = cv2.VideoCapture(0)
vidCapture.set(cv2.CAP_PROP_FRAME_WIDTH,FRAME_WIDTH)
vidCapture.set(cv2.CAP_PROP_FRAME_HEIGHT,FRAME_HEIGHT)

while True:
    timeCheck = time.time()
    # Get HSV values from the GUI sliders.
    lowHue = cv2.getTrackbarPos('lowHue', 'Tracking')
    lowSat = cv2.getTrackbarPos('lowSat', 'Tracking')
    lowVal = cv2.getTrackbarPos('lowVal', 'Tracking')
    highHue = cv2.getTrackbarPos('highHue', 'Tracking')
    highSat = cv2.getTrackbarPos('highSat', 'Tracking')
    highVal = cv2.getTrackbarPos('highVal', 'Tracking')


    # Get webcam frame
    _, frame = vidCapture.read()

    # Convert the frame to HSV colour model.
    frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Repeated for second HSV frame.
    frame2HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    
    # HSV values to define a colour range we want to create a mask from.
    colorLow = np.array([lowHue,lowSat,lowVal])
    colorHigh = np.array([highHue,highSat,highVal])
    mask = cv2.inRange(frameHSV, colorLow, colorHigh)
    #Blur the image to remove static and make tracking easier
    blurred2 = cv2.GaussianBlur(frameHSV, (11, 11), 0)
    hsv2 = cv2.cvtColor(blurred2, cv2.COLOR_BGR2HSV)
    # Show the first mask
    cv2.imshow('mask - tennis ball', mask)
    
    # resize the frame, blur it, and convert it to the HSV
    # color space
    frame2 = imutils.resize(frame, width=320, height=240)
    blurred = cv2.GaussianBlur(frame2, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    # construct a mask for the color "green", then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask
    mask2 = cv2.inRange(hsv, blueLower, blueUpper)
    
    #show the second mask
    #cv2.imshow('mask-folder', mask2)
    

    #set up boundaries in order to find biggest contours and draw a box around them for tennis ball
    im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #set up boundaries in order to find biggest contours and draw a box around them for surface
    im2, contours2, hierarchy = cv2.findContours(mask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    #find biggest contours for tennis ball
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
    #find biggest contours for surface
    contour_sizes2 = [(cv2.contourArea(contour), contour) for contour in contours2]
    biggest_contour2 = max(contour_sizes2, key=lambda x: x[0])[1]
    
    #draw green box around tennis ball
    x,y,w,h = cv2.boundingRect(biggest_contour)
    tennisBall = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    #draw blue box arround surface
    x2,y2,w2,h2 = cv2.boundingRect(biggest_contour2)
    surface = cv2.rectangle(frame2,(x2,y2),(x2+w2,y2+h2),(255,0,0),2)
    
    if hitTest(x,y,x2,y2,w,h,w2,h2) == True:
        tennisBall = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
    
    # Show final output image
    cv2.imshow('Tracking', frame)
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
    print('FPS: - ', 1/(time.time() - timeCheck))
    
cv2.destroyAllWindows()
vidCapture.release()