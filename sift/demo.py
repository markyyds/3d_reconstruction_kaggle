import numpy as np
import cv2 as cv

if __name__ == '__main__':
    img = cv.imread('demo.jpg')
    gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create()
    kp = sift.detect(gray,None)
    img=cv.drawKeypoints(gray,kp,img)
    img=cv.drawKeypoints(gray,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv.imwrite('sift_keypoints.jpg',img)