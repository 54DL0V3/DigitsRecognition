# myHough.py
import numpy as np
import math
import cv2 as cv

def myHoughLine(imgThresh,img, minVertical, minHorizontal, showSteps):

    # create Hough Space
    h = len(imgThresh)
    w = len(imgThresh[0])
    rho = max(h,w) #int(math.sqrt(pow(h,2)+pow(w,2)))
    theta = 360
    scale = 1
    houghLine = np.zeros((rho*scale,theta*scale),np.uint8)
    np.savetxt("houghLine.txt",houghLine,fmt = '%5d')

    # calculating

    imgPossibleHorizontalLines = np.zeros((h,w),np.uint8)

    #find horizontal lines
    for i in range(rho*scale):
        if (30<int(i/scale)<h-30) or (int(i/scale)>=h):
            continue
        for j in range(theta*scale):
            if((j/scale)<75) or (115<(j/scale)):
                continue
            imgHoughLine = imgThresh.copy()
            rad_Theta = np.deg2rad(j/scale)
            for x in range (w):
                y = int((i/scale-x*math.cos(rad_Theta))/math.sin(rad_Theta))
                if (0<=y<h):
                    imgHoughLine[y,x]=255
                    if(imgThresh[y,x]==255):
                        houghLine[i,j]+=1
            if (houghLine[i,j] < minHorizontal):
                houghLine[i,j] = 0
            else:
                # Draw possible lines
                for x in range (w):
                    y = int((i/scale-x*math.cos(rad_Theta))/math.sin(rad_Theta))
                    if (0<=y<h):
                        imgPossibleHorizontalLines[y,x]=255

            if showSteps:
                cv.imshow("imgHoughLine",imgHoughLine)
                cv.imshow("imgPossibleHorizontalLines",imgPossibleHorizontalLines)
                cv.waitKey(1)
    
    #find vertical lines
    imgPossibleVerticalLines = np.zeros((h,w),np.uint8)
    for i in range(rho*scale):
        if (60<int(i/scale)<w-30) or (int(i/scale)>=w):
            continue
        for j in range(theta*scale):
            if 15<(j/scale)<345 :
                continue
            imgHoughLine = imgThresh.copy()
            rad_Theta = np.deg2rad(j/scale)
            for y in range (h):
                x = int((i/scale-y*math.sin(rad_Theta))/math.cos(rad_Theta))
                if (0<=x<w):
                    imgHoughLine[y,x]=255
                    if(imgThresh[y,x]==255):
                        houghLine[i,j]+=1
            if (houghLine[i,j] < minVertical):
                houghLine[i,j] = 0
            else:
                # Draw possible lines
                for y in range (h):
                    x = int((i/scale-y*math.sin(rad_Theta))/math.cos(rad_Theta))
                    if (0<=x<w):
                        imgPossibleVerticalLines[y,x]=255

            if showSteps :
                cv.imshow("imgHoughLine",imgHoughLine)
                cv.imshow("imgPossibleVerticalLines",imgPossibleVerticalLines)
                cv.waitKey(1)

    
    #np.savetxt("houghLine.txt",houghLine,fmt = '%3d')
    if showSteps :
        cv.imshow("houghSpace",houghLine)
        cv.waitKey(1)

    ##########################################################################################################
    
    # find all intersections

    intersections = np.zeros((h,w),np.uint8)
    for i in range(h):
        for j in range(w):
            if imgPossibleVerticalLines[i,j] == 255 and imgPossibleHorizontalLines[i,j] == 255:
                intersections[i,j] = 1
    
    # find top-left intersection
    yTL = 0
    xTL = 0

    yBL = h-1
    xBL = 0

    yTR = 0
    xTR = w-1

    yBR = h-1
    xBR = w-1 

    temp = 0
    for i in reversed(range(int(h/2))):
        for j in reversed(range(int(w/2))):
            if (intersections[i,j] == 1) and temp < (pow(i,2) + pow (j,2)):
                temp = (pow(i,2) + pow(j,2))
                yTL = i
                xTL = j
    
    
    # find bottom-left intersection
    temp = 0
    for i in range(int(h/2),h):
        for j in reversed(range(int(w/2))):
            if intersections[i,j] == 1 and temp < (pow(h-1-i,2) + pow (j,2)):
                temp = (pow(h-1-i,2) + pow(j,2))
                yBL = i
                xBL = j

    # find top-right intersection
    temp = 0
    for i in reversed(range(int(h/2))):
        for j in range(int(w/2),w):
            if intersections[i,j] == 1 and temp < (pow(i,2) + pow (w-1-j,2)):
                temp = (pow(i,2) + pow(w-1-j,2))
                yTR = i
                xTR = j

    # find bottom-right intersection
    temp = 0
    for i in range(int(h/2),h):
        for j in range(int(w/2),w):
            if intersections[i,j] == 1 and temp < (pow(h-1-i,2) + pow (w-1-j,2)):
                temp = (pow(h-1-i,2) + pow(w-1-j,2))
                yBR = i
                xBR = j

    left = max(xTL,xBL) + 5
    right = min(xTR,xBR) - 5

    top = max(yTL,yTR) + 5
    bottom = min(yBL,yBR) - 5

    imgOutput = img[top:bottom,left:right]
    imgOutputThresh = imgThresh[top:bottom,left:right]
    
    return imgOutput, imgOutputThresh
# end function


def cvHoughLine(img):
    print()
#end function
