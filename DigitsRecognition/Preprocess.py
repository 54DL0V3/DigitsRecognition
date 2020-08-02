# Preprocess.py

import cv2 as cv
import numpy as np
import math
import myMath

# module level variables ##########################################################################
GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5)
ADAPTIVE_THRESH_BLOCK_SIZE = 19
ADAPTIVE_THRESH_WEIGHT = 9

###################################################################################################
def preprocess(imgOriginal):
    imgGrayscale = extractValue(imgOriginal)

    imgMaxContrastGrayscale = maximizeContrast(imgGrayscale)

    height, width = imgGrayscale.shape

    imgBlurred = np.zeros((height, width, 1), np.uint8)

    imgBlurred = cv.GaussianBlur(imgMaxContrastGrayscale, GAUSSIAN_SMOOTH_FILTER_SIZE, 0)

    imgThresh = cv.adaptiveThreshold(imgBlurred, 255.0, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT)

    return imgGrayscale, imgThresh
# end function

###################################################################################################
def extractValue(imgOriginal):
    height, width, numChannels = imgOriginal.shape

    imgHSV = np.zeros((height, width, 3), np.uint8)

    imgHSV = cv.cvtColor(imgOriginal, cv.COLOR_BGR2HSV)

    imgHue, imgSaturation, imgValue = cv.split(imgHSV)

    return imgValue
# end function

###################################################################################################
def maximizeContrast(imgGrayscale):

    height, width = imgGrayscale.shape

    imgTopHat = np.zeros((height, width, 1), np.uint8)
    imgBlackHat = np.zeros((height, width, 1), np.uint8)

    structuringElement = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))

    imgTopHat = cv.morphologyEx(imgGrayscale, cv.MORPH_TOPHAT, structuringElement)
    imgBlackHat = cv.morphologyEx(imgGrayscale, cv.MORPH_BLACKHAT, structuringElement)

    imgGrayscalePlusTopHat = cv.add(imgGrayscale, imgTopHat)
    imgGrayscalePlusTopHatMinusBlackHat = cv.subtract(imgGrayscalePlusTopHat, imgBlackHat)

    return imgGrayscalePlusTopHatMinusBlackHat
# end function

###################################################################################################
#
# input: image Thresh
# output: image Thresh with linesize = 1px
def makeThin(imgThresh):
    
    (h,w) = imgThresh.shape[:2]
    imgThin = np.zeros((h,w),np.uint8)
    for i in range(h):
        j=0
        s=0
        while j < w-1 :
            if imgThresh[i,j]==0 and imgThresh[i,j+1]==255:
                s=j+1
                e=s
                c = True
                for t in range(s,w-1):
                    if imgThresh[i,t]==255 and imgThresh[i,t+1]==0:
                        c = False
                        e=t
                        break
                if c: e = w-1
                imgThin[i,(e-s)//2+s]=255
                j=e+1
            else:
                j+=1

    for j in range(w):
        i=0
        s=0
        while i < h-1 :
            if imgThresh[i,j]==0 and imgThresh[i+1,j]==255:
                s=i+1
                e=s
                c = True
                for t in range(s,h-1):
                    if imgThresh[t,j]==255 and imgThresh[t+1,j]==0:
                        c = False
                        e=t
                        break
                if c: e = h-1
                imgThin[(e-s)//2+s,j]=255
                i=e+1
            else:
                i+=1

  
    l = np.array([1,2,0])
    r = np.array([1,2,0])
    for i in range(h):
        for j in range(w):
            if imgThin[i,j]!=255:
                continue
            temp=imgThin[i-1:i+2,j-1:j+2]
            if temp.sum()==255:
                imgThin[i,j]=0
    return imgThin
# end function

###################################################################################################
#
# input: image Thresh, padding
# output: image Thresh with linesize = (input + pad*2) px
def makeFat(imgThresh,padding):
    blurFilter = np.zeros((padding*2+1,padding*2+1),np.uint8);
    for i in range(padding*2+1):
        for j in range(padding*2+1):
            blurFilter[i,j]=1
    (h,w) = imgThresh.shape[:2]

    imgFat = myMath.convolve(imgThresh,blurFilter)
    for i in range(h):
        for j in range(w):
            imgFat[i,j] = 0 if imgFat[i,j]==0 else 255
            


    
    return imgFat
    
# end function


###################################################################################################
#
# resize the side of image to be standard 
def standardize(imgThresh,padding):
    h = len(imgThresh)
    w = len(imgThresh[0])

    top=h
    bottom = 0
    left = w
    right = 0
    for i in range(h):
        for j in range(w):
            if (imgThresh[i,j]==255):
                if i>bottom: bottom = i
                elif i <top: top = i

                if j > right: right = j
                elif j< left: left = j
    
    img = imgThresh[top:bottom+1,left:right+1]

    img = cv.resize(img,dsize=(50,100),interpolation = cv.INTER_AREA)

    imgOutput = np.zeros((100+padding*2,50+padding*2),np.uint8)

    for i in range(padding,padding+100):
        for j in range(padding, padding+ 50):
            imgOutput[i,j] = img[i-padding,j- padding]

    return imgOutput
    
# end function








