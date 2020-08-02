# myMath.py
import os
import numpy as np
import cv2 as cv


TEST_KERNEL = np.array([[0,-1,0],
                        [-1,4,-1],
                        [0,-1,0]])

BOX_BLUR = np.array([[1,1,1],
                    [1,1,1],
                    [1,1,1]])

GAUSSIAN_BLUR_3x3 = np.array([[1,2,1],
                            [2,4,2],
                            [1,2,1]])

EDGE_V1 = np.array([[0,-1,0],
                    [-1,4,-1],
                    [0,-1,0]])

EDGE_V2 = np.array([[-1,-1,-1],
                    [-1,8,-1],
                    [-1,-1,-1]])

###################################################################################################
# 
def convolve(image, kernel):
    (iH,iW) = image.shape[:2]
    (kH,kW) = kernel.shape[:2]

    pad = (kW - 1)//2
    imageWithPad = np.zeros((iH+2*pad,iW+2*pad),np.uint32)
    for i in range(pad,pad+iH):
        for j in range(pad,pad+iW):
            imageWithPad[i,j] = image[i-pad,j-pad]
    output = np.zeros((iH,iW),np.float32)

    for i in range(pad,iH+pad):
        for j in range(pad,iW+pad):
            top = i - pad
            bottom = i + pad +1
            left = j - pad
            right = j+pad +1
            roi = imageWithPad[top:bottom, left:right]
            try:
                k = (roi*kernel).sum()
            except:
                #print("Warning: operands could not be broadcast together.")
                continue

            output[i-pad,j-pad] = k
    
    # rescale output image to be in range [0,255]
    temp = kernel.sum()
    if temp !=0:
        for i in range(iH):
            for j in range(iW):
                output[i,j] = int(output[i,j]/temp)
    else:
        for i in range(iH):
            for j in range(iW):
                output[i,j] = int(output[i,j])

    output = rescale(output)
    output = output.astype("uint8")

    return output
# end function

###################################################################################################
#
# rescale image to be in range [0,255]
def rescale(input):
    (iH,iW) = input.shape[:2]
    max = input[0,0]
    min = input[0,0]

    for i in range(iH):
        for j in range(iW):
            if input[i,j] > max : max = input[i,j]
            if input[i,j] < min : 
                min = input[i,j]
    for i in range(iH):
        for j in range(iW):
            input[i,j] = int((input[i,j]-min)/(max-min)*255)
    return input
# end function