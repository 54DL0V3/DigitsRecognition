import cv2 as cv
import os
import numpy as np
import Preprocess
import myHoughLine


def getSamples(inputPath,outputFolder,showSteps,mode):
    # mode = 1: get training sample
    # mode = 0: get test sample
    print("Getting samples ...")
    listFiles = os.listdir(inputPath)
    for file in listFiles:
        print(file)
        getSamplesFromFile(file,inputPath,outputFolder,showSteps,mode)
    print("Done")
    return
#end function


def getSamplesFromFile(file,inputPath,outputFolder,showSteps, mode):

    y_const = 40
    x_const = 90


    (fileName,fileType) = file.split(".") 
    img = cv.imread(inputPath+ "/" + fileName +"."+ fileType)

    img = cv.resize(img,dsize=(2100,1480),interpolation=cv.INTER_AREA)

    (h,w,d) = img.shape

    imgGrayscale, imgThresh = Preprocess.preprocess(img)

    x_axis = np.zeros(w,int)
    y_axis = np.zeros(h,int)
    yBlur = np.zeros((h,w), np.uint8)

    for i in range(h):
        for j in range(w):
            px = imgThresh[i,j]
            y_axis[i] += px
        y_axis[i] /= w
        y_axis[i] = 0 if (y_axis[i]<y_const) else 255

    if (showSteps):
        for i in range(h):
            for j in range(w):
                yBlur[i,j] = y_axis[i]
    
        cv.imshow("img",imgThresh)
        cv.imshow("yBlur",yBlur)
        cv.waitKey(1)
    ##########################################################################################
    y = np.zeros(h,int)
    i = 0
    while (i < h):
        if (y_axis[i] == 255):
            for j in range(i+1,h):
                if (y_axis[j] == 0): break
            t = int((i+j-1)/2)
            y[t] = 1
            i=j+1
        else:
            i+=1

    #Get avg row height
    for i in range(h):
        if(y[i]==1):
            break

    for j in reversed(range(h)):
        if(y[j]==1): break

    avgRowHeight = (j-i)/9

    #get the end point of the last row
    i=h-1
    while(i>=0):
        if y[i] == 1: break
        else: i-=1
    
    j=i-1
    while(j>=0):
        if (y[j]==1):
            if(i-j<avgRowHeight*0.8): y[j]=0
            else: i=j
            j-=1
        else:
            j-=1

    # get the start point of the first row
    for i in range(h):
        if y[i] == 1: break

    count = 0
    count2 = 0
    j=i+1
    while (j<h):
        if y[j]==1:
            rowThresh = imgThresh[i:j,0:w]
            row = img[i:j,0:w]
            rowHeight = j-i
            for k in range(w):
                for l in range(rowHeight): 
                    px = rowThresh[l,k]
                    x_axis[k] += px
                x_axis[k] /= (rowHeight) 
                x_axis[k] = 0 if (x_axis[k] < x_const) else 255
            
            if(showSteps): 
                xblur = np.zeros((rowHeight,w), np.uint8)
                for k in range(w):
                    for l in range(rowHeight):
                        xblur[l,k]=x_axis[k]
                cv.imshow("xblur",xblur)
                cv.imshow("row",row)
                cv.waitKey(1)

            x = np.zeros(w,int)
            k = 0
            while (k < w):
                if (x_axis[k] == 255):
                    for l in range(k+1,w):
                        if (x_axis[l] == 0): break
                    t = int((k+l-1)/2)
                    x[t] = 1
                    k=l+1
                else:
                    k+=1
        
            #Delete the first x[k]==1
            for k in range(w): 
                if(x[k]==1): 
                    x[k]=0
                    break

            #Get avg box width
            for k in range(w):
                if(x[k]==1):
                    break

            for l in reversed(range(0,w)):
                if(x[l]==1): break

            avgBoxWidth=(l-k)/10

            count1=0
            l=k+1
            while (l<w):
                if(x[l]==1):
                    if ((l-k)<avgBoxWidth*0.9): x[l]=0
                    else:
                        box = np.zeros((rowHeight, l-k, d), np.uint8)
                        padding = 10
                        left = 0 if (k - padding) < 0 else k - padding
                        right = w if (l + padding) > w-1 else l + padding
                        top = 0 if (i - padding) < 0 else i - padding
                        bottom = h-1 if (j + padding) >= h else j + padding

                        boxThresh = imgThresh[top:bottom,left:right]
                        boxImg = img[top:bottom,left:right]

                        # use HoughLine Transform
                        box, boxThresh = myHoughLine.myHoughLine(boxThresh,boxImg,100,120,showSteps)
                        ####################################################################################

                        #if (showSteps):
                        if True:
                            cv.imshow("box",box)
                            cv.imshow("boxThresh",boxThresh)
                            cv.waitKey(1)
                        # save sample
                        #if (mode==1):
                        #    dir = outputFolder+"/"+str(count)+"/"+fileName
                        #    if (os.path.exists(dir)):
                        #        pass
                        #    else:
                        #        os.mkdir(dir)
                        #    cv.imwrite(dir+"/"+str(count1)+".jpg",box)
                        #elif (mode==0):
                        #    cv.imwrite(outputFolder+"/"+str(count2)+".jpg",box)

                        k=l
                        count1+=1
                        count2+=1
                    l+=1
                else:
                    l+=1

            i=j
            j+=1
            count +=1
        else: j+=1


    return
#end function

