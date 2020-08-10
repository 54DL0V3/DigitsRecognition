import cv2 as cv
import os
import numpy as np
from scipy import misc
from sklearn.decomposition import PCA

import Preprocess
import myMath
import GetSamples
import myHoughLine

np.random.seed(1)

inputPath = 'RawSamples'
outputFolder = 'Samples'
classifiedFolder = 'Classified'

padding = 3
height = 100 + padding*2
width = 50 + padding*2

def Classify(showSteps):

    outputImgHeight = int(height/2)
    outputImgWidth = int(width/2)

    print("Loading...")
    for folder in os.listdir("Samples"):
        file = open(classifiedFolder+"/"+str(outputImgHeight)+'x'+str(outputImgWidth)+'.'+ folder+".txt","w")
        for list in os.listdir("Samples/"+folder):
            for sample in os.listdir("Samples/"+folder+"/"+list):
                imgDir = "Samples/"+folder+"/"+list+"/"+sample
                print(imgDir)
                img = cv.imread(imgDir)
                imgGrayscale, imgThresh = Preprocess.preprocess(img)
                imgThin = Preprocess.makeThin(imgThresh)
                imgStd = Preprocess.standardize(imgThin,padding)
                imgFat = Preprocess.makeFat(imgStd,padding)
                imgFat = cv.resize(imgFat,dsize=(outputImgWidth,outputImgHeight),interpolation=cv.INTER_AREA)
                if showSteps:
                    cv.imshow("img",img)
                    cv.imshow("imgThresh",imgThresh)
                    cv.imshow("imgThin", imgThin)
                    cv.imshow("imgStd",imgStd)
                    cv.imshow("imgFat", imgFat)
                    cv.waitKey(1)
                # resize output
                for k in range(outputImgHeight):
                        for l in range(outputImgWidth):
                            file.write(str(imgFat[k,l])+" ")
                file.write("\n")
        file.close()
    print("Done")
    return
#end function

def Test(imgDir,n):
    imgInput = cv.imread(imgDir)

    (h,w,d) = imgInput.shape
    if h>212 or w>166:
        imgInput = cv.resize(imgInput,dsize=(166,212),interpolation = cv.INTER_AREA)
    cv.imshow("imgInput",imgInput)
    cv.waitKey(1)

    print("Preprocessing...")
    imgGrayscale, imgThresh = Preprocess.preprocess(imgInput)
    imgThin = Preprocess.makeThin(imgThresh)
    imgStd = Preprocess.standardize(imgThin,padding)
    img = Preprocess.makeFat(imgStd,padding)
    
    cv.imshow("imgThresh",imgThresh)
    cv.imshow("imgThin",imgThin)
    
    cv.imshow("img",img)
    cv.waitKey(1)
    print("Done")
    print("Comparing...")
    max=0
    for file in os.listdir("Classified"):
        nline = sum(1 for line in open("Classified/"+file))
        if(nline>max): max=nline
    
    comparisonResults = np.zeros((10,max),int)
    for f in os.listdir("Classified"):
        (fileName,fileType) = f.split(".")
        file = open("Classified/"+f,"r")
        lineCount = 0
        for line in file:
            data = line.split()
            for k in range(height):
                for l in range(width):
                    if(int(img[k,l])==int(data[k*width+l])): 
                        comparisonResults[int(fileName),lineCount]+=1
            comparisonResults[int(fileName),lineCount]/=(height*width/100)
            print(fileName + "-" + str(lineCount) +"-"+ str(comparisonResults[int(fileName),lineCount]) + "%")
            lineCount+=1
        file.close()
    print("Done")
    print("Collecting Result")
    kNN = np.zeros((2,n),int)
    for i in range(10):
        for j in range(max):
            for l in range(n):
                if (comparisonResults[i,j]>kNN[0,l]):
                    for k in reversed(range(l+1,n)):
                        kNN[0:1,k]=kNN[0:1,k-1]
                    kNN[0,l]=comparisonResults[i,j]
                    kNN[1,l]=i
                    break

    for i in range(n):
        print(f'kNN[{i}]={kNN[0,i]}%-{kNN[1,i]}')

    result = np.zeros((2,10),int)
    for i in range(n):
        result[0,kNN[1,i]] += 1 
        result[1,kNN[1,i]] += kNN[0,i]

    r=0
    for i in range(1,10):
        if (result[0,i]>result[0,r]):
            r=i
        elif (result[0,i]==result[0,r]) and (result[1,i]>result[1,r]):
            r=i
    print(f'Result:{r}')
    
    
    return
#end function

def runTestcases():

    # filename structure
    path = 'Classified/' # path to the database
    digits = range(0, 10) # 10 digits
    surfix = '.txt'
    # data dimension

    nLines = sum( 1 for line in open(path+"0"+surfix)) # number of lines in a file data
    D = height * width
    N = len(digits)*nLines 
    # collect all data
    X = np.zeros((N, D),np.float)
    cnt = 0
    for digit in digits:
        filePath = path + str(digit) + surfix
        file = open(filePath,'r')
        for line in file:
            temp = line.split(' ')
            for i in range(D):
                X[cnt, i] = np.int(temp[i])
            cnt += 1
        file.close()

    # testing

    # colum = recognize result
    # row = answer
    # testResult[row,col] = amount of results match the answer
    testResult = np.zeros((10,10),np.uint) 

    D = height*width
    count = 0
    for testImg in X:
        imgOri = np.zeros((height,width),np.uint8)
        for k in range(height):
            for l in range(width):
                imgOri[k,l] = np.uint8(float(testImg[k*width+l]))
        cv.imshow("testImg",imgOri)
        cv.waitKey(1)

        # compute distance from vector projected data to each vector in pca data
        listDistances = np.zeros((1800,2),np.float)
        for i in range(1800):
            classifiedData = X[i].copy()
            distance = 0
            for a,b in zip(classifiedData,testImg):
                distance += 1 if a==b else 0 # 1D space
            listDistances[i,1] = distance
            listDistances[i,0] = int(i/180)

        # sort distances by decending values
        listDistances = sorted(listDistances, key=lambda a :a[1], reverse=True)

        # using kNN
        nNearestNeighbors = 11
        kNN = np.zeros((nNearestNeighbors,2))
        for i in range(nNearestNeighbors):
            kNN[i,0] = listDistances[i][0]
            kNN[i,1] = listDistances[i][1]

        temp = np.zeros((2,10),int)
        for i in range(1,nNearestNeighbors):
            temp[0,int(kNN[i,0])] += 1 
            temp[1,int(kNN[i,0])] += kNN[i,1]

        print("---------------------------------------------------------")
        for i in range(nNearestNeighbors):
            print(f'kNN[{i}]={kNN[i,0]}-{kNN[i,1]/5936*100}%')
        r=0
        for i in range(1,10):
            if (temp[0,i]>temp[0,r]):
                r=i
            elif (temp[0,i]==temp[0,r]) and (temp[1,i]>temp[1,r]):
                r=i
        ans = int(count/180)

        print(f'Recognize Result:{r} - {ans}')
        
        testResult[ans,r] +=1
        count+=1
        print("---------------------------------------------------------")
        cv.waitKey(1)

    np.savetxt("testResult.txt",testResult,fmt='%d')
    return


def main():

    print("__DigitsRecognition__")

    #GetSamples.getSamples("RawSamples","Samples",True,1)

    #Classify(False)
    
    #Test(imgDir,11)

    runTestcases()

    cv.waitKey(0)
    cv.destroyAllWindows()
    return
#end function

if __name__ == '__main__':
    main()