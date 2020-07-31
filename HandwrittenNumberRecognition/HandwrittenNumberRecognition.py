import GetSamples
import cv2 as cv
import os
import numpy as np
import Preprocess
import myMath

inputPath = 'RawSamples'
outputFolder = 'Samples'
classifiedFolder = 'Classified'

padding = 3
height = 100 + padding*2
width = 50 + padding*2

def Classify(showSteps):

    print("Loading...")
    for folder in os.listdir("Samples"):
        file = open(classifiedFolder+"/"+folder+".txt","w")
        for list in os.listdir("Samples/"+folder):
            for sample in os.listdir("Samples/"+folder+"/"+list):
                imgDir = "Samples/"+folder+"/"+list+"/"+sample
                print(imgDir)

                img = cv.imread(imgDir)
                imgGrayscale, imgThresh = Preprocess.preprocess(img)
                imgThin = Preprocess.makeThin(imgThresh)
                imgStd = Preprocess.standardize(imgThin,padding)
                imgFat = Preprocess.makeFat(imgStd,padding)

                if showSteps:
                    cv.imshow("img",img)
                    cv.imshow("imgThresh",imgThresh)
                    cv.imshow("imgThin", imgThin)
                    cv.imshow("imgStd",imgStd)
                    cv.imshow("imgFat", imgFat)
                    cv.waitKey(1)

                for k in range(height):
                        for l in range(width):
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


def main():
    #GetSamples.getSamples("RawSamples","Samples",False,1)

    #Classify(True)

    Test("TestSamples/10.jpg",5)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return
#end function

if __name__ == '__main__':
    main()