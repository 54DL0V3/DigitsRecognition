import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

height = 53
width = 28
MAX_K = height*width

####################################################################################################

def myPCA(K):
    print("Doing PCA - K="+str(K))
    # filename structure
    path = 'Classified/' # path to the database
    digits = range(0, 10) # 10 digits
    surfix = '.txt'
    # data dimension

    nLines = sum( 1 for line in open(path+str(height)+"x"+str(width)+"."+"0"+surfix)) # number of lines in a file data
    D = height * width
    N = len(digits)*nLines 
    # collect all data
    X = np.zeros((N, D),np.float)
    cnt = 0
    for digit in digits:
        filePath = path + str(height) + "x" + str(width) + "." + str(digit) + surfix
        file = open(filePath, 'r')
        for line in file:
            temp = line.split(' ')
            for i in range(D):
                X[cnt, i] = np.int(temp[i])
            cnt += 1
        file.close()

    # compute mean
    X_Transpose = X.T.copy(order="K")
    mean = np.zeros(D,np.float)
    for i in range(D):
        mean[i] = 1/N*sum(X_Transpose[i,:])
    #mean = np.mean(X_Transpose,axis=1)
    
    # standardize data
    for i in range(D):
        X_Transpose[i,:] = X_Transpose[i,:] - mean[i]
    # find Covariance matrix

    corvarianceMatrix = 1/D*X_Transpose.dot(X_Transpose.T)
    # compute the eigenvalues and right eigenvectors of Covariance matrix
    # w - eigenvalues
    # v - normalized eigenvectors
    w,v = np.linalg.eig(corvarianceMatrix)
    # sort eigenvectors by descending eigenvalues
    wv = list(zip(w,v))
    wv = sorted(wv, key=lambda a :a[0], reverse=True)
    # get eigenvectors which eigenvalues are in top K biggest values
    kV = [v for _,v in wv[:K]]
    for i in range(len(kV)):
        kV[i]=kV[i].real

    # build matrix Uk which colums form the orthogonal system, these are also known as Principal Component
    # form a subspace close to the standardize data
    Uk = np.array(kV).T
    # project standardize data to the subspace
    X_Projected = Uk.T.dot(X_Transpose)
    
    
    np.savetxt("Data/originalData."+ str(height) + "x" + str(width) + "." + str(K)+".txt", X)
    np.savetxt("Data/data."+ str(height) + "x" + str(width) + "." + str(K)+".txt", X_Projected)
    np.savetxt("Data/mean."+ str(height) + "x" + str(width) + "." + str(K)+".txt", mean)
    np.savetxt("Data/principalComponent."+ str(height) + "x" + str(width) + "." + str(K)+".txt", Uk)
    return [X, X_Projected, Uk, mean] 

# end function
####################################################################################################
def regenerateData(pcaData, Uk, dataMean):
    D = height*width
    K = len(Uk[0])
    regeneratedData = Uk.dot(pcaData)
    
    for i in range(D):
        regeneratedData[i, :] = regeneratedData[i, :] + dataMean[i]

    regeneratedData = regeneratedData.T
    np.savetxt("Data/regeneratedData."+ str(height) + "x" + str(width) + "." + str(K)+".txt"
        ,regeneratedData)
    return regeneratedData
# end function

def display(K):
    fileOriginal = open("Data/originalData."+str(K)+".txt", "r")
    fileRegenerated = open("Data/regeneratedData."+str(K)+".txt", "r")
    Original = [line for line in fileOriginal]
    Regenerated = [line for line in fileRegenerated]
    nLines = 1800-180
    for i in range(0, nLines, 18):
        Ori = Original[i].split(" ")
        Reg = Regenerated[i].split(" ")
        minValue = 255
        maxValue = 0
        for j in range(len(Reg)):
            Reg[j] = float(Reg[j])
        for j in range(len(Reg)):
            if (Reg[j] > maxValue): maxValue = Reg[j]
            if (Reg[j] < minValue): minValue = Reg[j]
        for j in range(len(Reg)):
            temp = Reg[j]
            Reg[j] = np.uint8((temp-minValue)/(maxValue-minValue)*255)



        imgOri = np.zeros((height, width), np.uint8)
        imgReg = np.zeros((height, width), np.uint8)
        for k in range(height):
            for l in range(width):
                imgOri[k, l] = np.uint8(float(Ori[k*width+l]))
                imgReg[k, l] = Reg[k*width+l]
        cv.imshow("imgOri", imgOri)
        cv.imshow("imgReg", imgReg)

        cv.waitKey(0)
    return

def runTestcases(K):

    # get necessary data
    try:
        # open files contain data
        fMean = open("Data/mean."+ str(height) + "x" + str(width) + "." + str(K)+".txt", "r")
        fPcaData = open("Data/data."+ str(height) + "x" + str(width) + "." + str(K)+".txt", "r")
        fUk = open("Data/principalComponent." + str(height) + "x" + str(width) + "." + str(K)+".txt", "r")
        fOri = open("Data/originalData."+ str(height) + "x" + str(width) + "." + str(K)+".txt", "r")
        # read data from file
        try:
            mean = np.loadtxt(fMean, dtype=np.float)
            pcaData = np.loadtxt(fPcaData, dtype=np.float)
            Uk = np.loadtxt(fUk, dtype=np.float)
            X = np.loadtxt(fOri, dtype=np.float)
        except:
            print("Load data failed.")
        # close files    
        fMean.close() 
        fPcaData.close()
        fUk.close()
    except IOError:
        # if any file is not exist, do pca
        [X, pcaData, Uk, mean] = myPCA(K=500)
    
    pcaData = pcaData.T
    # testing

    # colum = recognize result
    # row = answer
    # testResult[row,col] = amount of results match the answer
    testResult = np.zeros((10, 10), np.uint) 

    D = height*width
    count = 0
    for testImg in X:
        imgOri = np.zeros((height, width), np.uint8)
        for k in range(height):
            for l in range(width):
                imgOri[k, l] = np.uint8(float(testImg[k*width+l]))
        cv.imshow("testImg", imgOri)
        cv.waitKey(1)

        # standardize test data
        testData = testImg.copy()
        for i in range(D):
            testData[i] -= mean[i]
        # project standardize data to the subspace
        test_Projected = Uk.T.dot(testData.T)
        # compute distance from vector projected data to each vector in pca data
        listDistances = np.zeros((1800, 2), np.float)
        for i in range(1800):
            pca = pcaData[i].copy()
            distance = 0
            for a,b in zip(pca, test_Projected):
                distance += abs(a-b) # 1D space
            listDistances[i, 1] = distance
            listDistances[i, 0] = int(i/180)

        # sort distances by decending values
        listDistances = sorted(listDistances, key=lambda a: a[1], reverse=False)

        # using kNN
        nNearestNeighbors= 11
        kNN = np.zeros((nNearestNeighbors ,2))
        for i in range(nNearestNeighbors):
            kNN[i,0] = listDistances[i][0]
            kNN[i,1] = listDistances[i][1]

        temp = np.zeros((2, 10), int)
        for i in range(1, nNearestNeighbors):
            temp[0,int(kNN[i, 0])] += 1 
            temp[1,int(kNN[i, 0])] += kNN[i, 1]

        print("---------------------------------------------------------")
        for i in range(nNearestNeighbors):
            print(f'kNN[{i}]={kNN[i,0]}-{kNN[i,1]}')
        r=0
        for i in range(1, 10):
            if( temp[0,i]>temp[0,r]):
                r= i
            elif( temp[0, i]==temp[0, r]) and (temp[1, i]>temp[1, r]):
                r= i
        ans = int(count/180)

        print(f'Recognize Result:{r} - {ans}')
        
        testResult[ans, r]+= 1
        count+= 1
        print("---------------------------------------------------------")
        cv.waitKey(1)

    np.savetxt("Data/testResult."+ str(height) + "x" + str(width) + "." + str(K)+".txt", testResult, fmt='%d')
    return

def main():
    print("__PCA__")

    #[_,pcaData,Uk,dataMean] = myPCA(K=100)
    #regenerateData(pcaData,Uk,dataMean)
    #display(K=500)

    # runTestcases(K=1000)
    # runTestcases(K=1484)
    # runTestcases(K=250)
    runTestcases(K=10)

    return

# end main

if __name__ == "__main__":
    main()
