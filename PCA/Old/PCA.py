import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

#############################################################################################################
def myPCA(K):
    # filename structure
    path = 'Classified/' # path to the database
    #digits = range(0, 10) # 10 digits
    digits = range(0, 2)
    surfix = '.txt'
    # data dimension
    h, w = 106, 56 # hight, weight
    nLines = sum( 1 for line in open(path+"0"+surfix)) # number of lines in a file data
    D = h * w
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
    
    #np.savetxt("Data/dataOriginal."+str(K)+"v2.txt",X)

    # compute mean
    X_Transpose = X.T
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
    
    np.savetxt("Data/originalData."+str(K)+"v2.txt",X)
    np.savetxt("Data/data."+str(K)+"v2.txt",X_Projected)
    np.savetxt("Data/mean."+str(K)+"v2.txt",mean)
    np.savetxt("Data/principalComponent."+str(K)+"v2.txt",Uk)
    return [X_Projected,Uk,mean] 

# end function
#############################################################################################################
def regenerateData(pcaData,Uk,dataMean):
    h,w = 106,56
    D = h*w
    regeneratedData = Uk.dot(pcaData)
    
    for i in range(D):
        regeneratedData[i,:] = regeneratedData[i,:] + dataMean[i]

    regeneratedData = regeneratedData.T
    np.savetxt("Data/regeneratedData.10v2.txt",regeneratedData)
    return regeneratedData
# end function

def display():
    fileOriginal = open("Data/originalData.10v2.txt","r")
    fileRegenerated = open("Data/regeneratedData.10v2.txt","r")
    Original = [line for line in fileOriginal]
    Regenerated = [line for line in fileRegenerated]
    for i in range(0,180*2,18):
        Ori = Original[i].split(" ")
        Reg = Regenerated[i].split(" ")
        imgOri = np.zeros((106,56),np.uint8)
        imgReg = np.zeros((106,56),np.uint8)
        for k in range(106):
            for l in range(56):
                imgOri[k,l] = np.uint8(float(Ori[k*56+l]))
                imgReg[k,l] = np.uint8(float(Reg[k*56+l]))
        cv.imshow("imgOri",imgOri)
        cv.imshow("imgReg",imgReg)

        cv.waitKey(0)
    return



def main():
    print("__PCA__")
    [pcaData,Uk,dataMean] = myPCA(K=10)
    regenerateData(pcaData,Uk,dataMean)

    display()

    return

# end main

if __name__ == "__main__":
    main()
