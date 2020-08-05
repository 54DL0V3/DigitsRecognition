import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

#############################################################################################################
def myPCA():
    # filename structure
    path = 'Classified/' # path to the database
    digits = range(0, 10) # 10 digits
    surfix = '.txt'
    # data dimension
    h, w, K = 56, 106, 1000 # hight, weight, new dim
    nLines = sum( 1 for line in open(path+"0"+surfix)) # number of lines in a file data
    D = h * w
    N = 10*nLines 
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
    # compute mean
    X_Transpose = X.T
    mean = np.mean(X_Transpose,axis=1) # axis = 1 : mean working along the row
    # standardize data
    for i in range(D):
        X_Transpose[i,:] = X_Transpose[i,:] - mean[i]
    # find Covariance matrix
    corvarianceMatrix = 1/D*X_Transpose.T.dot(X_Transpose)
    # compute the eigenvalues and right eigenvectors of Covariance matrix
    # w - eigenvalues
    # v - normalized eigenvectors
    w,v = np.linalg.eig(corvarianceMatrix)
    # sort eigenvectors by descending eigenvalues
    wv = list(zip(w,v))
    wv = sorted(wv, key=lambda a :a[0], reverse=True)
    # get eigenvectors which eigenvalues are in top K biggest values
    kV = [v for _,v in wv[:K]]
    # build matrix Uk which colums form the orthogonal system, these are also known as Principal Component
    # form a subspace close to the standardize data
    Uk = np.array(kV).T
    # project standardize data to the subspace
    X_Projected = X_Transpose.dot(Uk)
    return [X_Projected,Uk] 

# end function

def main():
    print("__PCA__")
    myPCA()
    return

# end main

if __name__ == "__main__":
    main()
