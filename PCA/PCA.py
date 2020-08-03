import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


import Preprocess

np.random.seed(1)

#############################################################################################################
def example():
    # filename structure
    path = 'YALE/unpadded/' # path to the database
    ids = range(1, 16) # 15 persons
    states = ['centerlight', 'glasses', 'happy', 'leftlight',
            'noglasses', 'normal', 'rightlight','sad',
            'sleepy', 'surprised', 'wink' ]
    prefix = 'subject'
    surfix = '.pgm'
    # data dimension
    h, w, K = 116, 98, 100 # hight, weight, new dim
    D = h * w
    N = len(states)*15
    # collect all data
    X = np.zeros((D, N))
    cnt = 0
    for person_id in range(1, 16):
        for state in states:
            fn = path + prefix + str(person_id).zfill(2) + '.' + state + surfix
            img,_=Preprocess.preprocess(cv.imread(fn))
            X[:, cnt] = img.reshape(D)
            cnt += 1
    # Doing PCA, note that each row is a datapoint
    from sklearn.decomposition import PCA
    pca = PCA(n_components=K) # K = 100
    pca.fit(X.T)

    # projection matrix
    U = pca.components_.T

    for i in range(U.shape[1]):
        plt.axis('off')
        f1 = plt.imshow(U[:, i].reshape(116, 98), interpolation='nearest')
        f1.axes.get_xaxis().set_visible(False)
        f1.axes.get_yaxis().set_visible(False)
    # f2 = plt.imshow(, interpolation='nearest' )
        plt.gray()
        fn = 'eigenface' + str(i).zfill(2) + '.png'
    # plt.savefig(fn, bbox_inches='tight', pad_inches=0)
        #plt.show()

    # See reconstruction of first 6 persons 
    for person_id in range(1, 7):
        for state in ['centerlight']:
            fn = path + prefix + str(person_id).zfill(2) + '.' + state + surfix
            im = cv.imread(fn)
            plt.axis('off')
    # plt.imshow(im, interpolation='nearest' )
            f1 = plt.imshow(im, interpolation='nearest')
            f1.axes.get_xaxis().set_visible(False)
            f1.axes.get_yaxis().set_visible(False)
            plt.gray()
            fn = 'ori' + str(person_id).zfill(2) + '.png'
            plt.savefig(fn, bbox_inches='tight', pad_inches=0)
            plt.show()
            # reshape and subtract mean, don't forget 
            im,_= Preprocess.preprocess(im)
            x = im.reshape(D, 1) - pca.mean_.reshape(D, 1)
            # encode
            z = U.T.dot(x)
            #decode
            x_tilde = U.dot(z) + pca.mean_.reshape(D, 1)

            # reshape to orginal dim
            im_tilde = x_tilde.reshape(116, 98)
            plt.axis('off')
    # plt.imshow(im_tilde, interpolation='nearest' )
            f1 = plt.imshow(im_tilde, interpolation='nearest')
            f1.axes.get_xaxis().set_visible(False)
            f1.axes.get_yaxis().set_visible(False)
            plt.gray()
            fn = 'res' + str(person_id).zfill(2) + '.png'
    # plt.savefig(fn, bbox_inches='tight', pad_inches=0)
            plt.show()
    return 
# end function


#############################################################################################################
def myPCA():
    # filename structure
    path = 'Samples/' # path to the database
    numbers = range(0, 10) # 10 digits
    states = ['img0', 'img1', 'img2', 'img3','img4', 'img5',
            'img6','img7','img8', 'img9', 'img10','img11',
            'img12', 'img13', 'img14', 'img15', 'img16','img17']
    ids = range(0,10) #
    surfix = '.jpg'
    # data dimension
    h, w, K = 120, 160, 1000 # hight, weight, new dim
    D = h * w
    N = 10*len(states)*10
    # collect all data
    X = np.zeros((D, N))
    cnt = 0
    for number in numbers:
        for state in states:
            for id in ids:
                fn = path + str(number) +'/' + state +'/'+ str(id)+ surfix
                img,_=Preprocess.preprocess(cv.imread(fn))
                img = cv.resize(img,dsize=(w,h))
                X[:, cnt] = img.reshape(D)
                cnt += 1
    # Doing PCA, note that each row is a datapoint
    from sklearn.decomposition import PCA
    pca = PCA(n_components=K) # K = 1000
    pca.fit(X.T)

    # projection matrix
    U = pca.components_.T

    for i in range(U.shape[1]):
        plt.axis('off')
        f1 = plt.imshow(U[:, i].reshape(h, w), interpolation='nearest')
        f1.axes.get_xaxis().set_visible(False)
        f1.axes.get_yaxis().set_visible(False)
    # f2 = plt.imshow(, interpolation='nearest' )
        plt.gray()
        fn = 'eigenface' + str(i).zfill(2) + '.png'
    # plt.savefig(fn, bbox_inches='tight', pad_inches=0)
        #plt.show()

    # See reconstruction of first 6 persons 
    for number in numbers:
        for state in states:
            for id in [1]:
                fn = path + str(number) +'/' + state +'/'+ str(id)+ surfix
                im = cv.imread(fn)
                plt.axis('off')
    # plt.imshow(im, interpolation='nearest' )
                f1 = plt.imshow(im, interpolation='nearest')
                f1.axes.get_xaxis().set_visible(False)
                f1.axes.get_yaxis().set_visible(False)
                plt.gray()
                fn = 'ori' + str(number).zfill(2)+ str(state).zfill(2) + '.png'
                plt.savefig(fn, bbox_inches='tight', pad_inches=0)
                plt.show()
                # reshape and subtract mean, don't forget 
                im,_= Preprocess.preprocess(im)
                im = cv.resize(img,dsize=(w,h))
                x = im.reshape(D, 1) - pca.mean_.reshape(D, 1)
                # encode
                z = U.T.dot(x)
                #decode
                x_tilde = U.dot(z) + pca.mean_.reshape(D, 1)

                # reshape to orginal dim
                im_tilde = x_tilde.reshape(h, w)
                plt.axis('off')
        # plt.imshow(im_tilde, interpolation='nearest' )
                f1 = plt.imshow(im_tilde, interpolation='nearest')
                f1.axes.get_xaxis().set_visible(False)
                f1.axes.get_yaxis().set_visible(False)
                plt.gray()
                fn = 'res' + str(number).zfill(2)+ str(state).zfill(2) + '.png'
        # plt.savefig(fn, bbox_inches='tight', pad_inches=0)
                plt.show()
    return

# end function

def main():
    print("__PCA__")
    myPCA()
    return

# end main

if __name__ == "__main__":
    main()
