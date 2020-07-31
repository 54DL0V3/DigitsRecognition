def Classify_v1():
    listFiles = os.listdir(inputPath)
    for file in listFiles:
        (fileName,fileType) = file.split(".") 
        for j in range(10):
            imgAvg = np.zeros((height,width), np.uint8)
            imgAvgBlur = np.zeros((height,width), np.uint8)
            for i in range(10):
                imgCalibrated = Calibrate(outputFolder+"/"+str(j)+"/"+fileName+"_"+str(i)+".jpg")

                for k in range(height):
                    for l in range(width):
                        #imgAvg[k,l] += (imgCropped[k,l]/10)
                        imgAvg[k,l] += (imgCalibrated[k,l]/10)
            max = np.uint8(0)
            for k in range(1,height-1):
                for l in range(1,width-1):
                    imgAvgBlur[k,l] = imgAvg[k-1,l-1]/9 + imgAvg[k-1,l]/9 + imgAvg[k-1,l+1]/9 + imgAvg[k,l-1]/9 + imgAvg[k,l]/9 + imgAvg[k,l+1]/9 + imgAvg[k+1,l-1]/9 + imgAvg[k+1,l]/9 + imgAvg[k+1,l+1]/9
                    if (imgAvgBlur[k,l]>max): max = imgAvgBlur[k,l]
            for k in range(1,height-1):
                for l in range(1,width-1):
                    imgAvgBlur[k,l] = imgAvgBlur[k,l]/max*255
            
            cv.imshow(str(j),imgAvgBlur)
            file = open(classifiedFolder+"/"+str(j)+".txt","a")
            
            for k in range(height):
                    for l in range(width):
                        file.write(str(imgAvgBlur[k,l])+" ")

            file.write("\n")    
            file.close()
            #cv.imshow(str(j),imgAvgBlur)
        #cv.waitKey(0)
    cv.destroyAllWindows()
    return
#end function

def Classify_v2():
    print("Loading...")
    for folder in os.listdir("Samples"):
        file = open(classifiedFolder+"/"+folder+".txt","w")
        for list in os.listdir("Samples/"+folder):
            for sample in os.listdir("Samples/"+folder+"/"+list):
                imgDir = "Samples/"+folder+"/"+list+"/"+sample
                print(imgDir)
                img = cv.imread(imgDir)
                img = Calibrate(img)
                for k in range(height):
                        for l in range(width):
                            file.write(str(img[k,l])+" ")
                file.write("\n")
        file.close()
    print("Done")
    return
#end function

def Classify():
    print("Loading...")
    for folder in os.listdir("Samples"):
        file = open(classifiedFolder+"/"+folder+".txt","w")
        for list in os.listdir("Samples/"+folder):
            imgSum = np.zeros((height,width),np.uint8)
            for sample in os.listdir("Samples/"+folder+"/"+list):
                imgDir = "Samples/"+folder+"/"+list+"/"+sample
                print(imgDir)
                img = cv.imread(imgDir)
                img = Calibrate(img)
                for k in range(height):
                    for l in range(width):
                        if (imgSum[k,l]!=255) and (img[k,l]==255): imgSum[k,l]=255
            for k in range(height):
                    for l in range(width):
                        file.write(str(imgSum[k,l])+" ")
            file.write("\n")
        file.close()
    print("Done")
    return
#end functiondef Classify():
    print("Loading...")
    for folder in os.listdir("Samples"):
        file = open(classifiedFolder+"/"+folder+".txt","w")
        for list in os.listdir("Samples/"+folder):
            imgSum = np.zeros((height,width),np.uint8)
            for sample in os.listdir("Samples/"+folder+"/"+list):
                imgDir = "Samples/"+folder+"/"+list+"/"+sample
                print(imgDir)
                img = cv.imread(imgDir)
                img = Calibrate(img)
                for k in range(height):
                    for l in range(width):
                        if (imgSum[k,l]!=255) and (img[k,l]==255): imgSum[k,l]=255
            for k in range(height):
                    for l in range(width):
                        file.write(str(imgSum[k,l])+" ")
            file.write("\n")
        file.close()
    print("Done")
    return
#end function