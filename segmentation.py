import cv2
import numpy as np
import os
import scipy.ndimage



def getMeanArea(contours):
    meanArea=0
    for contour in contours:
        meanArea+=cv2.contourArea(contour)
    meanArea=(meanArea)/len(contours)
    return meanArea
    
    
def getRatioArea(contours):
    meanArea=0
    for contour in contours:
        meanArea+=cv2.contourArea(contour)
    cnsSorted = sorted(contours, key=lambda x:cv2.contourArea(x), reverse = True)
    ratioArea = cv2.contourArea(cnsSorted[0])/meanArea
    return ratioArea


def purify(img):
    img=cv2.copyMakeBorder(img,32,32,32,32,cv2.BORDER_CONSTANT)
    #cv2.imshow('img',img)
    #cv2.waitKey(0)
    #img=cv2.bitwise_not(img)
    #kernel=np.ones((3,3),np.uint8)
    #cv2.dilate(img,kernel,iterations=5)
    #cv2.erode(img,kernel,iterations=5)
    #img=cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    meanArea=getMeanArea(contours)
    nlabels,labels,stats,centroids=cv2.connectedComponentsWithStats(img,None,None,None,8,cv2.CV_32S)
    areas=stats[1:,cv2.CC_STAT_AREA]
    result=np.zeros((labels.shape),np.uint8)
    for i in range(nlabels-1):
        if areas[i]>=0.1*meanArea:
            result[labels==i+1]=255
    high=max(result.shape[0],result.shape[1])
    if high==result.shape[0]:
        dif=(high-result.shape[1])//2
        result=cv2.copyMakeBorder(result,0,0,dif,dif,cv2.BORDER_CONSTANT,value=0)
    else:
        dif=(high-result.shape[1])//2
        result=cv2.copyMakeBorder(result,dif,dif,0,0,cv2.BORDER_CONSTANT,value=0)
    #cv2.imshow('result',result)
    #print(result.shape)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return cv2.resize(result,(28,28),interpolation=cv2.INTER_AREA)


def extract_character(image, recursion = 0):
    thresh = cv2.copyMakeBorder(image, 8, 8, 8, 8, cv2.BORDER_REPLICATE)
    pad=5
    thresh=cv2.GaussianBlur(thresh, (3,3), 0)
    #thresh=cv2.medianBlur(image,3)
    #thresh = cv2.adaptiveThreshold(thresh, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, blockSize = 321, C = 28)
    ret,thresh=cv2.threshold(thresh,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    #cv2.imshow('thresh2',thresh)
    #cv2.waitKey(0)
    #cv2.imshow('Thresh',thresh)
    kernel1 = np.ones((3,3), np.uint8)
    thresh = cv2.dilate(thresh, kernel1, iterations = 1)
    #thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel3)
    if(recursion<2):
    	thresh2 = cv2.erode(thresh, np.ones((2,2), np.uint8), iterations = 2)
    	thresh2 = scipy.ndimage.median_filter(thresh2, (5, 1)) # remove line noise
    	thresh2 = scipy.ndimage.median_filter(thresh2, (1, 5)) # weaken circle noise
    	thresh2 = scipy.ndimage.median_filter(thresh2, (5, 1)) # remove line noise
    	thresh2 = scipy.ndimage.median_filter(thresh2, (1, 5)) # weaken circle noise
    	contours1, hierarchy = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
    	contours1, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    coords=[]
    count=0
    ratioArea = getRatioArea(contours1)
    #print(ratioArea)
    if(ratioArea<0.3 or recursion>1):
    	kernel2 = np.ones((2,2), np.uint8)
    elif(ratioArea>0.85 and recursion<1):
    	kernel2 = np.ones((5,5), np.uint8)
    else:
    	kernel2 = np.ones((3,3), np.uint8)
    if(ratioArea > 0.3 and recursion<2):
    	thresh = cv2.erode(thresh, kernel2, iterations = 2)
    	thresh = scipy.ndimage.median_filter(thresh, (5, 1)) # remove line noise
    	thresh = scipy.ndimage.median_filter(thresh, (1, 5)) # weaken circle noise
    	thresh = scipy.ndimage.median_filter(thresh, (5, 1)) # remove line noise
    	thresh = scipy.ndimage.median_filter(thresh, (1, 5)) # weaken circle noise
    thresh = cv2.dilate(thresh, kernel1, iterations = 1)
    #cv2.imshow('thresh',thresh)
    #cv2.waitKey(0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    coords=[]
    count=0
    meanArea=getMeanArea(contours)
    for contour in contours:
        (x,y,w,h)=cv2.boundingRect(contour)
        if cv2.contourArea(contour)>0.05*meanArea:
            if w / h > 1.25:
                #Split it in half into two letter regions
                half_width = int(w / 2)
                coords.append((x, y, half_width, h))
                coords.append((x + half_width, y, half_width, h))
                count=count+2
            else:  
                coords.append((x, y, w, h))
                count=count+1
    coords=sorted(coords,key=lambda x: x[0])
    img_paths=[]
    if(count >7 and recursion <3):
    	img_paths_array = extract_character(image, recursion + 1)
    	return img_paths_array
    else:
    	for i in range(count):
        	result=purify(thresh[coords[i][1]:coords[i][1]+coords[i][3],coords[i][0]:coords[i][0]+coords[i][2]])
        	#cv2.imshow('result',result)
        	#cv2.waitKey(0)
        	filename='character'+str(i)+'.jpeg'
        	cv2.imwrite(filename,cv2.bitwise_not(result))
        	img_paths.append(filename)
    	return np.array(img_paths)
    	
