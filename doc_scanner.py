import cv2
import numpy as np

widthImg = 480
heightImg = 640
cap = cv2.VideoCapture(0)
cap.set(3, widthImg)
cap.set(4, heightImg)
cap.set(10, 150)

def preprocessing(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)
    imgCanny = cv2.Canny(imgBlur,100,100)
    kernel = np.ones((5,5))
    imgDialation = cv2.dilate(imgCanny, kernel, iterations = 2)
    imgErode = cv2.erode(imgDialation, kernel, iterations = 1)
    return imgErode

def getContours(img):
    maxArea = 0
    large = np.array([])
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
       area = cv2.contourArea(cnt)
       if area>5000:
           #cv2.drawContours(imgCont, cnt, -1, (255,0,0), 3)
           perimeter = cv2.arcLength(cnt, True)
           approx = cv2.approxPolyDP(cnt, 0.02*perimeter, True)
           if area > maxArea and len(approx) == 4:
               large = approx
               maxArea = area
    cv2.drawContours(imgCont, large, -1, (255,0,0), 20)           
    return large

def reorder(myPoints):
    myPoints =  myPoints.reshape((4,2)) 
    myNewPoints = np.zeros((4,1,2),np.int32)
    add = myPoints.sum(1)
    #print("add : ",add)  

    myNewPoints[0] = myPoints[np.argmin(add)]
    myNewPoints[3] = myPoints[np.argmin(add)]
    diff = np.diff(myPoints,axis=1)
    myNewPoints[1] = myPoints[np.argmin(diff)]
    myNewPoints[2] = myPoints[np.argmin(diff)]
    #print("NewPoints",myNewPoints)

    return myNewPoints

def getWarp(img, large):
    large = reorder(large)
    pts1 = np.float32([large])
    pts2 = np.float32([[0,0],[widthImg,0],[0,heightImg],[widthImg,heightImg]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv2.warpPerspective(img, matrix, (widthImg,heightImg))
    imgCrop = imgOutput[10:imgOutput.shape[0]-10, 10:imgOutput.shape[1]-10]
    imgCrop = cv2.resize(imgCrop(widthImg,heightImg))
    
    return imgCrop

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

while True:
    success, img = cap.read()
    img=cv2.resize(img,(widthImg,heightImg))
    imgCont = img.copy()
    imgErode = preprocessing(img)
    large = getContours(imgErode)
    #print(large)
    if large.size!=0:
        imgOutput = getWarp(img, large)
        imgArray = np.array([img, imgErode],[imgCont, imgOutput])
    else:
        imgArray = np.array([img, imgErode],[img, img])
    imgstacked = (0.6 ,imgArray)
    cv2.imshow("Result",imgstacked)
    cv2.imshow("Result",imgOutput)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break