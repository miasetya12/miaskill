import cv2
import numpy as np
import utlis

###################################################
width = 900
height = 900
pathImage = r"semoga1.jpeg"
qNum = 10  # number of questions
nChoices = 5  # number of choices 
maxGrade = 100
answers = [0,1,2,3,4,0,1,2,3,3]
webcamFeed = True
###################################################
cap = cv2.VideoCapture(r"test/video.mp4")
count = 0

while True:
    if webcamFeed:
        success, img = cap.read()
    else:
        img = cv2.imread(pathImage)

    # Pre Processing
    img = cv2.resize(img, (width, height)) # RESIZE IMAGE
    imgFinal = img.copy()
    imgContours = img.copy()
    imgBiggestContour = img.copy()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # CONVERT IMAGE TO GRAY SCALE
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1) # ADD GAUSSIAN BLUR
    imgCanny = cv2.Canny(imgBlur, 10, 50) # APPLY CANNY 


    try:
        # Finding all contours
        contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # FIND ALL CONTOURS
        cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10) # DRAW ALL DETECTED CONTOURS

        # Find Rectangle Contours
        rectCon = utlis.rectContour(contours) # FILTER FOR RECTANGLE CONTOURS
        biggestContour = utlis.getCornerPoints(rectCon[0]) # GET CORNER POINTS OF THE BIGGEST RECTANGLE
        # print(biggestContour)
        gradePoints = utlis.getCornerPoints(rectCon[1])  # GET CORNER POINTS OF THE SECOND BIGGEST RECTANGLE
        # print(gradePoints)

        if biggestContour.size != 0 and gradePoints.size != 0:
            cv2.drawContours(imgBiggestContour,biggestContour, -1, (0, 255, 0), 20)
            cv2.drawContours(imgBiggestContour,gradePoints, -1, (255, 0, 0), 20)
            biggestContour = utlis.reorder(biggestContour)# REORDER FOR WARPING
            gradePoints = utlis.reorder(gradePoints) # REORDER FOR WARPING

            pt1 = np.float32(biggestContour) # PREPARE POINTS FOR WARP
            pt2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])# PREPARE POINTS FOR WARP
            matrix = cv2.getPerspectiveTransform(pt1, pt2) # GET TRANSFORMATION MATRIX
            imgWarpColored = cv2.warpPerspective(img, matrix, (width, height)) # APPLY WARP PERSPECTIVE

            ptsG1 = np.float32(gradePoints)  # PREPARE POINTS FOR WARP
            # PREPARE POINTS FOR WARP
            ptsG2 = np.float32([[0, 0], [325, 0], [0, 150], [325, 150]])
            matrixG = cv2.getPerspectiveTransform(ptsG1, ptsG2)# GET TRANSFORMATION MATRIX
            imgGradeDisplay = cv2.warpPerspective(img, matrixG, (325, 150)) # APPLY WARP PERSPECTIVE

        # Apply threshild
            imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY) # CONVERT TO GRAYSCALE
            imgThresh = cv2.threshold(imgWarpGray, 170, 255, cv2.THRESH_BINARY_INV)[1] # APPLY THRESHOLD AND INVERSE

            boxes = utlis.splitBoxes(imgThresh, qNum, nChoices) # GET INDIVIDUAL BOXES
            cv2.imshow("test", boxes[0])

        # Find the boxes with heighest non zero pixels
            #print(cv2.countNonZero(boxes[0]), cv2.countNonZero(boxes[1]))
            myPixelsVal = np.zeros((qNum, nChoices)) # TO STORE THE NON ZERO VALUES OF EACH BOX
            countC = 0
            countR = 0
            for image in boxes:
                totalPixels = cv2.countNonZero(image)
                myPixelsVal[countR][countC] = totalPixels
                countC += 1
                if (countC == nChoices):
                    countR += 1
                    countC = 0
            # print(myPixelsVal)

        # find index values of the marking in each raw
            myIndex = []
            for x in range(0, qNum):
                arr = myPixelsVal[x]
                # print('arr',arr)
                myIndexVal = np.where(arr == np.max(arr))
                # print(myIndexVal[0])
                myIndex.append(myIndexVal[0][0])
            # print(myIndex)

        # Grading
            grading = []
            for x in range(0, qNum):
                if answers[x] == myIndex[x]:
                    grading.append(1)
                else:
                    grading.append(0)
            # print(grading)

        # find the final score
            score = (sum(grading)/qNum) * maxGrade
            # print(score)

        # Displaying answers and score
            imgResults = imgWarpColored.copy()
            imgResults = utlis.showAnswers(imgResults, myIndex, grading, answers, qNum, nChoices)

        # Mask and combine the answers over the original image
            imgRawDrawing = np.zeros_like(imgWarpColored)
            imgRawDrawing = utlis.showAnswers(imgRawDrawing, myIndex, grading, answers, qNum, nChoices)# DRAW ON NEW IMAGE
            invmatrix = cv2.getPerspectiveTransform(pt2, pt1)# INVERSE TRANSFORMATION MATRIX# INVERSE TRANSFORMATION MATRIX
            imgInvWarp = cv2.warpPerspective(imgRawDrawing, invmatrix, (width, height)) # INV IMAGE WARP

        # Mask and combine the grade over the original image
            imgRawGrade = np.zeros_like(imgGradeDisplay) # NEW BLANK IMAGE WITH GRADE AREA SIZE
            cv2.putText(imgRawGrade, str(round(score, 1))+"%", (30, 100),cv2.FONT_HERSHEY_SIMPLEX, 3, (250, 250, 250), 5) # ADD THE GRADE TO NEW IMAGE
            invMatrixG = cv2.getPerspectiveTransform(ptsG2, ptsG1)  # INVERSE TRANSFORMATION MATRIX
            imgInvGradeDisplay = cv2.warpPerspective(imgRawGrade, invMatrixG, (width, height)) # INV IMAGE WARP
            # SHOW ANSWERS AND GRADE ON FINAL IMAGE
            imgFinal = cv2.addWeighted(imgFinal, 1, imgInvWarp, 1, 0)
            imgFinal = cv2.addWeighted(imgFinal, 0.8, imgInvGradeDisplay, -1, 0)
        # IMAGE ARRAY FOR DISPLAY
        imgBlank = np.zeros_like(img)
        imageArray = ([img, imgGray, imgBlur, imgCanny],
                      [imgContours, imgBiggestContour, imgWarpColored, imgThresh],
                      [imgResults, imgRawDrawing, imgInvWarp, imgFinal])

    except:
        imgBlank = np.zeros_like(img)
        imageArray = ([img, imgGray, imgBlur, imgCanny],
                      [imgContours, imgBiggestContour, imgWarpColored, imgThresh],
                      [imgBlank, imgBlank, imgBlank, imgFinal])

    imgStacked = utlis.stackImages(imageArray, 0.3)

    cv2.imshow('final', imgFinal)
    cv2.imshow("Stacked Images", imgStacked)
    #cv2.moveWindow("Stacked Images",0,0)
   
    # SAVE IMAGE WHEN 's' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("D:\Dropbox\python\openCV\OMR\Scanned/myImage " +str(count)+".jpg", imgFinal)