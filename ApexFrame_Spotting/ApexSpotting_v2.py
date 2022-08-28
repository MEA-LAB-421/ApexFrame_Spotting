import os
import cv2
import dlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import trange
from utils import loadImages

predictor_path = "landmarks/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector() # face_detector
predictor = dlib.shape_predictor(predictor_path) # facial landmark predictor

def faceDetector(img):
    faces = detector(img, 1) # all potential faces
    for id, face in enumerate(faces):
        shape = predictor(img, face)
        landmarks = np.array([[p.x, p.y] for p in shape.parts()])
        return face, landmarks # use only one face, others are ignored
    return -1, -1
    
def pol2Cart(rho, phi): # Convert polar coordinates to cartesian coordinates for computation of optical strain
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y

def opticalStrain(u, v): # calculate optical strain from optical flow value
    ux = u - pd.DataFrame(u).shift(-1, axis=1)
    uy = u - pd.DataFrame(u).shift(-1, axis=0)
    vx = v - pd.DataFrame(v).shift(-1, axis=1)
    vy = v - pd.DataFrame(v).shift(-1, axis=0)
    res = np.array(np.sqrt(ux**2 + vy**2 + 1/2*(uy + vx)**2).ffill(1).ffill(0))
    return res

def getLandmark(landmarks, idx, shiftX, shiftY, type):
    # get landmark with offset
    if type == 'CASMEII' or type == 'SMIC':
        limitX = 640
        limitY = 480
    elif type == 'SAMM':
        limitX = 960
        limitY = 650
    x = max(min(landmarks[idx][0] + shiftX, limitX - 1), 0)
    y = max(min(landmarks[idx][1] + shiftY, limitY - 1), 0)
    return (x, y)

def printIdx(img, pos, id): # just for test
    cv2.putText(img, str(id), pos, fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                     fontScale=0.3, color=(0, 255, 0))

def generateMask(mask, pts, value): # generate poly region mask
    cv2.fillPoly(mask, [np.array(pts)], value)

def localFeature(Gshape, OF, OS, Vertexs, img = None): # get local feature include mean & StdDev of OS or OF
    # Gshape: global image shape
    # OF : global optical flow
    # OS : global optical strain
    # Vertexs : Vertexs of ROI
    # img : to test result
    Mask = np.zeros(Gshape, np.uint8)
    generateMask(Mask, Vertexs, 1)
    OFMean, OFStd = cv2.meanStdDev(src = OF, mask = Mask)
    OSMean, OSStd = cv2.meanStdDev(src = OS, mask = Mask)
    return OFMean, OSMean

def extractFeature(seq, K, type, OFMethod = 'Farneback'): # calculate globle dense optical flow, 
    # seq : video clip with micro expression
    # K : interval between 2 frames to calculate optical flow (ignored now)
    # type : type of dataset
    # OFMethod : method to solve optical flow, 'Farneback' or 'TVL1'
    # Farneback method is about 15 times faster than TV-L1 method, but TVL1 will be a little more accurate than Farneback method
    seqLength = len(seq)
    referenceFrame = None
    # Optical strain of global or local regions in different frames 
    globalOS = np.zeros(seqLength)
    rightEyebrowOS = np.zeros(seqLength)
    leftEyebrowOS = np.zeros(seqLength)
    rightEyeOS = np.zeros(seqLength)
    leftEyeOS = np.zeros(seqLength)
    rightMouthOS = np.zeros(seqLength)
    leftMouthOS = np.zeros(seqLength)

    globalOF = np.zeros(seqLength)
    rightEyebrowOF = np.zeros(seqLength)
    leftEyebrowOF = np.zeros(seqLength)
    rightEyeOF = np.zeros(seqLength)
    leftEyeOF = np.zeros(seqLength)
    rightMouthOF = np.zeros(seqLength)
    leftMouthOF = np.zeros(seqLength)

    interval = int(K * seqLength)
    for idx in trange(seqLength):
        if idx == 0: # use onset frame to predict landmarks
            face, landmarks = faceDetector(seq[idx])
            if type == 'SMIC':
                marginY = seq[idx].shape[0] // 10
                marginX = seq[idx].shape[1] // 10
            else:
                marginY = seq[idx].shape[0] // 20
                marginX = seq[idx].shape[1] // 20
            T = max(face.top() - marginY, 0)
            B = min(face.bottom() + marginY, seq[idx].shape[0])
            L = max(face.left() - marginX, 0)
            R = min(face.right() + marginX, seq[idx].shape[1])
            referenceFrame = seq[idx][T:B, L:R]
            face, landmarks = faceDetector(referenceFrame)
            # The first number of landmark point is its group, the second is its index in the group.

            # Face region
            P01 = (face.left(), face.top())
            P02 = (face.right(), face.top())
            P03 = (face.right(), face.bottom())
            P04 = (face.left(), face.bottom())

            # Right eye (on the left of img)
            eyeShift = 10  
            P11 = getLandmark(landmarks, 36, -eyeShift, 0, type)
            P12 = getLandmark(landmarks, 37, 0, -eyeShift, type) 
            P13 = getLandmark(landmarks, 38, 0, -eyeShift, type)
            P14 = getLandmark(landmarks, 39, eyeShift, 0, type)
            P15 = getLandmark(landmarks, 40, 0, eyeShift, type)
            P16 = getLandmark(landmarks, 41, 0, eyeShift, type)

            # Left eye (on the Right of img)
            P21 = getLandmark(landmarks, 42, -eyeShift, 0, type)
            P22 = getLandmark(landmarks, 43, 0, -eyeShift, type) 
            P23 = getLandmark(landmarks, 44, 0, -eyeShift, type)
            P24 = getLandmark(landmarks, 45, eyeShift, 0, type)
            P25 = getLandmark(landmarks, 46, 0, eyeShift, type)
            P26 = getLandmark(landmarks, 47, 0, eyeShift, type)

            eyebrowShift = 6
            # Right eyebrow (ROI 1)
            P31 = getLandmark(landmarks, 17, -eyebrowShift, 0, type)
            P32 = getLandmark(landmarks, 19, 0, -eyebrowShift, type)
            P33 = getLandmark(landmarks, 21, eyebrowShift, 0, type)
            P34 = P12

            # Left eyebrow (ROI 2)
            P41 = getLandmark(landmarks, 22, -eyebrowShift, 0, type)
            P42 = getLandmark(landmarks, 24, 0, -eyebrowShift, type)
            P43 = getLandmark(landmarks, 26, eyebrowShift, 0, type)
            P44 = P23

            # rightMouth (ROI 3)
            mouthShift = 12
            P51 = getLandmark(landmarks, 48, -mouthShift, 0, type)
            P52 = getLandmark(landmarks, 49, 0, -mouthShift, type)
            P53 = getLandmark(landmarks, 59, 0, mouthShift, type)

            # leftMouth (ROI 4)
            P61 = getLandmark(landmarks, 53, 0, -mouthShift, type)
            P62 = getLandmark(landmarks, 54, mouthShift, 0, type)
            P63 = getLandmark(landmarks, 55, 0, mouthShift, type)

            # Nose (reference object)
            P71 = getLandmark(landmarks, 28, 0, 0, type)

        imgA = referenceFrame
        imgB = seq[idx][T:B, L:R]
        imgA = cv2.cvtColor(imgA, cv2.COLOR_RGB2GRAY)
        imgB = cv2.cvtColor(imgB, cv2.COLOR_RGB2GRAY)
        imgA = np.array(imgA, np.uint8)
        imgB = imgB - imgB.mean() + imgA.mean()
        imgB = np.array(imgB, np.uint8)

        # calculate optical flow/strain
        if OFMethod == 'Farneback':
            flow = cv2.calcOpticalFlowFarneback(imgA, imgB, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        elif OFMethod == 'TVL1':
            opticalFlow = cv2.optflow.DualTVL1OpticalFlow_create()
            flow = opticalFlow.calc(imgA, imgB, None)
        
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        u, v = pol2Cart(magnitude, angle)
        opStrain = opticalStrain(u, v)

        # remove background movement like head movement
        u = abs(u - u[P71[1]-4:P71[1]+4, P71[0]-4:P71[0]+4].mean())
        v = abs(v - v[P71[1]-4:P71[1]+4, P71[0]-4:P71[0]+4].mean())
        opFlow = np.sqrt(u**2 + v**2)
        opStrain = opStrain - opStrain[P71[1]-4:P71[1]+4, P71[0]-4:P71[0]+4].mean()

        # calculate globle facial OS feature without eye region
        globalMask = np.zeros(imgA.shape, np.uint8)
        generateMask(globalMask, [P01, P02, P03, P04], 1) # choose facial region
        generateMask(globalMask, [P11, P12, P13, P14, P15, P16], 0) # mask
        generateMask(globalMask, [P21, P22, P23, P24, P25, P26], 0)
        generateMask(globalMask, [P31, P32, P33, P34], 0)
        generateMask(globalMask, [P41, P42, P43, P44], 0)
        generateMask(globalMask, [P51, P52, P53], 0)
        generateMask(globalMask, [P61, P62, P63], 0)
        
        globalMean, globalStd = cv2.meanStdDev(src = opFlow, mask = globalMask)
        globalOF[idx] = globalMean
        globalMean, globalStd = cv2.meanStdDev(src = opStrain, mask = globalMask)
        globalOS[idx] = globalMean

        # calculate local feature in eyebrow regions
        rightEyebrowOF[idx], rightEyebrowOS[idx] = localFeature(imgA.shape, opFlow, opStrain, [P31, P32, P33, P34], imgB)
        leftEyebrowOF[idx],  leftEyebrowOS[idx]  = localFeature(imgA.shape, opFlow, opStrain, [P41, P42, P43, P44], imgB)

        # calculate local feature in mouth region
        rightMouthOF[idx],   rightMouthOS[idx]   = localFeature(imgA.shape, opFlow, opStrain, [P51, P52, P53], imgB)
        leftMouthOF[idx],    leftMouthOS[idx]    = localFeature(imgA.shape, opFlow, opStrain, [P61, P62, P63], imgB)

        # calculate local feature in eye regions, they will be used to find apex frame only when other ROIs have no peaks.
        rightEyeOF[idx], rightEyeOS[idx] = localFeature(imgA.shape, opFlow, opStrain, [P11, P12, P13, P14, P15, P16], imgB)
        leftEyeOF[idx],  leftEyeOS[idx]  = localFeature(imgA.shape, opFlow, opStrain, [P21, P22, P23, P24, P25, P26], imgB)

    OS = [globalOS, rightEyebrowOS, leftEyebrowOS, rightMouthOS, leftMouthOS, rightEyeOS, leftEyeOS]
    OF = [globalOF, rightEyebrowOF, leftEyebrowOF, rightMouthOF, leftMouthOF, rightEyeOF, leftEyeOF]
    return OS, OF


def drawPic(value, path, type = 'FLow', startIdx = 0, K = 0):
    n = len(value[0])
    x = np.linspace(startIdx + 1,  startIdx + n - int(K*n) , n - int(K*n) )

    plt.title('Optical ' + type)  
    plt.xlabel('Time (frame index)') 
    plt.ylabel('Optical ' + type)  

    plt.plot(x, value[0], marker='o', markersize=3)  
    plt.plot(x, value[1], marker='s', markersize=3)
    plt.plot(x, value[2], marker='d', markersize=3)
    plt.plot(x, value[3], marker='p', markersize=3)
    plt.plot(x, value[4], marker='h', markersize=3)

    plt.legend(['Global', 'Right Eyebrow', 'Left Eyebrow', 'Right Angulus Oris', 'Left Angulus Oris'])  
    plt.savefig(os.path.join(path, 'Optical ' + type + '.eps'), dpi=600, format='eps')
    plt.cla()

    plt.title('Optical ' + type + ' in eye regions')  
    plt.xlabel('Time (frame index)') 
    plt.ylabel('Optical ' + type)  

    plt.plot(x, value[0], marker='o', markersize=3)  
    plt.plot(x, value[5], marker='s', markersize=3)
    plt.plot(x, value[6], marker='d', markersize=3)

    plt.legend(['Global', 'Right Eye', 'Left Eye'])  
    plt.savefig(os.path.join(path, 'Optical ' + type + ' in eye regions'+'.eps'), dpi=600, format='eps')
    plt.cla()

def saveData(filename, data, startIdx = 0): # save OS/OF data into file
    file = open(filename, 'w')
    file.write(str(startIdx) + '\n')
    for i in range(len(data)):
        s = str(data[i]).replace('[','').replace(']','') # delete [ ]
        s = s.replace(',','') +'\n'
        file.write(s)
    file.close()

def running(dataPath, resPath, type = 'CASMEII'):
    # dataPath : path of input dataset
    # resPath : path to save result, include line charts and raw data of OF/OS
    # type : type of dataset

    # folds of all subjects
    if type == 'CASMEII' or type == 'SAMM':
        subjectList = os.listdir(dataPath) 
        for subject in subjectList: # traverse all subject
            subjectPath = dataPath + '/' + subject
            if not os.path.isdir(subjectPath):continue
            subResPath = resPath + '/' + subject
            if not os.path.exists(subResPath):
                os.makedirs(subResPath)

            MEList = os.listdir(subjectPath)
            for ME in MEList: # traverse all Micro-Expression sequences of the subject
                print('Calculating OF/OS in ' + subject + '/' + ME)
                MEPath = subjectPath + '/' + ME
                MEResPath = subResPath + '/' + ME
                if os.path.exists(MEResPath): continue
                if not os.path.exists(MEResPath):
                    os.makedirs(MEResPath)
                startIdx, imgs = loadImages(MEPath, type)
                
                OS, OF = extractFeature(imgs, 0.2, type)
                drawPic(OF, MEResPath, 'Flow', startIdx, 0)
                drawPic(OS, MEResPath, 'Strain', startIdx, 0)
                saveData(os.path.join(MEResPath, 'OF.txt'), OF, startIdx)
                saveData(os.path.join(MEResPath, 'OS.txt'), OS, startIdx)
    elif type == 'SMIC': # SMIC
        xlsData = pd.read_excel(dataPath + '/' + 'SMIC-HS-E_annotation.xlsx')
        for i in xlsData.index:
            subject = xlsData.iloc[i]['Subject']
            name = xlsData.iloc[i]['Filename']
            onset = xlsData.iloc[i]['OnsetF']
            offset = xlsData.iloc[i]['OffsetF']
            if subject<10:
                subjectPath = dataPath + '/s0' + str(subject)
            else:
                subjectPath = dataPath + '/s' + str(subject)
            if not os.path.isdir(subjectPath): continue
            if subject<10:
                subResPath = resPath + '/s0' + str(subject)
            else:
                subResPath = resPath + '/s' + str(subject)
            if not os.path.exists(subResPath):
                os.makedirs(subResPath)

            MEPath = subjectPath + '/' + name
            MEResPath = subResPath + '/' + name
            print('Calculating OF/OS in ' + MEPath)
            if os.path.exists(MEResPath): continue
            if not os.path.exists(MEResPath):
                os.makedirs(MEResPath)
            
            startIdx, imgs = loadImages(MEPath, type, onset, offset)
            OS, OF = extractFeature(imgs, 0.2, type)
            drawPic(OF, MEResPath, 'Flow', startIdx, 0)
            drawPic(OS, MEResPath, 'Strain', startIdx, 0)
            saveData(os.path.join(MEResPath, 'OF.txt'), OF, startIdx)
            saveData(os.path.join(MEResPath, 'OS.txt'), OS, startIdx)

def tmpRun(filePath):
    with open(filePath, 'r') as paths:
        pathList = paths.readlines()
        for path in pathList:
            MEPath = '../../dataset/CASME2/CASME2_RAW_selected/sub' + path[:-1]
            MEResPath = '../result/sub' + path[:-1]
            startIdx, imgs = loadImages(MEPath, 'CASMEII')
            print('Calculating OF/OS in ' + 'sub' + path)
            OS, OF = extractFeature(imgs, 0.2)
            drawPic(OF, MEResPath, 'Flow', startIdx, 0)
            drawPic(OS, MEResPath, 'Strain', startIdx, 0)
            saveData(os.path.join(MEResPath, 'OF.txt'), OF, startIdx)
            saveData(os.path.join(MEResPath, 'OS.txt'), OS, startIdx)

if __name__ == '__main__':
    running('../../dataset/CASME2/CASME2_RAW_selected', '../resultCASMEII', 'CASMEII')
    # running('../../dataset/SAMM/SAMM', '../resultSAMM', 'SAMM')
    # running('../../dataset/SMIC/SMIC-E_raw image/HS_long/SMIC-HS-E', '../resultSMIC_HS_E', 'SMIC')
    # tmpRun('namelist.txt')
