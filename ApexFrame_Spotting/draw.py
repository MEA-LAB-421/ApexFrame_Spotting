# angulus oris
import os
import numpy as np
import ApexSpotting_v2
import ApexSpotting
from predict_v2 import loadResult

featurePath = '../result'
subjectList = os.listdir(featurePath) 
for subject in subjectList: # traverse all subject
    subjectPath = featurePath + '/' + subject
    if not os.path.isdir(subjectPath):continue
    MEList = os.listdir(subjectPath)
    for ME in MEList:
        MEPath = subjectPath + '/' + ME
        print(MEPath)
        OF = []
        if not os.path.exists(MEPath + '/' + 'OF.txt'): continue
        startIdx = loadResult(MEPath + '/' + 'OF.txt', OF)
        OF = np.array(OF, dtype = 'float32').reshape(5, -1)
        ApexSpotting.drawPic(OF, MEPath)