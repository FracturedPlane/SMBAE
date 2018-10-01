

import matplotlib.pyplot as plt
# from matplotlib import mpl
import numpy as np
# import matplotlib.animation as animation
import random
import sys
import json
from PolicyTrainVisualize import PolicyTrainVisualize

if __name__ == "__main__":
    
    trainingDatas = []
    
    # Need to train a better Baseline
    trainData={}
    trainData['fileName']='../../../Dropbox/Research/Projects/CharacterAnimation/Data/gapgame_2d/A_CACLA/Gaps_Sphere/Deep_CNN/trainingData_A_CACLA.json'
    trainData['name']='Off Policy 1'
    trainingDatas.append(trainData)
    
    trainData={}
    trainData['fileName']='../../../Dropbox/Research/Projects/CharacterAnimation/Data/gapgame_2d/A_CACLA/Gaps_Sphere_2/Deep_CNN/trainingData_A_CACLA.json'
    trainData['name']='Off Policy 2'
    trainingDatas.append(trainData)
    
    trainData={}
    trainData['fileName']='../../../Dropbox/Research/Projects/CharacterAnimation/Data/gapgame_2d/A_CACLA/Gaps_Sphere_OnPolicy/Deep_CNN/trainingData_A_CACLA.json'
    trainData['name']='On Policy 1'
    trainingDatas.append(trainData)
    
    trainData={} 
    trainData['fileName']='../../../Dropbox/Research/Projects/CharacterAnimation/Data/gapgame_2d/A_CACLA/Gaps_Sphere_OnPolicy_2/Deep_CNN/trainingData_A_CACLA.json'
    trainData['name']='On Policy 2'
    trainingDatas.append(trainData)
    
    
    for i in range(len(trainingDatas)):
        datafile = trainingDatas[i]['fileName']
        file = open(datafile)
        trainingDatas[i]['data'] = json.load(file)
        # print "Training data: " + str(trainingData)
        file.close()
    
    if (len(sys.argv) == 3):
        length = int(sys.argv[2])
    
    rlv = PolicyTrainVisualize("Training Curves")
    rlv.updateRewards(trainingDatas)
    rlv.init()
    rlv.saveVisual("agent")
    rlv.show()