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
    otherDatas = []
    
    # Need to train a better Baseline
    trainData={}
    trainData['fileName']='../../../Dropbox/Research/Projects/CharacterAnimation/Data/mocapImitation2D/A_CACLA/Simple_Walk_PD_Imitate/Deep_NN_Dropout_Critic/trainingData_A_CACLA.json'
    trainData['name']='Baseline'
    trainData['colour'] = (1.0, 0.0, 0.0, 1.0)
    trainingDatas.append(trainData)
    trainData={}
    trainData['fileName']='../../../Dropbox/Research/Projects/CharacterAnimation/Data/mocapImitation2D/A_CACLA/Simple_Walk_Imitate/Deep_NN_Wide_Dropout_Critic/trainingData_A_CACLA.json'
    trainData['name']='Baseline'
    trainData['colour'] = (1.0, 0.0, 0.0, 1.0)
    trainingDatas.append(trainData)
    
    trainData={}
    trainData['fileName']='../../../Dropbox/Research/Projects/CharacterAnimation/Data/mocapImitation2D/A_CACLA/Simple_Walk_PD_Imitate_MBAE/Deep_NN_Dropout_Critic/trainingData_A_CACLA.json'
    trainData['name']='Baseline + MBAE'
    trainData['colour'] = (0.0, 0.0, 1.0, 1.0)
    otherDatas.append(trainData)
    
    trainData={}
    trainData['fileName']='../../../Dropbox/Research/Projects/CharacterAnimation/Data/mocapImitation2D/A_CACLA/Simple_Walk_PD_Imitate_MBAE2/Deep_NN_Dropout_Critic/trainingData_A_CACLA.json'
    trainData['name']='Baseline + MBAE'
    trainData['colour'] = (0.0, 0.0, 1.0, 1.0)
    otherDatas.append(trainData)
    
    for i in range(len(otherDatas)):
        datafile = otherDatas[i]['fileName']
        file = open(datafile)
        otherDatas[i]['data'] = json.load(file)
        # print "Training data: " + str(trainingData)
        file.close()
    
    
    for i in range(len(trainingDatas)):
        datafile = trainingDatas[i]['fileName']
        file = open(datafile)
        trainingDatas[i]['data'] = json.load(file)
        # print "Training data: " + str(trainingData)
        file.close()
    
    if (len(sys.argv) == 3):
        length = int(sys.argv[2])
    
    """
    trainData["mean_reward"]=[]
    trainData["std_reward"]=[]
    trainData["mean_bellman_error"]=[]
    trainData["std_bellman_error"]=[]
    trainData["mean_discount_error"]=[]
    trainData["std_discount_error"]=[]
    
    """
    settings = None
    if (len(sys.argv) >= 2):
        settingsFileName = sys.argv[1]
        settingsFile = open(settingsFileName, 'r')
        settings = json.load(settingsFile)
        settingsFile.close()
    rlv = PolicyTrainVisualize("Training Curves", settings=settings)
    if (len(sys.argv) == 3):
        length = int(sys.argv[2])
        rlv.setLength(length)
    rlv.setBinSize(1)
    rlv.updateRewards(trainingDatas, otherDatas)
    rlv.init()
    rlv.saveVisual("2D_LLC_MBAE_Training_curves")
    rlv.show()
    