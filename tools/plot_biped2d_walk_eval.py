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
    trainData['fileName']='../../../Dropbox/Research/Projects/CharacterAnimation/Data/simbiconBiped2D/A3C/Simple_Walk/Deep_NN/trainingData_A3C.json'
    trainData['name']='Baseline'
    trainingDatas.append(trainData)
    
    
    trainData={}
    trainData['fileName']='../../../Dropbox/Research/Projects/CharacterAnimation/Data/simbiconBiped2D/A_CACLA/Simple_walk/Deep_NN_Dropout_Critic/trainingData_A_CACLA.json'
    trainData['name']='Baseline + DropoutCritic'
    trainingDatas.append(trainData)
    
    
    for i in range(len(trainingDatas)):
        datafile = trainingDatas[i]['fileName']
        file = open(datafile)
        trainingDatas[i]['data'] = json.load(file)
        # print "Training data: " + str(trainingData)
        file.close()
    
    """
    trainData["mean_reward"]=[]
    trainData["std_reward"]=[]
    trainData["mean_bellman_error"]=[]
    trainData["std_bellman_error"]=[]
    trainData["mean_discount_error"]=[]
    trainData["std_discount_error"]=[]
    
    """
    
    rlv = PolicyTrainVisualize("Training Curves")
    if (len(sys.argv) == 2):
        length = int(sys.argv[1])
        rlv.setLength(length)
    rlv.updateRewards(trainingDatas)
    rlv.init()
    rlv.saveVisual("Biped2.5D_Training_curves")
    rlv.show()