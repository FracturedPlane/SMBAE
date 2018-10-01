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
    trainData['fileName']='../../../Dropbox/Research/Projects/CharacterAnimation/Data/nav_Game/PPO/Nav_Sphere_5D/Deep_NN_TanH/trainingData_PPO.json'
    trainData['name']='Baseline'
    trainData['colour'] = (1.0, 0.0, 0.0, 1.0)
    trainingDatas.append(trainData)
    
    trainData={}
    trainData['fileName']='../../../Dropbox/Research/Projects/CharacterAnimation/Data/nav_Game/PPO/Nav_Sphere_5D/Deep_NN_TanH/trainingData_PPO.json'
    trainData['name']='Baseline'
    trainData['colour'] = (1.0, 0.0, 0.0, 1.0)
    trainingDatas.append(trainData)
    
    # Final method
    trainData={}
    trainData['fileName']='../../../Dropbox/Research/Projects/CharacterAnimation/Data/nav_Game/PPO/Nav_Sphere_MBAE_5D/Deep_NN_TanH/trainingData_PPO.json'
    trainData['name']='PPO + MBAE'
    trainData['colour'] = (0.0, 0.0, 1.0, 1.0)
    trainingDatas.append(trainData)
    
    trainData={}
    trainData['fileName']='../../../Dropbox/Research/Projects/CharacterAnimation/Data/nav_Game/PPO/Nav_Sphere_MBAE_5D/Deep_NN_TanH_2/trainingData_PPO.json'
    trainData['name']='PPO + MBAE'
    trainData['colour'] = (0.0, 0.0, 1.0, 1.0)
    trainingDatas.append(trainData)
    
    trainData={}
    trainData['fileName']='../../../Dropbox/Research/Projects/CharacterAnimation/Data/nav_Game/PPO/Nav_Sphere_MBAE_5D/Deep_NN_TanH_3/trainingData_PPO.json'
    trainData['name']='PPO + MBAE'
    trainData['colour'] = (0.0, 0.0, 1.0, 1.0)
    trainingDatas.append(trainData)
    
    trainData={}
    trainData['fileName']='../../../Dropbox/Research/Projects/CharacterAnimation/Data/nav_Game/PPO/Nav_Sphere_MBAE_FULL_5D/Deep_NN_TanH_pretrained_fd/trainingData_PPO.json'
    trainData['name']='PPO + MBAE_FULL'
    trainData['colour'] = (0.0, 1.0, 0.0, 1.0)
    trainingDatas.append(trainData)
    
    trainData={}
    trainData['fileName']='../../../Dropbox/Research/Projects/CharacterAnimation/Data/nav_Game/PPO/Nav_Sphere_MBAE_FULL_5D/Deep_NN_TanH/trainingData_PPO.json'
    trainData['name']='PPO + MBAE_FULL'
    trainData['colour'] = (0.0, 1.0, 0.0, 1.0)
    trainingDatas.append(trainData)
    
    
    
    
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
    rlv.updateRewards(trainingDatas)
    rlv.init()
    rlv.saveVisual("MBAE_Training_curves")
    rlv.show()