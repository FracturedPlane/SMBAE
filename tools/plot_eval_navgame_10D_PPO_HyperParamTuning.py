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
    trainData['fileName']='./nav_Game/PPO/Nav_Sphere_MBAE_FULL_10D/Deep_NN_TanH/trainingData_PPO.json'
    trainData['name']='Baseline'
    trainData['colour'] = (0.0, 0.0, 0.0, 1.0)
    trainingDatas.append(trainData)
    
    trainData={}
    trainData['fileName']='./nav_Game/PPO/Nav_Sphere_MBAE_FULL_10D_2/Deep_NN_TanH/trainingData_PPO.json'
    trainData['name']='Baseline'
    trainData['colour'] = (0.0, 0.0, 0.0, 1.0)
    trainingDatas.append(trainData)
    
    # Final method
    trainData={}
    trainData['fileName']='./nav_Game/PPO/Nav_Sphere_MBAE_FULL_10D_LESS_MBAE/Deep_NN_TanH/trainingData_PPO.json'
    trainData['name']='PPO + MBAE Less'
    trainData['colour'] = (0.0, 0.0, 1.0, 1.0)
    trainingDatas.append(trainData)
    
        # Final method
    trainData={}
    trainData['fileName']='./nav_Game/PPO/Nav_Sphere_MBAE_FULL_10D_LESS_MBAE_2/Deep_NN_TanH/trainingData_PPO.json'
    trainData['name']='PPO + MBAE Less'
    trainData['colour'] = (0.0, 0.0, 1.0, 1.0)
    trainingDatas.append(trainData)
    
    trainData={}
    trainData['fileName']='./nav_Game/PPO/Nav_Sphere_MBAE_FULL_10D_MORE_MBAE/Deep_NN_TanH/trainingData_PPO.json'
    trainData['name']='PPO + MBAE More'
    trainData['colour'] = (0.0, 1.0, 0.0, 1.0)
    trainingDatas.append(trainData)
    
    trainData={}
    trainData['fileName']='./nav_Game/PPO/Nav_Sphere_MBAE_FULL_10D_MORE_MBAE_2/Deep_NN_TanH/trainingData_PPO.json'
    trainData['name']='PPO + MBAE More'
    trainData['colour'] = (0.0, 1.0, 0.0, 1.0)
    trainingDatas.append(trainData)
    
    trainData={}
    trainData['fileName']='./nav_Game/PPO/Nav_Sphere_MBAE_FULL_10D_SMALLER_MBAE/Deep_NN_TanH/trainingData_PPO.json'
    trainData['name']='PPO + MBAE Smaller'
    trainData['colour'] = (1.0, 0.0, 0.0, 1.0)
    trainingDatas.append(trainData)
    
    trainData={}
    trainData['fileName']='./nav_Game/PPO/Nav_Sphere_MBAE_FULL_10D_SMALLER_MBAE_2/Deep_NN_TanH/trainingData_PPO.json'
    trainData['name']='PPO + MBAE Smaller'
    trainData['colour'] = (1.0, 0.0, 0.0, 1.0)
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