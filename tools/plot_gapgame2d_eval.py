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
    """
    trainData={}
    trainData['fileName']='../../../Dropbox/Research/Projects/CharacterAnimation/Data/gapgame_2d/A_CACLA/Gaps_Sphere_perfect/Deep_CNN/trainingData_A_CACLA.json'
    trainData['name']='Baseline'
    trainingDatas.append(trainData)
    """
    # Need to train a better Baseline
    trainData={}
    trainData['fileName']='../../../Dropbox/Research/Projects/CharacterAnimation/Data/gapgame_2d/A_CACLA/Gaps_Sphere/Deep_CNN_Dropout_Critic/trainingData_A_CACLA.json'
    trainData['name']='Baseline'
    trainingDatas.append(trainData)
    """
    trainData={}
    trainData['fileName']='../../../Dropbox/Research/Projects/CharacterAnimation/Data/gapgame_2d/A_CACLA/Gaps_Sphere/Deep_CNN_Dropout/trainingData_A_CACLA.json'
    trainData['name']='Baseline + Dropout'
    trainingDatas.append(trainData)
    
    trainData={}
    trainData['fileName']='../../../Dropbox/Research/Projects/CharacterAnimation/Data/gapgame_2d/A_CACLA/Gaps_Sphere/Deep_CNN_Dropout_Critic/trainingData_A_CACLA.json'
    trainData['name']='Baseline + DropoutOnCriticOnly'
    trainingDatas.append(trainData)
    """
    """
    trainData={}
    trainData['fileName']='../../../Dropbox/Research/Projects/CharacterAnimation/Data/gapgame_2d/A_CACLA/Gaps_Sphere_ActorBufFix/Deep_CNN/trainingData_A_CACLA.json'
    trainData['name']='Baseline + ActorBatchSize32'
    trainingDatas.append(trainData)
    
    trainData={}
    trainData['fileName']='../../../Dropbox/Research/Projects/CharacterAnimation/Data/gapgame_2d/A_CACLA/Gaps_Sphere_MBAE/Deep_CNN/trainingData_A_CACLA.json'
    trainData['name']='Baseline + MBAE'
    trainingDatas.append(trainData)
    
    trainData={}
    trainData['fileName']='../../../Dropbox/Research/Projects/CharacterAnimation/Data/gapgame_2d/A_CACLA/Gaps_Sphere_OnPolicy_2/Deep_CNN/trainingData_A_CACLA.json'
    trainData['name']='Baseline + OnPolicy'
    trainingDatas.append(trainData)
    
    trainData={}
    trainData['fileName']='../../../Dropbox/Research/Projects/CharacterAnimation/Data/gapgame_2d/A_CACLA/Gaps_Sphere_OnPolicy_ProxReg/Deep_CNN/trainingData_A_CACLA.json'
    trainData['name']='Baseline + OnPolicy + ProxReg'
    trainingDatas.append(trainData)
    
    trainData={}
    trainData['fileName']='../../../Dropbox/Research/Projects/CharacterAnimation/Data/gapgame_2d/A_CACLA/Gaps_Sphere_OnPolicy_KLReg/Deep_CNN/trainingData_A_CACLA.json'
    trainData['name']='Baseline + OnPolicy + KLReg'
    trainingDatas.append(trainData)
    
    trainData={}
    trainData['fileName']='../../../Dropbox/Research/Projects/CharacterAnimation/Data/gapgame_2d/A_CACLA/Gaps_Sphere_ProximalREg/Deep_CNN/trainingData_A_CACLA.json'
    trainData['name']='Baseline + ProximalRegularization'
    trainingDatas.append(trainData)
    """
    """
    trainData={}
    trainData['fileName']='../../../Dropbox/Research/Projects/CharacterAnimation/Data/gapgame_2d/A_CACLA/Gaps_Sphere_MBAE/Deep_CNN/trainingData_A_CACLA.json'
    trainData['name']='Baseline + MBAE2'
    trainingDatas.append(trainData)
    """
    """
    trainData={}
    trainData['fileName']='../../../Dropbox/Research/Projects/CharacterAnimation/Data/gapgame_2d/A_CACLA/Gaps_Sphere_MBAE/Deep_CNN_Dropout_Critic/trainingData_A_CACLA.json'
    trainData['name']='Baseline + MBAE2'
    trainingDatas.append(trainData)
    """
    """
    trainData={}
    trainData['fileName']='../../../Dropbox/Research/Projects/CharacterAnimation/Data/gapgame_2d/A_CACLA/Gaps_Sphere_MBAE_2/Deep_CNN_Dropout_Critic/trainingData_A_CACLA.json'
    trainData['name']='Baseline + MBAE2 + Dyna'
    trainingDatas.append(trainData)
    """
    trainData={}
    trainData['fileName']='../../../Dropbox/Research/Projects/CharacterAnimation/Data/gapgame_2d/A_CACLA/Gaps_Sphere_FD/Deep_CNN_SingleNet/trainingData_A_CACLA.json'
    trainData['name']='Baseline + SingleNet'
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
    rlv.saveVisual("GapGame2D_Training_curves")
    rlv.show()
    
    