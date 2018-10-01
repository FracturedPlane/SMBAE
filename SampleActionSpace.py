
import sys
import os
import json
from numpy import dtype
sys.path.append("../")
sys.path.append("../characterSimAdapter/")
import math
import numpy as np

import characterSim
from ModelEvaluation import evalModel

import random
import cPickle

import cProfile, pstats, io
# import memory_profiler
# import psutil
import gc
# from guppy import hpy; h=hpy()
# from memprof import memprof

from ModelUtil import *
import itertools

def generateSamples(action_bounds, samples):
    num_samples=samples
    # construct samples using cartesian product
    # itertools.product(a,b)
    parameter_vectors=[]
    for i in range(len(action_bounds[0])):
        samps = np.linspace(action_bounds[0][i], action_bounds[1][i], num=num_samples)
        print samps
        parameter_vectors.append(samps)
    samples = list(itertools.product(*parameter_vectors))
    return samples

# @profile(precision=5)
# @memprof(plot = True)
def sampleActionSpace(settingsFileName):
        
    # pr = cProfile.Profile()
    # pr.enable()
    file = open(settingsFileName)
    settings = json.load(file)
    print "Settings: " + str(settings)
    file.close()
    anchor_data_file = open(settings["anchor_file"])
    _anchors = getAnchors(anchor_data_file)
    anchor_data_file.close()
    model_type= settings["model_type"]
    directory= settings["data_folder"]
    num_actions= settings["num_actions"]
    rounds = settings["rounds"]
    epochs = settings["epochs"]
    num_states=settings["num_states"]
    epsilon = settings["epsilon"]
    discount_factor=settings["discount_factor"]
    max_reward=settings["max_reward"]
    batch_size=settings["batch_size"]
    max_state=np.array(settings["max_state"], dtype=float)
    print "Sim config file name: " + str(settings["sim_config_file"])
    c = characterSim.Configuration(str(settings["sim_config_file"]))
    # c = characterSim.Configuration("../data/epsilon0Config.ini")
    action_space_continuous=True
    omega = settings["omega"]
    exploration_rate = settings["exploration_rate"]
    action_bounds = np.array(settings["action_bounds"], dtype=float)
    
    exp = characterSim.Experiment(c)
    
    state = characterSim.State()
    
    exp.getActor().init()
    exp.init()
    
    paramSampler = exp.getActor().getParamSampler()
    num_samples=2
    samples = []
    
    # construct samples using cartesian product
    # itertools.product(a,b)
    samples = generateSamples(action_bounds, num_samples)
    print len(samples)
    
    
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    state_num = 0
    action_result_samples=[]
    
    print "Starting first round"
    for actionSample in samples:
        exp.getActor().initEpoch()
        exp.getEnvironment().generateEnvironmentSample()
        exp.getEnvironment().initEpoch()
        # Actor should be FIRST here
        state = exp.getEnvironment().getState()
        initialState = exp.getEnvironment().getSimInterface().getController().getCOMPosition()
        state_ = state.getParams()
        
        reward = actContinuous(exp,actionSample)
        
        resultState = exp.getEnvironment().getSimInterface().getController().getCOMPosition()
        resultVel = exp.getEnvironment().getSimInterface().getController().getCOMVelocity()
        # print exp.getEnvironment().getSimInterface().getController().getLinks()[0].getCenterOfMassPosition().getX()
        if (state_num % 100) == 0:
            print actionSample, resultState.getX() - initialState.getX(), resultState.getY() - initialState.getY(), resultVel.length() 
        state_num += 1
        sample_=[]
        sample_.append(resultState.getX() - initialState.getX())
        sample_.append(resultState.getY() - initialState.getY())
        sample_.append(resultVel.length())
        samples_ = generateSamples(action_bounds, 3)
        next_action_result_samples=[]
        for actionSample_ in samples_:
            exp.getActor().initEpoch()
            exp.getEnvironment().generateEnvironmentSample()
            exp.getEnvironment().initEpoch()
            # Actor should be FIRST here
            state = exp.getEnvironment().getState()
            initialState = exp.getEnvironment().getSimInterface().getController().getCOMPosition()
            state_ = state.getParams()
            
            reward = actContinuous(exp,actionSample)
            
            # initialState = exp.getEnvironment().getSimInterface().getController().getCOMPosition()
            state_ = state.getParams()
            
            reward = actContinuous(exp,actionSample_)
            
            resultState = exp.getEnvironment().getSimInterface().getController().getCOMPosition()
            resultVel = exp.getEnvironment().getSimInterface().getController().getCOMVelocity()
            sample__=[]
            sample__.append(resultState.getX() - initialState.getX())
            sample__.append(resultState.getY() - initialState.getY())
            sample__.append(resultVel.length())
            next_action_result_samples.append(sample__)
            
        sample_.append(next_action_result_samples)
        action_result_samples.append(sample_)
        # print "Current Tuple: " + str(experience.current())
    
    return action_result_samples
    
if (__name__ == "__main__"):
    
    import matplotlib.pyplot as plt
    samples = sampleActionSpace(sys.argv[1])
    # print samples
    # samples = np.array(samples)
    # colours = (samples[:,2]- np.min(samples[:,2])) / (np.max(samples[:,2]) - np.min(samples[:,2]))
    # print colours
    
    for action in samples:
        # print "Action: " + str(action)
        
        # print "Action2: " + str(action2)
        actions = [[action[0], action[1], action[2]]] * len(action[3])
        # actions = np.array(actions)
        print actions
        actions2 = action[3]
        result = [None]*(len(actions)+len(actions2))
        result[::2] = actions
        result[1::2] = actions2
        actions = np.array(result)
        plt.plot(actions[:,0], actions[:,1], alpha=0.4)
        colours = (actions[:,2]- np.min(actions[:,2])) / (np.max(actions[:,2]) - np.min(actions[:,2]))
        plt.scatter(actions[:,0], actions[:,1], c=colours)
    # plt.plot(samples[:,0], samples[:,1])
    # plt.scatter(samples[:,0], samples[:,1], c=colours)
    plt.title("Action space sampling")
    plt.xlabel("X distance")
    plt.ylabel("Y distance")
    plt.show()
    