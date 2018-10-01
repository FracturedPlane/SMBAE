
"""
    This file is designed to print out a bunch of info about the passed model
"""

import sys
import numpy as np
import dill
import dill as pickle
import dill as cPickle
import sys
sys.path.append('../')

def getModelInfo(settings_file_name, model_file_name):
    
    from model.ModelUtil import getSettings
    settings = getSettings(settings_file_name)
    # settings['shouldRender'] = True
    import os    
    os.environ['THEANO_FLAGS'] = "mode=FAST_RUN,device="+settings['training_processor_type']+",floatX="+settings['float_type']
    
    ## Theano needs to be imported after the flags are set.
    # from ModelEvaluation import *
    # from model.ModelUtil import *
    from ModelEvaluation import SimWorker, evalModelParrallel, collectExperience
    # from model.ModelUtil import validBounds
    from model.LearningAgent import LearningAgent, LearningWorker
    from util.SimulationUtil import validateSettings, createEnvironment, createRLAgent, createActor
    from util.SimulationUtil import getDataDirectory, createForwardDynamicsModel, createSampler
    
    
    from util.ExperienceMemory import ExperienceMemory
    from RLVisualize import RLVisualize
    from NNVisualize import NNVisualize
    
    directory= getDataDirectory(settings)
    
    rounds = settings["rounds"]
    epochs = settings["epochs"]
    # num_states=settings["num_states"]
    epsilon = settings["epsilon"]
    discount_factor=settings["discount_factor"]
    reward_bounds=np.array(settings["reward_bounds"])
    # reward_bounds = np.array([[-10.1],[0.0]])
    batch_size=settings["batch_size"]
    train_on_validation_set=settings["train_on_validation_set"]
    state_bounds = np.array(settings['state_bounds'])
    discrete_actions = np.array(settings['discrete_actions'])
    num_actions= discrete_actions.shape[0] # number of rows
    print ("Sim config file name: " + str(settings["sim_config_file"]))
    # c = characterSim.Configuration(str(settings["sim_config_file"]))
    # c = characterSim.Configuration("../data/epsilon0Config.ini")
    action_space_continuous=settings['action_space_continuous']
    
    settings['load_saved_model'] = False
    # new_model = createRLAgent(settings['agent_name'], state_bounds, discrete_actions, reward_bounds, settings)
    
    print("Loading model: ", model_file_name)
    f = open(model_file_name, 'rb')
    old_model = dill.load(f)
    f.close()
    print ("State Length: ", len(old_model.getStateBounds()[0]) )
    
    params = old_model.getNetworkParameters()
    print ("Network Critic shape")
    for i in range(len(params[0])):
        print (params[0][i].shape)
    print ("Network Actor shape")
    for i in range(len(params[1])):
        print (params[1][i].shape)
        
    print ("Network Critic params")
    for i in range(len(params[0])):
        print ("Layer: ", i , " shape ", params[0][i].shape)
        print (params[0][i])
    print ("Network Actor params")
    for i in range(len(params[1])):
        print ("Layer: ", i , " shape ", params[1][i].shape)
        print (params[1][i])
        
    ### Modify state bounds
    print ("State bounds:")
    print (old_model.getStateBounds())
    print ("Action bounds:")
    print (old_model.getActionBounds())
    print ("Reward bounds:")
    print (old_model.getRewardBounds())
    

if __name__ == '__main__':


    getModelInfo(sys.argv[1], sys.argv[2])