
"""
I should put some notes here on how to use this...

    This script loads a model with name agent.pkl in the directory that would be created
    by the setttings file. It also creats a new network using the configuration
    from the settings file. You need to put the network model you want to copy the 
    parameters from in the data folder. The new network will be created in the same 
    folder with the name agent_Injected.pkl.

"""

import sys
import numpy as np
import dill
import dill as pickle
import dill as cPickle
import sys
sys.path.append('../')

def combineNetworkModels(settings_file_name):
    
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
    from util.SimulationUtil import getDataDirectory, createForwardDynamicsModel, createSampler, getAgentName
    
    
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
    new_model = createRLAgent(settings['agent_name'], state_bounds, discrete_actions, reward_bounds, settings)
    params = new_model.getNetworkParameters()
    print ("New Network Critic shape")
    for i in range(len(params[0])):
        print (params[0][i].shape)
    print ("New Network Critic shape, done")
    
    if (True):
        file_name=directory+getAgentName()+".pkl"
    else:
        file_name=directory+getAgentName()+"_Best.pkl"
    print("Loading model: ", file_name)
    f = open(file_name, 'rb')
    old_model = dill.load(f)
    f.close()
    print ("State Length: ", len(old_model.getStateBounds()[0]) )
    
    if (True):
        new_model.setAgentNetworkParamters(old_model)
        new_model.setCombinedNetworkParamters(old_model)
        # new_model.setMergeLayerNetworkParamters(old_model)
        new_model.setMergeLayerNetworkParamters(old_model, zeroInjectedMergeLayer=True)
    else:
        new_model.setNetworkParameters(old_model.getNetworkParameters())
        
        
    params = new_model.getNetworkParameters()
    print ("New Network Critic shape")
    for i in range(len(params[0])):
        print (params[0][i].shape)
    for i in range(len(params[1])):
        print (params[1][i].shape)
    ### Modify state bounds
    state_bounds[:,settings['num_terrain_features']: len(state_bounds[0])] = old_model.getStateBounds()
    print ("State bounds: ", state_bounds.shape)
    print (state_bounds) 
    new_model.setStateBounds(state_bounds)
    new_model.setActionBounds(old_model.getActionBounds())
    new_model.setRewardBounds(old_model.getRewardBounds())
    
    file_name=directory+getAgentName()+"_Injected.pkl"
    f = open(file_name, 'wb')
    dill.dump(new_model, f)
    f.close()
    

if __name__ == '__main__':


    combineNetworkModels(sys.argv[1])