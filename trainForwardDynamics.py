
"""
theano.config.device='gpu'
theano.config.mode='FAST_RUN'
theano.config.floatX='float32'
"""

import numpy as np
import sys
import json
sys.path.append('../')
import dill
import datetime
import os

def trainForwardDynamics(settingsFileName):
    """
    State is the input state and Action is the desired output (y).
    """
    # from model.ModelUtil import *
    
    np.random.seed(23)
    file = open(settingsFileName)
    settings = json.load(file)
    print ("Settings: " , str(json.dumps(settings)))
    file.close()
    import os    
    os.environ['THEANO_FLAGS'] = "mode=FAST_RUN,device="+settings['training_processor_type']+",floatX="+settings['float_type']
    
    # import theano
    # from theano import tensor as T
    # import lasagne
    from util.SimulationUtil import validateSettings
    from util.SimulationUtil import getDataDirectory
    from util.SimulationUtil import createForwardDynamicsModel, createRLAgent
    from model.NeuralNetwork import NeuralNetwork
    from util.ExperienceMemory import ExperienceMemory
    import matplotlib.pyplot as plt
    import math
    # from ModelEvaluation import *
    # from util.SimulationUtil import *
    import time  
    
    settings = validateSettings(settings)
    
    # anchor_data_file = open(settings["anchor_file"])
    # _anchors = getAnchors(anchor_data_file)
    # print ("Length of anchors epochs: ", str(len(_anchors)))
    # anchor_data_file.close()
    train_forward_dynamics=True
    model_type= settings["model_type"]
    directory= getDataDirectory(settings)
        
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    if (settings['train_forward_dynamics']):
        if "." in settings['forward_dynamics_model_type']:
            ### convert . to / and copy file over
            file_name = settings['forward_dynamics_model_type']
            k = file_name.rfind(".")
            file_name = file_name[:k]
            file_name_read = file_name.replace(".", "/")
            file_name_read = file_name_read + ".py"
            print ("model file name:", file_name)
            print ("os.path.basename(file_name): ", os.path.basename(file_name))
            file = open(file_name_read, 'r')
            out_file = open(directory+file_name+".py", 'w')
            out_file.write(file.read())
            file.close()
            out_file.close()
            
    discrete_actions = np.array(settings['discrete_actions'])
    num_actions= discrete_actions.shape[0] # number of rows
    rounds = settings["rounds"]
    epochs = settings["epochs"]
    # num_states=settings["num_states"]
    epsilon = settings["epsilon"]
    discount_factor=settings["discount_factor"]
    # max_reward=settings["max_reward"]
    reward_bounds = np.array([[-10.1],[0.0]])
    batch_size=settings["batch_size"]
    train_on_validation_set=settings["train_on_validation_set"]
    state_bounds = np.array(settings['state_bounds'])
    discrete_actions = np.array(settings['discrete_actions'])
    print ("Sim config file name: ", str(settings["sim_config_file"]))
    # c = characterSim.Configuration(str(settings["sim_config_file"]))
    # c = characterSim.Configuration("../data/epsilon0Config.ini")
    action_space_continuous=settings['action_space_continuous']
    # states2 = np.transpose(np.repeat([states], 2, axis=0))
    # print states2
    if action_space_continuous:
        action_bounds = np.array(settings["action_bounds"], dtype=float)
    
    if action_space_continuous:
        experience = ExperienceMemory(len(state_bounds[0]), len(action_bounds[0]), settings['expereince_length'], continuous_actions=True, settings=settings)
    else:
        experience = ExperienceMemory(len(state_bounds[0]), 1, settings['expereince_length'])
    experience.setSettings(settings)
    file_name=directory+getAgentName()+"expBufferInit.hdf5"
    # experience.saveToFile(file_name)
    experience.loadFromFile(file_name)
    state_bounds = experience._state_bounds
    print ("Samples in experience: ", experience.samples())
    
    
    if (settings['train_forward_dynamics']):
        if ( settings['forward_dynamics_model_type'] == "SingleNet"):
            print ("Creating forward dynamics network: Using single network model")
            model = createRLAgent(settings['agent_name'], state_bounds, discrete_actions, reward_bounds, settings)
            forwardDynamicsModel = createForwardDynamicsModel(settings, state_bounds, action_bounds, None, None, agentModel=model)
            # forwardDynamicsModel = model
        else:
            print ("Creating forward dynamics network")
            # forwardDynamicsModel = ForwardDynamicsNetwork(state_length=len(state_bounds[0]),action_length=len(action_bounds[0]), state_bounds=state_bounds, action_bounds=action_bounds, settings_=settings)
            forwardDynamicsModel = createForwardDynamicsModel(settings, state_bounds, action_bounds, None, None, agentModel=None)
        if settings['visualize_learning']:
            from NNVisualize import NNVisualize
            title = file_name = settings['forward_dynamics_model_type']
            k = title.rfind(".") + 1
            if (k > len(title)): ## name does not contain a .
                k = 0 
            file_name = file_name[k:]
            nlv = NNVisualize(title=str("Forward Dynamics Model") + " with " + str(file_name))
            nlv.setInteractive()
            nlv.init()
    if (settings['train_reward_predictor']):
        if settings['visualize_learning']:
            rewardlv = NNVisualize(title=str("Reward Model") + " with " + str(settings["model_type"]), settings=settings)
            rewardlv.setInteractive()
            rewardlv.init()
    
    # experience = ExperienceMemory(len(state_bounds[0]), len(action_bounds[0]), experience_length, continuous_actions=True)
    """
    for i in range(experience_length):
        action_ = np.array([actions[i]])
        state_ = np.array([states[i]])
        # print "Action: " + str([actions[i]])
        experience.insert(norm_state(state_, state_bounds), norm_action(action_, action_bounds),
                           norm_state(state_, state_bounds), norm_reward(np.array([0]), reward_bounds))
    """
    trainData = {}
    trainData["mean_reward"]=[]
    trainData["std_reward"]=[]
    trainData["mean_bellman_error"]=[]
    trainData["std_bellman_error"]=[]
    trainData["mean_discount_error"]=[]
    trainData["std_discount_error"]=[]
    trainData["mean_forward_dynamics_loss"]=[]
    trainData["std_forward_dynamics_loss"]=[]
    trainData["mean_forward_dynamics_reward_loss"]=[]
    trainData["std_forward_dynamics_reward_loss"]=[]
    trainData["mean_eval"]=[]
    trainData["std_eval"]=[]
    # dynamicsLosses=[]
    best_dynamicsLosses=1000000
    _states, _actions, _result_states, _rewards, _falls, _G_ts, exp_actions__ = experience.get_batch(batch_size)
    """
    _states = theano.shared(np.array(_states, dtype=theano.config.floatX))
    _actions = theano.shared(np.array(_actions, dtype=theano.config.floatX))
    _result_states = theano.shared(np.array(_result_states, dtype=theano.config.floatX))
    _rewards = theano.shared(np.array(_rewards, dtype=theano.config.floatX))
    """
    forwardDynamicsModel.setData(_states, _actions, _result_states)
    for round_ in range(rounds):
        t0 = time.time()
        for epoch in range(epochs):
            _states, _actions, _result_states, _rewards, _falls, _G_ts, exp_actions__ = experience.get_batch(batch_size)
            # print _actions 
            # dynamicsLoss = forwardDynamicsModel.train(states=_states, actions=_actions, result_states=_result_states)
            # forwardDynamicsModel.setData(_states, _actions, _result_states)
            dynamicsLoss = forwardDynamicsModel.train(_states, _actions, _result_states, _rewards)
            # dynamicsLoss = forwardDynamicsModel._train()
        t1 = time.time()
        if (round_ % settings['plotting_update_freq_num_rounds']) == 0:
            dynamicsLoss_ = forwardDynamicsModel.bellman_error(_states, _actions, _result_states, _rewards)
            # dynamicsLoss_ = forwardDynamicsModel.bellman_error((_states), (_actions), (_result_states))
            if ( settings['use_stochastic_forward_dynamics'] ):
                dynamicsLoss = np.mean(dynamicsLoss_)
            else:
                dynamicsLoss = np.mean(np.fabs(dynamicsLoss_))
            if (settings['train_reward_predictor']):
                dynamicsRewardLoss_ = forwardDynamicsModel.reward_error(_states, _actions, _result_states, _rewards)
                dynamicsRewardLoss = np.mean(np.fabs(dynamicsRewardLoss_))
                # dynamicsRewardLosses.append(dynamicsRewardLoss)
                dynamicsRewardLosses = dynamicsRewardLoss
            if (settings['train_forward_dynamics'] and ((round_ % settings['plotting_update_freq_num_rounds']) == 0)):
                # dynamicsLosses.append(dynamicsLoss)
                mean_dynamicsLosses = dynamicsLoss
                std_dynamicsLosses = np.std((dynamicsLoss_))
                if (settings['train_forward_dynamics']):
                    trainData["mean_forward_dynamics_loss"].append(mean_dynamicsLosses)
                    trainData["std_forward_dynamics_loss"].append(std_dynamicsLosses)
                print ("Round: " + str(round_) + " Epoch: " + str(epoch) + " ForwardPredictionLoss: " + str(dynamicsLoss) + " in " + str(datetime.timedelta(seconds=(t1-t0))) + " seconds")
                # print ("State Bounds: ", forwardDynamicsModel.getStateBounds(), " exp: ", experience.getStateBounds())
                # print ("Action Bounds: ", forwardDynamicsModel.getActionBounds(), " exp: ", experience.getActionBounds())
                # print (str(datetime.timedelta(seconds=(t1-t0))))
                if (settings['visualize_learning']):
                    nlv.updateLoss(np.array(trainData["mean_forward_dynamics_loss"]), np.array(trainData["std_forward_dynamics_loss"]))
                    nlv.redraw()
                    nlv.setInteractiveOff()
                    nlv.saveVisual(directory+"trainingGraphNN")
                    nlv.setInteractive()
            if (settings['train_reward_predictor']):
                mean_dynamicsRewardLosses = np.mean(dynamicsRewardLoss)
                std_dynamicsRewardLosses = np.std(dynamicsRewardLoss_)
                dynamicsRewardLosses = []
                trainData["mean_forward_dynamics_reward_loss"].append(mean_dynamicsRewardLosses)
                trainData["std_forward_dynamics_reward_loss"].append(std_dynamicsRewardLosses)
            if (settings['train_reward_predictor'] and settings['visualize_learning']):
                rewardlv.updateLoss(np.array(trainData["mean_forward_dynamics_reward_loss"]), np.array(trainData["std_forward_dynamics_reward_loss"]))
                rewardlv.redraw()
                rewardlv.setInteractiveOff()
                rewardlv.saveVisual(directory+"rewardTrainingGraph")
                rewardlv.setInteractive()
        
        if (round_ % settings['saving_update_freq_num_rounds']) == 0:
            if mean_dynamicsLosses < best_dynamicsLosses:
                best_dynamicsLosses = mean_dynamicsLosses
                print ("Saving BEST current forward dynamics model: " + str(best_dynamicsLosses))
                file_name_dynamics=directory+"forward_dynamics_"+"_Best_pretrain.pkl"
                f = open(file_name_dynamics, 'wb')
                dill.dump(forwardDynamicsModel, f)
                f.close()
            
            if settings['save_trainData']:
                fp = open(directory+"FD_trainingData_" + str(settings['agent_name']) + ".json", 'w')
                # print ("Train data: ", trainData)
                ## because json does not serialize np.float32 
                for key in trainData:
                    trainData[key] = [float(i) for i in trainData[key]]
                json.dump(trainData, fp)
                fp.close()
                # draw data
        # print "Error: " + str(error)
    

if __name__ == '__main__':
    
    trainForwardDynamics(sys.argv[1])
    