# import theano
# from theano import tensor as T
import numpy as np
# import lasagne
import sys
import copy
sys.path.append('../')
sys.path.append('../model/')
from model.ModelUtil import *
from algorithm.AlgorithmInterface import AlgorithmInterface
import dill

# from actor.ActorInterface import *



def printPolicyInfo(settings_file_name):

    settings = getSettings(settings_file_name)
    # settings['shouldRender'] = True
    import os    
    os.environ['THEANO_FLAGS'] = "mode=FAST_RUN,device="+settings['training_processor_type']+",floatX="+settings['float_type']
    
    from util.SimulationUtil import validateSettings, createEnvironment, createRLAgent, createActor
    from util.SimulationUtil import getDataDirectory, createForwardDynamicsModel
    from util.ExperienceMemory import ExperienceMemory
    from model.LearningAgent import LearningAgent, LearningWorker
    from RLVisualize import RLVisualize
    from NNVisualize import NNVisualize
    
    model_type= settings["model_type"]
    directory= getDataDirectory(settings)
    rounds = settings["rounds"]
    epochs = settings["epochs"]
    # num_states=settings["num_states"]
    epsilon = settings["epsilon"]
    discount_factor=settings["discount_factor"]
    # max_reward=settings["max_reward"]
    batch_size=settings["batch_size"]
    state_bounds = np.array(settings['state_bounds'])
    action_space_continuous=settings["action_space_continuous"]  
    discrete_actions = np.array(settings['discrete_actions'])
    num_actions= discrete_actions.shape[0]
    reward_bounds=np.array(settings["reward_bounds"])
    action_space_continuous=settings['action_space_continuous']
    if action_space_continuous:
        action_bounds = np.array(settings["action_bounds"], dtype=float)
    
    print ("Sim config file name: " + str(settings["sim_config_file"]))
    
    ### Using a wrapper for the type of actor now
    if action_space_continuous:
        experience = ExperienceMemory(len(state_bounds[0]), len(action_bounds[0]), settings['expereince_length'], continuous_actions=True, settings=settings)
    else:
        experience = ExperienceMemory(len(state_bounds[0]), 1, settings['expereince_length'])
    # actor = ActorInterface(discrete_actions)
    actor = createActor(str(settings['environment_type']),settings, experience)
    masterAgent = LearningAgent(n_in=len(state_bounds[0]), n_out=len(action_bounds[0]), state_bounds=state_bounds, 
                              action_bounds=action_bounds, reward_bound=reward_bounds, settings_=settings)
    
    # c = characterSim.Configuration("../data/epsilon0Config.ini")
    file_name=directory+getAgentName()+"_Best.pkl"
    # file_name=directory+getAgentName()+".pkl"
    f = open(file_name, 'rb')
    model = dill.load(f)
    f.close()
    
    if (settings['train_forward_dynamics']):
        file_name_dynamics=directory+"forward_dynamics_"+"_Best.pkl"
        # file_name=directory+getAgentName()+".pkl"
        f = open(file_name_dynamics, 'rb')
        forwardDynamicsModel = dill.load(f)
        f.close()
    
    if ( settings["use_transfer_task_network"] ):
        task_directory = getTaskDataDirectory(settings)
        file_name=directory+getAgentName()+"_Best.pkl"
        f = open(file_name, 'rb')
        taskModel = dill.load(f)
        f.close()
        # copy the task part from taskModel to model
        print ("Transferring task portion of model.")
        model.setTaskNetworkParameters(taskModel)

    # this is the process that selects which game to play
    
    exp = createEnvironment(str(settings["sim_config_file"]), str(settings['environment_type']), settings, render=True)
    if (settings['train_forward_dynamics']):
        # actor.setForwardDynamicsModel(forwardDynamicsModel)
        forwardDynamicsModel.setActor(actor)
        masterAgent.setForwardDynamics(forwardDynamicsModel)
        # forwardDynamicsModel.setEnvironment(exp)
    # actor.setPolicy(model)
    exp.setActor(actor)
    exp.getActor().init()   
    exp.init()
    exp.generateValidationEnvironmentSample(0)
    expected_value_viz=None
    if (settings['visualize_expected_value']):
        expected_value_viz = NNVisualize(title=str("Expected Value") + " with " + str(settings["model_type"]), settings=settings)
        expected_value_viz.setInteractive()
        expected_value_viz.init()
        criticLosses = []
        
    masterAgent.setSettings(settings)
    masterAgent.setExperience(experience)
    masterAgent.setPolicy(model)
    
    
    print("State Bounds:", masterAgent.getStateBounds())
    print("Action Bounds:", masterAgent.getActionBounds())
    
    
if __name__ == "__main__":
    
    printPolicyInfo(sys.argv[1])
