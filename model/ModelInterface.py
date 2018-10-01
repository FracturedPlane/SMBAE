import theano
from theano import tensor as T
import numpy as np
import lasagne
import sys
sys.path.append('../')
from model.ModelUtil import *

# For debugging
# theano.config.mode='FAST_COMPILE'

def getActivationType(type_name):
    if ((type_name == 'leaky_rectify')):
        activation_type = lasagne.nonlinearities.leaky_rectify
    elif (type_name == 'relu'):
        activation_type = lasagne.nonlinearities.rectify
    elif (type_name == 'tanh'):
        activation_type = lasagne.nonlinearities.tanh
    elif ( type_name == 'linear'):
        activation_type = lasagne.nonlinearities.linear
    elif (type_name == 'sigmoid'):
        activation_type = lasagne.nonlinearities.sigmoid
    elif (type_name == 'softplus'):
        activation_type = theano.tensor.nnet.softplus
    else:
        print("Activation type: ", type_name, " not recognized")
    return activation_type

class ModelInterface(object):
    
    def __init__(self, n_in, n_out, state_bounds, action_bounds, reward_bound, settings_):

        # super(DeepCACLADropout,self).__init__(n_in, n_out, state_bounds, action_bounds, reward_bound, settings_)
        self._batch_size=settings_['batch_size']
        self._state_length = n_in
        self._action_length = n_out
        self._settings = settings_
        # data types for model
        self._dropout_p=settings_['dropout_p']
        
        ### Get a type of activation to use
        self._activation_type=lasagne.nonlinearities.leaky_rectify
        self._policy_activation_type=lasagne.nonlinearities.leaky_rectify
        if ("activation_type" in settings_ ):
            self._activation_type = getActivationType(settings_['activation_type'])
            self._policy_activation_type = self._activation_type
        if ("policy_activation_type" in settings_ ):
            self._policy_activation_type = getActivationType(settings_['policy_activation_type'])
            
        self._last_policy_layer_activation_type = lasagne.nonlinearities.linear
        if ('last_policy_layer_activation_type' in settings_ ):
            self._last_policy_layer_activation_type = getActivationType(settings_['last_policy_layer_activation_type'])
        
        self._last_std_policy_layer_activation_type = theano.tensor.nnet.softplus
        if ('_last_std_policy_layer_activation_type' in settings_ ):
            self._last_std_policy_layer_activation_type = getActivationType(settings_['_last_std_policy_layer_activation_type'])

        self._last_critic_layer_activation_type = lasagne.nonlinearities.linear
        if ('last_critic_layer_activation_type' in settings_ and (settings_['last_critic_layer_activation_type']) == 'linear'):
            self._last_critic_layer_activation_type = getActivationType(settings_['last_critic_layer_activation_type'])
        
    def getNetworkParameters(self):
        pass
    
    def setNetworkParameters(self, params):
        pass
    
    def getActorNetwork(self):
        """
            The output of this should be a list of layers...
        """
        return self._actor
    
    def getActorNetworkTaskPart(self):
        """
            The output of this should be a list of layers...
        """
        return self._actor_task_part
    
    def getCriticNetwork(self):
        """
            The output of this should be a list of layers...
        """
        return self._critic
    
    def getCriticNetworkTaskPart(self):
        """
            The output of this should be a list of layers...
        """
        return self._critic_task_part
    
    def getActorNetworkMergeLayer(self):
        return self._actor_merge_layer
    
    def getActorNetworkMergeLayers(self):
        """
            The layer should be the layer for which two layers may be combined
        """
        ### Should be one layer
        layers = lasagne.layers.get_all_layers(self.getActorNetworkMergeLayer(), treat_as_input=[self.getActorNetworkMergeLayer()])
        for i in range(0,len(layers)):
            print ("Actor merge layers[", i,"]: ", layers[i].W.get_value().shape)
        return layers
    
    def getCriticNetworkMergeLayer(self):
        return self._critic_merge_layer
        
    def getCriticNetworkMergeLayers(self):
        """
            The layer should be the layer for which two layers may be combined
            
        """
        layers = lasagne.layers.get_all_layers(self.getCriticNetworkMergeLayer(), treat_as_input=[self.getCriticNetworkMergeLayer()])
        for i in range(0,len(layers)):
            print ("Critic merge layers[", i,"]: ", layers[i].W.get_value().shape)
        return layers
    
    
    def getActorNetworkAgentPart(self):
        """
            The output of this should be a list of layers...
        """
        return self._actor_agent_part
    
    def getCriticNetworkAgentPart(self):
        """
            The output of this should be a list of layers...
        """
        return self._critic_agent_part
    
    def getActorNetworkCombinedPart(self):
        """
            The output of this should be a list of layers...
        """
        # all_paramsActA = lasagne.layers.helper.get_all_param_values(self.getActorNetwork())
        # combinedParams = all_paramsActA[-self._num_final_layers:]
        layers = lasagne.layers.get_all_layers(self.getActorNetwork(), treat_as_input=[self.getActorNetworkMergeLayer()])
        ## drop first merge layer
        layers = layers[1:]
        for i in range(0,len(layers)):
            print ("Actor layers[", i,"]: ", layers[i].W.get_value().shape)
        return layers
    
    def getCriticNetworkCombinedPart(self):
        """
            The output of this should be a list of layers...
        """
        # all_params = lasagne.layers.helper.get_all_param_values(self.getCriticNetwork())
        
        layers = lasagne.layers.get_all_layers(self.getCriticNetwork(), treat_as_input=[self.getCriticNetworkMergeLayer()])
        ## drop first merge layer
        layers = layers[1:]
        
        for i in range(0,len(layers)):
            print ("Critic layers[", i,"]: ", layers[i].W.get_value().shape)
        
        # combinedParams = all_params[-6:]
        return layers
    
    def getForwardDynamicsNetwork(self):
        """
            The output of this should be a list of layers...
        """
        return self._forward_dynamics_net
    def getRewardNetwork(self):
        """
            The output of this should be a list of layers...
        """
        return self._reward_net
    
    ### Setting network input values ###    
    def setStates(self, states):
        self._states_shared.set_value(states)
    def setActions(self, actions):
        self._actions_shared.set_value(actions)
    def setResultStates(self, resultStates):
        self._next_states_shared.set_value(resultStates)
    def setRewards(self, rewards):
        self._rewards_shared.set_value(rewards)
    def setTargets(self, targets):
        self._targets_shared.set_value(targets)
        
    ### Setting network input values ###    
    def getStateValues(self):
        return self._states_shared.get_value()
    def getActionValues(self):
        return self._actions_shared.get_value()
    def getResultStateValues(self):
        return self._next_states_shared.get_value()
    def getRewardValues(self):
        return self._rewards_shared.get_value()
    def getTargetValues(self):
        return self._target_shared.get_value()
    
    ####### Getting the shared variables to set values. #######  
    def getStates(self):
        return self._states_shared
    def getActions(self):
        return self._actions_shared
    def getResultStates(self):
        return self._next_states_shared
    def getRewards(self):
        return self._rewards_shared
    def getTargets(self):
        return self._targets_shared
    
    def getSettings(self):
        return self._settings
    
    ######### Symbolic Variables ######
    def getStateSymbolicVariable(self):
        return self._State
    def getActionSymbolicVariable(self):
        return self._Action
    def getResultStateSymbolicVariable(self):
        return self._ResultState
    def getRewardSymbolicVariable(self):
        return self._Reward
    def getTargetsSymbolicVariable(self):
        return self._Target
