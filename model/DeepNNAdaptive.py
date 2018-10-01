import theano
from theano import tensor as T
import numpy as np
import lasagne
import sys
sys.path.append('../')
from model.ModelUtil import *

# For debugging
# theano.config.mode='FAST_COMPILE'
from model.ModelInterface import ModelInterface

class DeepNNAdaptive(ModelInterface):
    
    def __init__(self, n_in, n_out, state_bounds, action_bounds, reward_bound, settings_):

        super(DeepNNAdaptive,self).__init__(n_in, n_out, state_bounds, action_bounds, reward_bound, settings_)
        self._dropout_p=settings_['dropout_p']
        
        # data types for model
        self._State = T.matrix("State")
        self._State.tag.test_value = np.random.rand(self._batch_size,self._state_length)
        self._ResultState = T.matrix("ResultState")
        self._ResultState.tag.test_value = np.random.rand(self._batch_size,self._state_length)
        self._Reward = T.col("Reward")
        self._Reward.tag.test_value = np.random.rand(self._batch_size,1)
        self._Action = T.matrix("Action")
        self._Action.tag.test_value = np.random.rand(self._batch_size, self._action_length)
        # create a small convolutional neural network
        stateInput = lasagne.layers.InputLayer((None, self._state_length), self._State)
        self._stateInputVar = stateInput.input_var
        actionInput = lasagne.layers.InputLayer((None, self._action_length), self._Action)
        self._actionInputVar = actionInput.input_var

        networkAct = stateInput        
        layer_sizes = self._settings['policy_network_layer_sizes']        
        for i in range(len(layer_sizes)):
            
            if (layer_sizes[i] == 'agent_part'):
                self._actor_agent_part = networkAct
            if (layer_sizes[i] == 'merge_part'):
                self._actor_merge_layer = networkAct
            else:
                networkAct = lasagne.layers.DenseLayer(
                        networkAct, num_units=layer_sizes[i],
                        nonlinearity=self._policy_activation_type)

        
        self._actor = lasagne.layers.DenseLayer(
                networkAct, num_units=self._action_length,
                nonlinearity=self._last_policy_layer_activation_type)
        # self._b_o = init_b_weights((n_out,))
        if (self._settings['use_stocastic_policy']):
            with_std = lasagne.layers.DenseLayer(
                    networkAct, num_units=self._action_length,
                    ### Reduce the initial size of std
                    W=lasagne.init.GlorotUniform(gain=0.01),
                    nonlinearity=self._last_std_policy_layer_activation_type)
            self._actor = lasagne.layers.ConcatLayer([self._actor, with_std], axis=1)
        
        
        if ( settings_['agent_name'] == 'algorithm.DPG.DPG' 
             or settings_['agent_name'] == 'algorithm.QProp.QProp'
             ):
            
            if ( ('train_extra_value_function' in settings_ and (settings_['train_extra_value_function']) )
                 or (settings_['agent_name'] == 'algorithm.QProp.QProp') # A must for Q-Prop 
                 ):
                ## create an extra value function
                layer_sizes = self._settings['critic_network_layer_sizes']
                # print ("Network layer sizes: ", layer_sizes)
                network = stateInput
                for i in range(len(layer_sizes)):

                    if ( (layer_sizes[i] == 'agent_part')
                         or (layer_sizes[i] == 'merge_part')
                         or (layer_sizes[i] == 'integrate_actor_part') 
                         ):
                        continue
                    network = lasagne.layers.DenseLayer(
                                network, num_units=layer_sizes[i],
                                nonlinearity=self._activation_type)

                self._value_function = lasagne.layers.DenseLayer(
                        network, num_units=1,
                        nonlinearity=lasagne.nonlinearities.linear)
                
            if ( not ( "integrate_actor_part" in layer_sizes)):
                network = lasagne.layers.ConcatLayer([stateInput, actionInput])
        elif ( 'ppo_use_seperate_nets' in settings_ and (settings_['ppo_use_seperate_nets'] == False) ):
            network = networkAct
        else:
            network = stateInput
            
        layer_sizes = self._settings['critic_network_layer_sizes']
        print ("Network layer sizes: ", layer_sizes)
        for i in range(len(layer_sizes)):
            
            if (layer_sizes[i] == 'agent_part'):
                self._critic_agent_part = network
            if (layer_sizes[i] == 'merge_part'):
                self._critic_merge_layer = network
            if (layer_sizes[i] == 'integrate_actor_part'):
                # self._critic_merge_layer = network
                network = lasagne.layers.ConcatLayer([network, actionInput])
            else:
                network = lasagne.layers.DenseLayer(
                        network, num_units=layer_sizes[i],
                        nonlinearity=self._activation_type)
        
        self._critic = lasagne.layers.DenseLayer(
                network, num_units=1,
                nonlinearity=self._last_critic_layer_activation_type)
        # self._b_o = init_b_weights((n_out,))
          # print "Initial W " + str(self._w_o.get_value()) 
        
        self._states_shared = theano.shared(
            np.zeros((self._batch_size, self._state_length),
                     dtype=theano.config.floatX))

        self._next_states_shared = theano.shared(
            np.zeros((self._batch_size, self._state_length),
                     dtype=theano.config.floatX))

        self._rewards_shared = theano.shared(
            np.zeros((self._batch_size, 1), dtype=theano.config.floatX),
            broadcastable=(False, True))

        self._actions_shared = theano.shared(
            np.zeros((self._batch_size, self._action_length), dtype=theano.config.floatX),
            )
