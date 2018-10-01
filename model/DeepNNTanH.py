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

class DeepNNTanH(ModelInterface):
    
    def __init__(self, n_in, n_out, state_bounds, action_bounds, reward_bound, settings_):

        super(DeepNNTanH,self).__init__(n_in, n_out, state_bounds, action_bounds, reward_bound, settings_)
        
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
        input = lasagne.layers.InputLayer((None, self._state_length), self._State)
        self._stateInputVar = input.input_var
        inputAction = lasagne.layers.InputLayer((None, self._action_length), self._Action)
        self._actionInputVar = inputAction.input_var
        # self._b_o = init_b_weights((n_out,))
        # networkAct = lasagne.layers.InputLayer((None, self._state_length), self._State)
        
                
        """
        networkAct = lasagne.layers.DenseLayer(
                networkAct, num_units=256,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        """
        networkAct = input
        
        networkAct = lasagne.layers.DenseLayer(
                networkAct , num_units=128,
                nonlinearity=self._activation_type)
        
        networkAct = lasagne.layers.DenseLayer(
                networkAct, num_units=64,
                nonlinearity=self._activation_type)
        
        networkAct = lasagne.layers.DenseLayer(
                networkAct, num_units=32,
                nonlinearity=self._activation_type)
    
        self._actor = lasagne.layers.DenseLayer(
                networkAct, num_units=self._action_length,
                nonlinearity=self._last_policy_layer_activation_type)
        
        if (self._settings['use_stocastic_policy'] and ( not ( 'use_fixed_std' in self.getSettings() and ( self.getSettings()['use_fixed_std'])))):
            print ("Adding stochastic layer")
            with_std = lasagne.layers.DenseLayer(
                    networkAct, num_units=self._action_length,
                    nonlinearity=theano.tensor.nnet.softplus)
            self._actor = lasagne.layers.ConcatLayer([self._actor, with_std], axis=1)
        # self._b_o = init_b_weights((n_out,))
        else:
            print ("NOT Adding stochastic layer")
        if ( settings_['agent_name'] == 'algorithm.DPG.DPG'):
            
            if ('train_extra_value_function' in settings_ and (settings_['train_extra_value_function'])):
                ## create an extra value function
                network = lasagne.layers.DenseLayer(
                        input, num_units=128,
                        nonlinearity=self._activation_type)
                network = lasagne.layers.DropoutLayer(network, p=self._dropout_p, rescale=True)
                
                network = lasagne.layers.DenseLayer(
                        network, num_units=64,
                        nonlinearity=self._activation_type)
                network = lasagne.layers.DropoutLayer(network, p=self._dropout_p, rescale=True)
                """
                if ( settings_['agent_name'] == 'algorithm.DPG.DPG'):
                    network = lasagne.layers.ConcatLayer([network, inputAction])
                """
                network = lasagne.layers.DenseLayer(
                        network, num_units=32,
                        nonlinearity=self._activation_type)
                network = lasagne.layers.DropoutLayer(network, p=self._dropout_p, rescale=True)
                
                network = lasagne.layers.DenseLayer(
                        network, num_units=16,
                        nonlinearity=self._activation_type)
                network = lasagne.layers.DropoutLayer(network, p=self._dropout_p, rescale=True)
                """
                network = lasagne.layers.DenseLayer(
                        network, num_units=8,
                        nonlinearity=activation_type)
                network = lasagne.layers.DropoutLayer(network, p=self._dropout_p, rescale=True)
                """
                self._value_function = lasagne.layers.DenseLayer(
                        network, num_units=1,
                        nonlinearity=lasagne.nonlinearities.linear)
                
            ## put inputs together for DPG
            input = lasagne.layers.ConcatLayer([input, inputAction])
        network = input
        """
        network = lasagne.layers.DenseLayer(
                network, num_units=256,
                nonlinearity=activation_type)
        """
        
        network = lasagne.layers.DenseLayer(
                network, num_units=128,
                nonlinearity=self._activation_type)
        network = lasagne.layers.DropoutLayer(network, p=self._dropout_p, rescale=True)
        
        network = lasagne.layers.DenseLayer(
                network, num_units=64,
                nonlinearity=self._activation_type)
        network = lasagne.layers.DropoutLayer(network, p=self._dropout_p, rescale=True)
        """
        if ( settings_['agent_name'] == 'algorithm.DPG.DPG'):
            network = lasagne.layers.ConcatLayer([network, inputAction])
        """
        network = lasagne.layers.DenseLayer(
                network, num_units=32,
                nonlinearity=self._activation_type)
        network = lasagne.layers.DropoutLayer(network, p=self._dropout_p, rescale=True)
        
        network = lasagne.layers.DenseLayer(
                network, num_units=16,
                nonlinearity=self._activation_type)
        network = lasagne.layers.DropoutLayer(network, p=self._dropout_p, rescale=True)
        """
        network = lasagne.layers.DenseLayer(
                network, num_units=8,
                nonlinearity=activation_type)
        network = lasagne.layers.DropoutLayer(network, p=self._dropout_p, rescale=True)
        """
        self._critic = lasagne.layers.DenseLayer(
                network, num_units=1,
                nonlinearity=self._last_critic_layer_activation_type)
        
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
