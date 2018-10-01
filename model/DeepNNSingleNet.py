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

class DeepNNSingleNet(ModelInterface):
    
    def __init__(self, n_in, n_out, state_bounds, action_bounds, reward_bound, settings_):

        super(DeepNNSingleNet,self).__init__(n_in, n_out, state_bounds, action_bounds, reward_bound, settings_)
        
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
        
        activation_type=lasagne.nonlinearities.leaky_rectify
        if ("activation_type" in settings_ and (settings_['activation_type'] == 'leaky_rectify')):
            activation_type = lasagne.nonlinearities.leaky_rectify
        elif ("activation_type" in settings_ and (settings_['activation_type'] == 'relu')):
            activation_type = lasagne.nonlinearities.rectify
        elif ("activation_type" in settings_ and (settings_['activation_type'] == 'tanh')):
            activation_type = lasagne.nonlinearities.tanh
        elif ("activation_type" in settings_ and (settings_['activation_type'] == 'linear')):
            activation_type = lasagne.nonlinearities.linear
            
        last_policy_layer_activation_type = lasagne.nonlinearities.tanh
        if ('last_policy_layer_activation_type' in settings_ and (settings_['last_policy_layer_activation_type']) == 'linear'):
            last_policy_layer_activation_type=lasagne.nonlinearities.linear
        if ("last_policy_layer_activation_type" in settings_ and (settings_['last_policy_layer_activation_type'] == 'leaky_rectify')):
            last_policy_layer_activation_type = lasagne.nonlinearities.leaky_rectify
        elif ("last_policy_layer_activation_type" in settings_ and (settings_['last_policy_layer_activation_type'] == 'relu')):
            last_policy_layer_activation_type = lasagne.nonlinearities.rectify
        elif ("last_policy_layer_activation_type" in settings_ and (settings_['last_policy_layer_activation_type'] == 'tanh')):
            last_policy_layer_activation_type = lasagne.nonlinearities.tanh
        
        """
        networkAct = lasagne.layers.DenseLayer(
                networkAct, num_units=256,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        """
        networkAct = lasagne.layers.DenseLayer(
                input, num_units=128,
                nonlinearity=activation_type)
        networkAct = lasagne.layers.DropoutLayer(networkAct, p=self._dropout_p, rescale=True)
        
        networkAct = lasagne.layers.DenseLayer(
                networkAct, num_units=64,
                nonlinearity=activation_type)
        networkAct = lasagne.layers.DropoutLayer(networkAct, p=self._dropout_p, rescale=True)
        
        networkMiddle = lasagne.layers.DenseLayer(
                networkAct, num_units=32,
                nonlinearity=activation_type)
        networkMiddle = lasagne.layers.DropoutLayer(networkMiddle, p=self._dropout_p, rescale=True)
        
        networkMiddleFD = lasagne.layers.ConcatLayer([networkMiddle, inputAction], axis=1)
        # networkMiddle = lasagne.layers.ConcatLayer([networkMiddle, actionInput], axis=1)
    
        self._actor = lasagne.layers.DenseLayer(
                networkMiddle, num_units=self._action_length,
                nonlinearity=last_policy_layer_activation_type)
        
        if (self._settings['use_stocastic_policy'] and ( not ( 'use_fixed_std' in self.getSettings() and ( self.getSettings()['use_fixed_std'])))):
            print ("Adding stochastic layer")
            with_std = lasagne.layers.DenseLayer(
                    networkMiddle, num_units=self._action_length,
                    nonlinearity=theano.tensor.nnet.softplus)
            self._actor = lasagne.layers.ConcatLayer([self._actor, with_std], axis=1)
        # self._b_o = init_b_weights((n_out,))
        else:
            print ("NOT Adding stochastic layer")
        if ( settings_['agent_name'] == 'algorithm.DPG.DPG'):
            
            
            input = lasagne.layers.ConcatLayer([networkMiddle, inputAction])

        network = lasagne.layers.DenseLayer(
                networkMiddle, num_units=16,
                nonlinearity=activation_type)
        network = lasagne.layers.DropoutLayer(network, p=self._dropout_p, rescale=True)

        self._critic = lasagne.layers.DenseLayer(
                network, num_units=1,
                nonlinearity=lasagne.nonlinearities.linear)        
        
        network = lasagne.layers.DenseLayer(
                networkMiddleFD, num_units=64,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        network = lasagne.layers.DropoutLayer(network, p=self._dropout_p, rescale=True)
        
        self._forward_dynamics_net = lasagne.layers.DenseLayer(
                network, num_units=self._state_length,
                nonlinearity=lasagne.nonlinearities.linear)
        
        network = lasagne.layers.DenseLayer(
                networkMiddleFD, num_units=32,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        network = lasagne.layers.DropoutLayer(network, p=self._dropout_p, rescale=True)
        
        self._reward_net = lasagne.layers.DenseLayer(
                network, num_units=1,
                nonlinearity=lasagne.nonlinearities.linear)
                # print ("Initial W " + str(self._w_o.get_value()) )
                
        self._encode_net = lasagne.layers.DenseLayer(
                networkMiddle, num_units=64,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        self._encode_net = lasagne.layers.DropoutLayer(self._encode_net, p=self._dropout_p, rescale=True)
        
        self._encode_net = lasagne.layers.DenseLayer(
                self._encode_net, num_units=128,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        self._encode_net = lasagne.layers.DropoutLayer(self._encode_net, p=self._dropout_p, rescale=True)
        
        self._encode_net = lasagne.layers.DenseLayer(
                self._encode_net, num_units=self._state_length,
                nonlinearity=lasagne.nonlinearities.linear)
        
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
    
    def getEncodeNet(self):
        return self._encode_net