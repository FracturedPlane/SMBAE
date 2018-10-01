import theano
from theano import tensor as T
import numpy as np
import lasagne
import sys
sys.path.append('../')
from model.ModelUtil import *
from util.nn import weight_norm



# elu
def elu_mine(x):
    return theano.tensor.switch(x > 0, x, theano.tensor.expm1(x))

# For debugging
# theano.config.mode='FAST_COMPILE'
from model.ModelInterface import ModelInterface

class ForwardDynamicsDenseSingleNetworkDropout(ModelInterface):
    
    def __init__(self, state_length, action_length, state_bounds, action_bounds, settings_):

        super(ForwardDynamicsDenseSingleNetworkDropout,self).__init__(state_length, action_length, state_bounds, action_bounds, 0, settings_)
        
        batch_size=32
        # data types for model
        self._State = T.matrix("State")
        self._State.tag.test_value = np.random.rand(batch_size,self._state_length)
        self._ResultState = T.matrix("ResultState")
        self._ResultState.tag.test_value = np.random.rand(batch_size,self._state_length)
        self._Reward = T.col("Reward")
        self._Reward.tag.test_value = np.random.rand(self._batch_size,1)
        self._Action = T.matrix("Action")
        self._Action.tag.test_value = np.random.rand(batch_size, self._action_length)
        # create a small convolutional neural network
        stateInput = lasagne.layers.InputLayer((None, self._state_length), self._State)
        self._stateInputVar = stateInput.input_var
        actionInput = lasagne.layers.InputLayer((None, self._action_length), self._Action)
        self._actionInputVar = actionInput.input_var
        input = lasagne.layers.ConcatLayer([stateInput, actionInput])
        
        # network = lasagne.layers.DropoutLayer(input, p=self._dropout_p, rescale=True)
        
        network = lasagne.layers.DenseLayer(
                input, num_units=128,
                nonlinearity=elu_mine)
        network = weight_norm(network)
        network = lasagne.layers.DropoutLayer(network, p=self._dropout_p, rescale=True)
        layersAct = [network]
        
        network = lasagne.layers.DenseLayer(
                network, num_units=64,
                nonlinearity=elu_mine)
        network = weight_norm(network)
        network = lasagne.layers.DropoutLayer(network, p=self._dropout_p, rescale=True)
        layersAct.append(network)
        network = lasagne.layers.ConcatLayer([layersAct[1], layersAct[0]])
        
        network = lasagne.layers.DenseLayer(
                network, num_units=32,
                nonlinearity=elu_mine)
        network = weight_norm(network)
        network = lasagne.layers.DropoutLayer(network, p=self._dropout_p, rescale=True)
        layersAct.append(network)
        network = lasagne.layers.ConcatLayer([layersAct[2], layersAct[1], layersAct[0]])
        ## This can be used to model the reward function
        self._reward_net = lasagne.layers.DenseLayer(
                network, num_units=1,
                nonlinearity=lasagne.nonlinearities.linear)
                # print ("Initial W " + str(self._w_o.get_value()) )
                
                
        self._forward_dynamics_net = lasagne.layers.DenseLayer(
                network, num_units=self._state_length,
                nonlinearity=lasagne.nonlinearities.linear)
                # print ("Initial W " + str(self._w_o.get_value()) )
                
        if (('use_stochastic_forward_dynamics' in self._settings) and 
            self._settings['use_stochastic_forward_dynamics']):
            with_std = lasagne.layers.DenseLayer(
                    network, num_units=self._state_length,
                    nonlinearity=theano.tensor.nnet.softplus)
            self._forward_dynamics_net = lasagne.layers.ConcatLayer([self._forward_dynamics_net, with_std], axis=1)
                
        self._states_shared = theano.shared(
            np.zeros((batch_size, self._state_length),
                     dtype=theano.config.floatX))

        self._next_states_shared = theano.shared(
            np.zeros((batch_size, self._state_length),
                     dtype=theano.config.floatX))

        self._actions_shared = theano.shared(
            np.zeros((batch_size, self._action_length), dtype=theano.config.floatX),
            )
        
        self._rewards_shared = theano.shared(
            np.zeros((self._batch_size, 1), dtype=theano.config.floatX),
            broadcastable=(False, True))
        
