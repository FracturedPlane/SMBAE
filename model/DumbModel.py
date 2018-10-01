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

class DumbModel(ModelInterface):
    
    def __init__(self, n_in, n_out, state_bounds, action_bounds, reward_bound, settings_):

        super(DumbModel,self).__init__(n_in, n_out, state_bounds, action_bounds, reward_bound, settings_)
        
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
        network = lasagne.layers.InputLayer((None, self._state_length), self._State)
        
        network = lasagne.layers.DenseLayer(
                network, num_units=8,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        
        network = lasagne.layers.DenseLayer(
                network, num_units=4,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        
        network = lasagne.layers.DenseLayer(
                network, num_units=2,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        
        network = lasagne.layers.DenseLayer(
                network, num_units=2,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
    
        self._critic = lasagne.layers.DenseLayer(
                network, num_units=1,
                nonlinearity=lasagne.nonlinearities.linear)
        # self._b_o = init_b_weights((n_out,))
        networkAct = lasagne.layers.InputLayer((None, self._state_length), self._State)
        
        networkAct = lasagne.layers.DenseLayer(
                networkAct, num_units=8,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        
        networkAct = lasagne.layers.DenseLayer(
                networkAct, num_units=4,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        
        networkAct = lasagne.layers.DenseLayer(
                networkAct, num_units=2,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
    
        self._actor = lasagne.layers.DenseLayer(
                networkAct, num_units=self._action_length,
                nonlinearity=lasagne.nonlinearities.linear)
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
