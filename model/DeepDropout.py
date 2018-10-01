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

class DeepDropout(ModelInterface):
    
    def __init__(self, n_in, n_out, state_bounds, action_bounds, reward_bound, settings_):

        super(DeepDropout,self).__init__(n_in, n_out, state_bounds, action_bounds, reward_bound, settings_)
        
        self._dropout_p=settings_['dropout_p']
        # data types for model
        # data types for model
        self._State = T.dmatrix("State")
        self._State.tag.test_value = np.random.rand(self._batch_size,self._state_length)
        self._ResultState = T.dmatrix("ResultState")
        self._ResultState.tag.test_value = np.random.rand(self._batch_size,self._state_length)
        self._Reward = T.col("Reward")
        self._Reward.tag.test_value = np.random.rand(self._batch_size,1)
        self._Action = T.dmatrix("Action")
        self._Action.tag.test_value = np.random.rand(self._batch_size, self._action_length)
        # create a small convolutional neural network
        inputLayerA = lasagne.layers.InputLayer((None, self._state_length), self._State)
        inputLayerA = lasagne.layers.DropoutLayer(inputLayerA, p=self._dropout_p, rescale=True)
        
        l_hid1A = lasagne.layers.DenseLayer(
                inputLayerA, num_units=128,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        l_hid1A = lasagne.layers.DropoutLayer(l_hid1A, p=self._dropout_p, rescale=True)
        
        l_hid2A = lasagne.layers.DenseLayer(
                l_hid1A, num_units=64,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        l_hid2A = lasagne.layers.DropoutLayer(l_hid2A, p=self._dropout_p, rescale=True)
        
        l_hid3A = lasagne.layers.DenseLayer(
                l_hid2A, num_units=32,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        l_hid3A = lasagne.layers.DropoutLayer(l_hid3A, p=self._dropout_p, rescale=True)
    
        self._critic = lasagne.layers.DenseLayer(
                l_hid3A, num_units=1,
                nonlinearity=lasagne.nonlinearities.linear)
        # self._b_o = init_b_weights((n_out,))
        inputLayerActA = lasagne.layers.InputLayer((None, self._state_length), self._State)
        inputLayerActA = lasagne.layers.DropoutLayer(inputLayerActA, p=self._dropout_p, rescale=True)
        
        l_hid1ActA = lasagne.layers.DenseLayer(
                inputLayerActA, num_units=128,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        l_hid1ActA = lasagne.layers.DropoutLayer(l_hid1ActA, p=self._dropout_p, rescale=True)
        
        l_hid2ActA = lasagne.layers.DenseLayer(
                l_hid1ActA, num_units=64,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        l_hid2ActA = lasagne.layers.DropoutLayer(l_hid2ActA, p=self._dropout_p, rescale=True)
        
        l_hid3ActA = lasagne.layers.DenseLayer(
                l_hid2ActA, num_units=32,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        l_hid3ActA = lasagne.layers.DropoutLayer(l_hid3ActA, p=self._dropout_p, rescale=True)
    
        self._actor = lasagne.layers.DenseLayer(
                l_hid3ActA, num_units=self._action_length,
                nonlinearity=lasagne.nonlinearities.linear)
        # self._b_o = init_b_weights((n_out,))
        
        
          # print ("Initial W " + str(self._w_o.get_value()) )
        
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
        
