import theano
from theano import tensor as T
import numpy as np
import lasagne
from lasagne import layers
from lasagne import init
from lasagne import nonlinearities
from lasagne.utils import as_tuple
import sys
sys.path.append('../')
from model.ModelUtil import *


# For debugging
# theano.config.mode='FAST_COMPILE'
from model.ModelInterface import ModelInterface
from model.ForwardDynamicsCNN import *

class ForwardDynamicsCNNTile(ModelInterface):
    
    def __init__(self, state_length, action_length, state_bounds, action_bounds, settings_):

        super(ForwardDynamicsCNNTile,self).__init__(state_length, action_length, state_bounds, action_bounds, 0, settings_)
        
                # data types for model
        self._State = T.tensor4("State")
        self._State.tag.test_value = np.random.rand(self._batch_size, 1, 1, self._state_length)
        self._ResultState = T.dmatrix("ResultState")
        self._ResultState.tag.test_value = np.random.rand(self._batch_size, self._state_length)
        self._Action = T.dmatrix("Action")
        self._Action.tag.test_value = np.random.rand(self._batch_size, self._action_length)
        
        # self._b_o = init_b_weights((n_out,))
        # networkAct = lasagne.layers.InputLayer((None, 1, 1, self._state_length), self._State)
        inputLayerState = lasagne.layers.InputLayer((None, 1, 1, self._state_length), self._State)
        inputLayerAction = lasagne.layers.InputLayer((None, self._action_length), self._Action)
        networkAct = lasagne.layers.Conv2DLayer(
            inputLayerState, num_filters=32, filter_size=(1,8),
            stride=(1,1),
            nonlinearity=lasagne.nonlinearities.leaky_rectify,
            W=lasagne.init.GlorotUniform())
        print ("Network Shape:", lasagne.layers.get_output_shape(networkAct))
        # network = lasagne.layers.MaxPool1DLayer(network, pool_size=3)
        
        networkAct = lasagne.layers.Conv2DLayer(
            networkAct, num_filters=16, filter_size=(1,4),
            stride=(1,1),
            nonlinearity=lasagne.nonlinearities.leaky_rectify,
            W=lasagne.init.GlorotUniform())
        print ("Network Shape:", lasagne.layers.get_output_shape(networkAct))
        # network = lasagne.layers.MaxPool1DLayer(network, pool_size=3)
        
        self._actor_task_part = networkAct
        """ 
        networkAct = lasagne.layers.Conv1DLayer(
            networkAct, num_filters=32, filter_size=4,
            nonlinearity=lasagne.nonlinearities.leaky_rectify,
            W=lasagne.init.GlorotUniform())
        
        
        networkAct = lasagne.layers.DenseLayer(
                networkAct, num_units=128,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        """
        # networkAct = lasagne.layers.ReshapeLayer(networkAct, (-1, 99))
        # networkAct = lasagne.layers.FlattenLayer(networkAct, 2)
        print ("Network Shape:", lasagne.layers.get_output_shape(networkAct))
        inputLayerAction = lasagne.layers.ReshapeLayer(inputLayerAction, (-1, 1, 1, 1))
        tiles = 16 # same as number of filters in previous CNN layer
        inputLayerAction = Repeat(inputLayerAction, tiles)
        print ("Action Network Shape:", lasagne.layers.get_output_shape(inputLayerAction))
        inputLayerAction = lasagne.layers.ReshapeLayer(inputLayerAction, (-1, tiles, 1, 1))
        print ("Action Network Shape:", lasagne.layers.get_output_shape(inputLayerAction))
        networkAct = lasagne.layers.ConcatLayer([networkAct, inputLayerAction], axis=3)
        print ("Network Shape:", lasagne.layers.get_output_shape(networkAct))
        
        # networkAct = lasagne.layers.DenseLayer(
        #         networkAct, num_units=64,
        #         nonlinearity=lasagne.nonlinearities.leaky_rectify)
        #print ("Network Shape:", lasagne.layers.get_output_shape(networkAct))
        # networkAct = lasagne.layers.ReshapeLayer(networkAct, (-1, 1, 1, 64))
        
        networkAct = Deconv2DLayer(
            networkAct, num_filters=16, filter_size=(1,4),
            stride=(1,1),
            nonlinearity=lasagne.nonlinearities.leaky_rectify)
        
        # network = lasagne.layers.MaxPool1DLayer(network, pool_size=3)
        print ("Network Shape:", lasagne.layers.get_output_shape(networkAct))
        networkAct = Deconv2DLayer(
            networkAct, num_filters=32, filter_size=(1,8),
            stride=(1,1),
            nonlinearity=lasagne.nonlinearities.leaky_rectify)
        # networkAct = lasagne.layers.ReshapeLayer(networkAct, (-1, 1, 1, 74))
        print ("Network Shape:", lasagne.layers.get_output_shape(networkAct))
        self._actor = lasagne.layers.DenseLayer(
                networkAct, num_units=self._state_length,
                nonlinearity=lasagne.nonlinearities.linear)
        # self._b_o = init_b_weights((n_out,))
        print ("Network Shape:", lasagne.layers.get_output_shape(self._actor))
        
        # self._actor = lasagne.layers.ReshapeLayer(self._actor, (-1, 1, 1, 208))
        
          # print ("Initial W " + str(self._w_o.get_value()) )
        
        self._states_shared = theano.shared(
            np.zeros((self._batch_size, 1, 1, self._state_length),
                     dtype=theano.config.floatX))

        self._next_states_shared = theano.shared(
            np.zeros((self._batch_size, self._state_length),
                     dtype=theano.config.floatX))

        self._actions_shared = theano.shared(
            np.zeros((self._batch_size, self._action_length), dtype=theano.config.floatX),
            )
        
    def setStates(self, states):
        """
            This is reshaped to work properly with a 1D convolution that likes 
            its input as (batch_size, channel, state_dimension)
            
            Parameters
        ----------
        states : a (batch_size, state_dimension) numpy array
        """
        states = np.array(states)
        states = np.reshape(states, (states.shape[0], 1, 1, states.shape[1]))
        self._states_shared.set_value(states)
    def setResultStates(self, resultStates):
        """
            This is reshaped to work properly with a 1D convolution that likes 
            its input as (batch_size, channel, state_dimension)
            
            Parameters
        ----------
        resultStates : a (batch_size, state_dimension) numpy array
        """
        # resultStates = np.array(resultStates)
        # resultStates = np.reshape(resultStates, (resultStates.shape[0], 1, 1, resultStates.shape[1]))
        self._next_states_shared.set_value(resultStates)
