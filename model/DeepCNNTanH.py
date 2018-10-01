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

class DeepCNNTanH(ModelInterface):
    
    def __init__(self, n_in, n_out, state_bounds, action_bounds, reward_bound, settings_):

        super(DeepCNNTanH,self).__init__(n_in, n_out, state_bounds, action_bounds, reward_bound, settings_)
        
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
        
        taskFeatures = lasagne.layers.SliceLayer(input, indices=slice(0, self._settings['num_terrain_features']), axis=1)
        # characterFeatures = lasagne.layers.SliceLayer(network, indices=slice(-(self._state_length-self._settings['num_terrain_features']), None), axis=1)
        characterFeatures = lasagne.layers.SliceLayer(input, indices=slice(self._settings['num_terrain_features'], self._state_length), axis=1)
        print ("taskFeatures Shape:", lasagne.layers.get_output_shape(taskFeatures))
        print ("characterFeatures Shape:", lasagne.layers.get_output_shape(characterFeatures))
        print ("State length: ", self._state_length)
        
        networkAct = lasagne.layers.ReshapeLayer(taskFeatures, (-1, 1, self._settings['num_terrain_features']))
        
        networkAct = lasagne.layers.Conv1DLayer(
            networkAct, num_filters=16, filter_size=8,
            nonlinearity=lasagne.nonlinearities.leaky_rectify)
        # network = weight_norm( network )
        
        # network = lasagne.layers.MaxPool1DLayer(network, pool_size=3)
        
        networkAct = lasagne.layers.Conv1DLayer(
            networkAct, num_filters=8, filter_size=4,
            nonlinearity=lasagne.nonlinearities.leaky_rectify,
            W=lasagne.init.GlorotUniform())
        """
        network = lasagne.layers.Conv1DLayer(
            network, num_filters=32, filter_size=4,
            nonlinearity=lasagne.nonlinearities.leaky_rectify)
        """
        self._actor_task_part = networkAct     
        
        networkAct_ = lasagne.layers.FlattenLayer(network, outdim=2)
        
        """
        networkAct = lasagne.layers.DenseLayer(
                networkAct, num_units=256,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        """
        """
        networkAct = lasagne.layers.DenseLayer(
                input, num_units=128,
                nonlinearity=self._activation_type)
        """
        """
        networkAct = lasagne.layers.DenseLayer(
                networkAct, num_units=256,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        """
        networkAct = lasagne.layers.DenseLayer(
                characterFeatures, num_units=128,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        
        networkAct = lasagne.layers.DenseLayer(
                networkAct, num_units=64,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        
        networkActMiddle = lasagne.layers.ConcatLayer([networkAct_, networkAct], axis=1)
        
        networkAct = lasagne.layers.DenseLayer(
                networkActMiddle, num_units=32,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        
        self._actor = lasagne.layers.DenseLayer(
                networkAct, num_units=self._action_length,
                nonlinearity=lasagne.nonlinearities.linear)
        # self._b_o = init_b_weights((n_out,))
        if (self._settings['use_stocastic_policy']):
            with_std = lasagne.layers.DenseLayer(
                    networkAct, num_units=self._action_length,
                    nonlinearity=theano.tensor.nnet.softplus)
            self._actor = lasagne.layers.ConcatLayer([self._actor, with_std], axis=1)
        
        
        if ( settings_['agent_name'] == 'algorithm.DPG.DPG'):
            input = lasagne.layers.ConcatLayer([input, inputAction])
        
        network = lasagne.layers.ReshapeLayer(taskFeatures, (-1, 1, self._settings['num_terrain_features']))
        
        network = lasagne.layers.Conv1DLayer(
            network, num_filters=16, filter_size=8,
            nonlinearity=lasagne.nonlinearities.leaky_rectify)
        # network = weight_norm( network )
        
        # network = lasagne.layers.MaxPool1DLayer(network, pool_size=3)
        
        network = lasagne.layers.Conv1DLayer(
            network, num_filters=8, filter_size=4,
            nonlinearity=lasagne.nonlinearities.leaky_rectify,
            W=lasagne.init.GlorotUniform())
        """
        network = lasagne.layers.Conv1DLayer(
            network, num_filters=32, filter_size=4,
            nonlinearity=lasagne.nonlinearities.leaky_rectify)
        """
        self._critic_task_part = network     
        
        network_ = lasagne.layers.FlattenLayer(network, outdim=2)
        
        """
        network = lasagne.layers.DenseLayer(
                network, num_units=256,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        """
        network = lasagne.layers.DenseLayer(
                characterFeatures, num_units=128,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        
        network = lasagne.layers.DenseLayer(
                network, num_units=64,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        
        networkMiddle = lasagne.layers.ConcatLayer([network_, network], axis=1)
        
        network = lasagne.layers.DenseLayer(
                network, num_units=32,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        
        network = lasagne.layers.DenseLayer(
                network, num_units=16,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
    
       
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
