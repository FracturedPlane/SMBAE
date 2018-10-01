
import theano
from theano import tensor as T
import numpy as np
# import lasagne
import sys
sys.path.append('../')
from model.ModelUtil import *
from keras.models import Sequential, Model
from keras.optimizers import SGD
from keras.layers import Input
from keras.layers.core import Dense, Dropout, Activation, Reshape, Flatten, Lambda
from keras.layers.convolutional import Conv1D
from keras.layers.merge import Concatenate
# from keras.utils.np_utils import to_categoricalnetwork
import keras.backend as K

# For debugging
# theano.config.mode='FAST_COMPILE'
from model.ModelInterface import ModelInterface

class DeepCNNKeras(ModelInterface):
    
    def __init__(self, n_in, n_out, state_bounds, action_bounds, reward_bound, settings_):

        super(DeepCNNKeras,self).__init__(n_in, n_out, state_bounds, action_bounds, reward_bound, settings_)
        
        ### Apparently after the first layer the patch axis is left out for most of the Keras stuff...
        input = Input(shape=(self._state_length,))
        input.trainable = True
        print ("Input ",  input)
        ## Custom slice layer, Keras does not have this layer type...
        taskFeatures = Lambda(lambda x: x[:,0:self._settings['num_terrain_features']], output_shape=(self._settings['num_terrain_features'],))(input)
        # taskFeatures = Lambda(lambda x: x[:,0:self._settings['num_terrain_features']])(input)
        characterFeatures = Lambda(lambda x: x[:,self._settings['num_terrain_features']:self._state_length], output_shape=(self._state_length-self._settings['num_terrain_features'],))(input)
        # characterFeatures = Reshape((-1, self._state_length-self._settings['num_terrain_features']))(characterFeatures)
        # characterFeatures = Lambda(lambda x: x[:,self._settings['num_terrain_features']:self._state_length], output_shape=(1,))(input)
        # print("TaskFeature shape: ", taskFeatures.output_shape)
        network = Reshape((self._settings['num_terrain_features'], 1))(taskFeatures)
        network = Conv1D(filters=16, kernel_size=8)(network)
        network = Activation('relu')(network)
        network = Conv1D(filters=16, kernel_size=8)(network)
        network = Activation('relu')(network)
        self._critic_task_part = network
        
        network = Flatten()(network)
        # characterFeatures = Flatten()(characterFeatures)
        # network = Concatenate(axis=1)([network, characterFeatures])
        ## network.shape should == (None, self._settings['num_terrain_features']) and
        ## characterFeatures.shape should == (None, state_length-self._settings['num_terrain_features'])
        print ("characterFeatures ", characterFeatures)
        print ("network ", network)
        network = Concatenate(axis=1)([network, characterFeatures])
        
        network = Dense(128, init='uniform')(network)
        print ("Network: ", network) 
        network = Activation('relu')(network)
        network = Dense(64, init='uniform')(network) 
        network = Activation('relu')(network)
        network = Dense(32, init='uniform')(network) 
        network = Activation('relu')(network)
        # 1 output, linear activation
        network = Dense(1, init='uniform')(network)
        network = Activation('linear')(network)
        self._critic = Model(input=input, output=network)
        
        
        inputAct = Input(shape=(self._state_length, ))
        inputAct.trainable = True
        print ("Input ",  inputAct)
        ## Custom slice layer, Keras does not have this layer type...
        taskFeaturesAct = Lambda(lambda x: x[:,0:self._settings['num_terrain_features']], output_shape=(self._settings['num_terrain_features'],))(inputAct)
        characterFeaturesAct = Lambda(lambda x: x[:,self._settings['num_terrain_features']:self._state_length], output_shape=(self._state_length-self._settings['num_terrain_features'], ))(inputAct)
        ## Keras/Tensorflow likes the channels to be at the end
        networkAct = Reshape((self._settings['num_terrain_features'], 1))(taskFeaturesAct)
        networkAct = Conv1D(filters=16, kernel_size=8)(networkAct)
        networkAct = Activation('relu')(networkAct)
        networkAct = Conv1D(filters=16, kernel_size=8)(networkAct)
        networkAct = Activation('relu')(networkAct)
        self._actor_task_part = networkAct
        
        networkAct = Flatten()(networkAct)
        networkAct = Concatenate(axis=1)([networkAct, characterFeaturesAct])
        
        networkAct = Dense(128, init='uniform')(networkAct)
        print ("Network: ", networkAct) 
        networkAct = Activation('relu')(networkAct)
        networkAct = Dense(64, init='uniform')(networkAct) 
        networkAct = Activation('relu')(networkAct)
        networkAct = Dense(32, init='uniform')(networkAct) 
        networkAct = Activation('relu')(networkAct)
        # 1 output, linear activation
        networkAct = Dense(self._action_length, init='uniform')(networkAct)
        networkAct = Activation('linear')(networkAct)
        self._actor = Model(input=inputAct, output=networkAct)
        
    def setStates(self, states):
        """
            This is reshaped to work properly with a 1D convolution that likes 
            its input as (batch_size, channel, state_dimension)
            
            Parameters
        ----------
        states : a (batch_size, state_dimension) numpy array
        """
        # states = np.array(states)
        # states = np.reshape(states, (states.shape[0], 1, states.shape[1]))
        # self._states_shared.set_value(states)
    def setResultStates(self, resultStates):
        """
            This is reshaped to work properly with a 1D convolution that likes 
            its input as (batch_size, channel, state_dimension)
            
            Parameters
        ----------
        resultStates : a (batch_size, state_dimension) numpy array
        """
        # resultStates = np.array(resultStates)
        # resultStates = np.reshape(resultStates, (resultStates.shape[0], 1, resultStates.shape[1]))
        # self._next_states_shared.set_value(resultStates)

