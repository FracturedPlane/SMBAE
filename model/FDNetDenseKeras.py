import theano
from theano import tensor as T
import numpy as np
import sys
sys.path.append('../')
from model.ModelUtil import *
from util.nn import weight_norm
from model.DeepNNKerasAdaptive import getKerasActivation

from keras.models import Sequential, Model
from keras.layers import Input
from keras.layers.core import Dense, Dropout, Activation, Reshape, Flatten, Lambda
from keras.layers.convolutional import Conv1D
from keras.layers.merge import Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras import regularizers
import keras
# from keras.utils.np_utils import to_categoricalnetwork
import keras.backend as K


# elu
def elu_mine(x):
    return theano.tensor.switch(x > 0, x, theano.tensor.expm1(x))

# For debugging
# theano.config.mode='FAST_COMPILE'
from model.ModelInterface import ModelInterface

class FDNetDenseKeras(ModelInterface):
    
    def __init__(self, state_length, action_length, state_bounds, action_bounds, settings_):

        super(FDNetDenseKeras,self).__init__(state_length, action_length, state_bounds, action_bounds, 0, settings_)
        
        self._result_state_length = state_length
        
        batch_size=32
        # data types for model
        
        input = Input(shape=(self._state_length,))
        self._stateInput = input
        input2 = Input(shape=(self._action_length,)) 
        self._actionInput = input2
        self._nextStateInput = Input(shape=(self._result_state_length,))
        self._noiseInput = Input(shape=(1,)) 
        
        input = self._stateInput
        insert_action_later = True
        double_insert_action = False
        add_layers_after_action = False
        if (not insert_action_later or (double_insert_action)):
            input = self._actor = keras.layers.concatenate(inputs=[self._stateInput, self._actionInput], axis=-1)
            
        decrease_param_factor=int(2)
        ## Activation types
        # activation_type = elu_mine
        # activation_type=lasagne.nonlinearities.tanh
        # activation_type=keras.layers.LeakyReLU(alpha=0.01)
        # activation_type="leaky_rectify"
        activation_type="tanh"
        # getKerasActivation(activation_type)
        # activation_type=getKerasActivation("tanh")
        # activation_type=lasagne.nonlinearities.rectify
        # network = lasagne.layers.DropoutLayer(input, p=self._dropout_p, rescale=True)
        network = Dense(int(128/decrease_param_factor), kernel_regularizer=regularizers.l2(self._settings['regularization_weight']))(input)
        network = getKerasActivation(activation_type)(network)   
        network = Dropout(rate=self._dropout_p)(network)     
        # layersAct = [network]
        
        if ( insert_action_later ):
            ### Lets try adding the action input later on in the network
            if ( add_layers_after_action ):
                networkA = Dense(int(32/decrease_param_factor), kernel_regularizer=regularizers.l2(self._settings['regularization_weight']))(self._actionInput)
                networkA = getKerasActivation(activation_type)(networkA)   
                network = keras.layers.concatenate(inputs=[network, networkA], axis=-1)
            else:
                network = keras.layers.concatenate(inputs=[network, self._actionInput], axis=-1)
        
        network = Dense(int(64/decrease_param_factor), kernel_regularizer=regularizers.l2(self._settings['regularization_weight']))(network)
        network = getKerasActivation(activation_type)(network)   
        network = Dropout(rate=self._dropout_p)(network)   
        
        network = Dense(int(32/decrease_param_factor), kernel_regularizer=regularizers.l2(self._settings['regularization_weight']))(network)
        network = getKerasActivation(activation_type)(network)   
        network = Dropout(rate=self._dropout_p)(network)   
        
        
        # layersAct.append(network)
        # network = lasagne.layers.ConcatLayer([layersAct[2], layersAct[1], layersAct[0]])
        # network = lasagne.layers.DropoutLayer(network, p=self._dropout_p, rescale=True)
        network = Dense(8, kernel_regularizer=regularizers.l2(self._settings['regularization_weight']))(network)
        network = getKerasActivation(activation_type)(network)   
        network = Dropout(rate=self._dropout_p)(network)   

        ## This can be used to model the reward function
        network = Dense(1, kernel_regularizer=regularizers.l2(self._settings['regularization_weight']))(network)
        network = getKerasActivation("linear")(network)   
        self._reward_net = Model(input=[self._stateInput, self._actionInput], output=network)
        print("Reward Net summary: ",  self._reward_net.summary())
                
        ### discriminator
        inputDiscrominator = keras.layers.concatenate(inputs=[self._stateInput, self._actionInput, self._nextStateInput], axis=-1)
        
        networkDiscrominator = Dense(int(128/decrease_param_factor), kernel_regularizer=regularizers.l2(self._settings['regularization_weight']))(inputDiscrominator)
        networkDiscrominator = getKerasActivation(activation_type)(networkDiscrominator)   
        networkDiscrominator = Dropout(rate=self._dropout_p)(networkDiscrominator) 
        # layersAct = [network]
        
        if ( insert_action_later ):
            ### Lets try adding the action input later on in the network
            if ( add_layers_after_action ):
                networkA = Dense(int(32/decrease_param_factor), kernel_regularizer=regularizers.l2(self._settings['regularization_weight']))(self._actionInput)
                networkA = getKerasActivation(activation_type)(networkA)
                networkDiscrominator = keras.layers.concatenate(inputs=[networkDiscrominator, networkA], axis=-1)
            else:
                networkDiscrominator = keras.layers.concatenate(inputs=[networkDiscrominator, self._actionInput], axis=-1)
        
        networkDiscrominator = Dense(int(64/decrease_param_factor), kernel_regularizer=regularizers.l2(self._settings['regularization_weight']))(networkDiscrominator)
        networkDiscrominator = getKerasActivation(activation_type)(networkDiscrominator)   
        networkDiscrominator = Dropout(rate=self._dropout_p)(networkDiscrominator) 
        
        # layersAct.append(network)
        # network = lasagne.layers.ConcatLayer([layersAct[1], layersAct[0]])
        
        networkDiscrominator = Dense(int(32/decrease_param_factor), kernel_regularizer=regularizers.l2(self._settings['regularization_weight']))(networkDiscrominator)
        networkDiscrominator = getKerasActivation(activation_type)(networkDiscrominator)   
        networkDiscrominator = Dropout(rate=self._dropout_p)(networkDiscrominator) 
        
        
        # layersAct.append(network)
        # network = lasagne.layers.ConcatLayer([layersAct[2], layersAct[1], layersAct[0]])
        # network = lasagne.layers.DropoutLayer(network, p=self._dropout_p, rescale=True)
        networkDiscrominator = Dense(8, kernel_regularizer=regularizers.l2(self._settings['regularization_weight']))(networkDiscrominator)
        networkDiscrominator = getKerasActivation(activation_type)(networkDiscrominator)   
        networkDiscrominator = Dropout(rate=self._dropout_p)(networkDiscrominator) 
        ## This can be used to model the reward function
        networkDiscrominator = Dense(1, kernel_regularizer=regularizers.l2(self._settings['regularization_weight']))(networkDiscrominator)
        networkDiscrominator = getKerasActivation("linear")(networkDiscrominator)   
        
        self._critic = Model(input=[self._stateInput, self._actionInput, self._nextStateInput], output=networkDiscrominator)
        print("Discriminator Net summary: ",  self._critic.summary())
                
        input = keras.layers.concatenate(inputs=[self._stateInput, self._actionInput], axis=-1)
        ### dynamics network
        if ("train_gan_with_gaussian_noise" in settings_ 
            and (settings_["train_gan_with_gaussian_noise"] == True)
            and "train_gan" in settings_
            and (settings_["train_gan"] == True)
            ):
            ## Add noise input
            input = keras.layers.concatenate(inputs=[input, self._noiseInput], axis=-1)
          
        # networkAct = lasagne.layers.DropoutLayer(input, p=self._dropout_p, rescale=True)
        networkAct = Dense(int(256/decrease_param_factor), kernel_regularizer=regularizers.l2(self._settings['regularization_weight']))(input)
        networkAct = getKerasActivation(activation_type)(networkAct)   
        networkAct = Dropout(rate=self._dropout_p)(networkAct)
        # networkAct = weight_norm(networkAct)
        layersAct = [networkAct]
        
        networkAct = Dense(int(128/decrease_param_factor), kernel_regularizer=regularizers.l2(self._settings['regularization_weight']))(networkAct)
        networkAct = getKerasActivation(activation_type)(networkAct)   
        networkAct = Dropout(rate=self._dropout_p)(networkAct)
        # networkAct = weight_norm(networkAct)
        
        if ( insert_action_later ):
            ### Lets try adding the action input later on in the network
            if ( add_layers_after_action ):
                networkActA = Dense(int(64/decrease_param_factor), kernel_regularizer=regularizers.l2(self._settings['regularization_weight']))(self._actionInput)
                networkAct = getKerasActivation(activation_type)(networkAct)   
                networkAct = keras.layers.concatenate(inputs=[networkAct, networkActA], axis=-1)
            else:
                networkAct = keras.layers.concatenate(inputs=[networkAct, self._actionInput], axis=-1)
            
        
        layersAct.append(networkAct)
        networkAct = keras.layers.concatenate(inputs=[layersAct[1], layersAct[0]], axis=-1)
        
        networkAct = Dense(int(64/decrease_param_factor), kernel_regularizer=regularizers.l2(self._settings['regularization_weight']))(networkAct)
        networkAct = getKerasActivation(activation_type)(networkAct)   
        networkAct = Dropout(rate=self._dropout_p)(networkAct)
        
        # networkAct = weight_norm(networkAct)
        layersAct.append(networkAct)
        # networkAct = keras.layers.concatenate(inputs=[layersAct[2], layersAct[1], layersAct[0]], axis=-1)
        
        networkAct = Dense(self._result_state_length, kernel_regularizer=regularizers.l2(self._settings['regularization_weight']))(networkAct)
        networkAct = getKerasActivation(activation_type)(networkAct)
    
        if ("train_gan_with_gaussian_noise" in settings_ 
            and (settings_["train_gan_with_gaussian_noise"] == True)
            and "train_gan" in settings_
            and (settings_["train_gan"] == True)
            ):
            self._forward_dynamics_net = Model(input=[self._stateInput, self._actionInput, self._noiseInput], output=networkAct)
                # print ("Initial W " + str(self._w_o.get_value()) )
        else:
            self._forward_dynamics_net = Model(input=[self._stateInput, self._actionInput], output=networkAct)

        print("FD Net summary: ",  self._forward_dynamics_net.summary())

        self._states_shared = theano.shared(
            np.zeros((batch_size, self._state_length),
                     dtype=theano.config.floatX))

        self._next_states_shared = theano.shared(
            np.zeros((batch_size, self._result_state_length),
                     dtype=theano.config.floatX))

        self._actions_shared = theano.shared(
            np.zeros((batch_size, self._action_length), dtype=theano.config.floatX),
            )
        
        self._rewards_shared = theano.shared(
            np.zeros((self._batch_size, 1), dtype=theano.config.floatX),
            broadcastable=(False, True))
        
