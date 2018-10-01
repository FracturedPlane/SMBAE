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

class DeepNNUNet(ModelInterface):
    
    def __init__(self, state_length, action_length, state_bounds, action_bounds, settings_):

        super(DeepNNUNet,self).__init__(state_length, action_length, state_bounds, action_bounds, 0, settings_)
        
        # self._result_state_length=20
        self._result_state_length=state_length
        # data types for model
        self._State = T.matrix("State")
        self._State.tag.test_value = np.random.rand(self._batch_size, self._state_length)
        self._ResultState = T.matrix("ResultState")
        self._ResultState.tag.test_value = np.random.rand(self._batch_size, self._result_state_length)
        self._Reward = T.col("Reward")
        self._Reward.tag.test_value = np.random.rand(self._batch_size,1)
        self._Action = T.matrix("Action")
        self._Action.tag.test_value = np.random.rand(self._batch_size, self._action_length)
        ## noise parameters for GAN
        self._Noise = T.matrix("Noise")
        self._Noise.tag.test_value = np.random.rand(self._batch_size,1)
        
        # create a small convolutional neural network
        inputState = lasagne.layers.InputLayer((None, self._state_length), self._State)
        self._stateInputVar = inputState.input_var
        inputAction = lasagne.layers.InputLayer((None, self._action_length), self._Action)
        self._actionInputVar = inputAction.input_var
        inputNextState = lasagne.layers.InputLayer((None, self._result_state_length), self._ResultState)
        self._resultStateInputVar = inputNextState.input_var
        # self._b_o = init_b_weights((n_out,))
        # networkAct = lasagne.layers.InputLayer((None, self._state_length), self._State)
        
        
        u_net = True
        insert_action_later = True
        double_insert_action = False
        add_layers_after_action = True
        
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
        
        if (not insert_action_later or (double_insert_action)):
            inputGenerator = lasagne.layers.ConcatLayer([inputState, inputAction])
        else:
            inputGenerator = inputState
        if ("train_gan_with_gaussian_noise" in settings_ and (settings_["train_gan_with_gaussian_noise"])):
            ## Add noise input
            inputNoise = lasagne.layers.InputLayer((None, 1), self._Noise)
            inputGenerator = lasagne.layers.ConcatLayer([inputGenerator, inputNoise])
        """
        networkAct = lasagne.layers.DenseLayer(
                networkAct, num_units=256,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        """
        # inputGenerator = lasagne.layers.ConcatLayer([inputState, inputAction])
        networkAct = lasagne.layers.DenseLayer(
                inputGenerator, num_units=128,
                nonlinearity=activation_type)
        networkAct1 = networkAct
        # networkAct = lasagne.layers.DropoutLayer(networkAct, p=self._dropout_p, rescale=True)
        
        networkAct = lasagne.layers.DenseLayer(
                networkAct, num_units=64,
                nonlinearity=activation_type)
        networkAct2 = networkAct
        # networkAct = lasagne.layers.DropoutLayer(networkAct, p=self._dropout_p, rescale=True)
        
        networkAct = lasagne.layers.DenseLayer(
                networkAct, num_units=32,
                nonlinearity=activation_type)
    
        if ( insert_action_later ):
            ### Lets try adding the action input later on in the network
            if ( add_layers_after_action ):
                networkActA = lasagne.layers.DenseLayer(
                    inputAction, num_units=32,
                    nonlinearity=activation_type)
                networkAct = lasagne.layers.ConcatLayer([networkAct, networkActA])
            else:
                networkAct = lasagne.layers.ConcatLayer([networkAct, inputAction])
            
        networkAct = lasagne.layers.DropoutLayer(networkAct, p=self._dropout_p, rescale=True)
        if (u_net):
            networkAct = lasagne.layers.ConcatLayer([networkAct, networkAct2])
        networkAct = lasagne.layers.DenseLayer(
                networkAct, num_units=64,
                nonlinearity=activation_type)
        # networkAct = lasagne.layers.DropoutLayer(networkAct, p=self._dropout_p, rescale=True)
        if (u_net):
            networkAct = lasagne.layers.ConcatLayer([networkAct, networkAct1])
        networkAct = lasagne.layers.DenseLayer(
                networkAct, num_units=128,
                nonlinearity=activation_type)
        # networkAct = lasagne.layers.DropoutLayer(networkAct, p=self._dropout_p, rescale=True)
        
        self._forward_dynamics_net = lasagne.layers.DenseLayer(
                networkAct, num_units=self._result_state_length,
                nonlinearity=last_policy_layer_activation_type)
        
        
        inputDiscriminator = lasagne.layers.ConcatLayer([inputState, inputAction, inputNextState])
        """
        network = lasagne.layers.DenseLayer(
                network, num_units=256,
                nonlinearity=activation_type)
        """
        network = lasagne.layers.DenseLayer(
                inputDiscriminator, num_units=128,
                nonlinearity=activation_type)
        network = lasagne.layers.DropoutLayer(network, p=self._dropout_p, rescale=True)
        
        network = lasagne.layers.DenseLayer(
                network, num_units=64,
                nonlinearity=activation_type)
        network = lasagne.layers.DropoutLayer(network, p=self._dropout_p, rescale=True)
        """
        if ( settings_['agent_name'] == 'algorithm.DPG.DPG'):
            network = lasagne.layers.ConcatLayer([network, inputAction])
        """
        network = lasagne.layers.DenseLayer(
                network, num_units=32,
                nonlinearity=activation_type)
        network = lasagne.layers.DropoutLayer(network, p=self._dropout_p, rescale=True)
        
        network = lasagne.layers.DenseLayer(
                network, num_units=16,
                nonlinearity=activation_type)
        network = lasagne.layers.DropoutLayer(network, p=self._dropout_p, rescale=True)
        """
        network = lasagne.layers.DenseLayer(
                network, num_units=8,
                nonlinearity=activation_type)
        network = lasagne.layers.DropoutLayer(network, p=self._dropout_p, rescale=True)
        """
        self._critic = lasagne.layers.DenseLayer(
                network, num_units=1,
                nonlinearity=lasagne.nonlinearities.linear)
        
        
        if (not insert_action_later or (double_insert_action)):
            inputReward = lasagne.layers.ConcatLayer([inputState, inputAction])
        else:
            inputReward = inputState
        """
        network = lasagne.layers.DenseLayer(
                input, num_units=128,
                nonlinearity=activation_type)
        network = weight_norm(network)
        network = lasagne.layers.DropoutLayer(network, p=self._dropout_p, rescale=True)
        layersAct = [network]
        
        network = lasagne.layers.DenseLayer(
                network, num_units=64,
                nonlinearity=activation_type)
        network = weight_norm(network)
        network = lasagne.layers.DropoutLayer(network, p=self._dropout_p, rescale=True)
        layersAct.append(network)
        network = lasagne.layers.ConcatLayer([layersAct[1], layersAct[0]])
        
        network = lasagne.layers.DenseLayer(
                network, num_units=32,
                nonlinearity=activation_type)
        network = weight_norm(network)
        network = lasagne.layers.DropoutLayer(network, p=self._dropout_p, rescale=True)
        layersAct.append(network)
        network = lasagne.layers.ConcatLayer([layersAct[2], layersAct[1], layersAct[0]])
        ## This can be used to model the reward function
        self._reward_net = lasagne.layers.DenseLayer(
                network, num_units=1,
                nonlinearity=lasagne.nonlinearities.linear)
                # print ("Initial W " + str(self._w_o.get_value()) )
        """
        
        network = lasagne.layers.DenseLayer(
                inputReward, num_units=128,
                nonlinearity=activation_type)
        # network = weight_norm(network)
        network = lasagne.layers.DropoutLayer(network, p=self._dropout_p, rescale=True)
        # layersAct = [network]
        
        if ( insert_action_later ):
            ### Lets try adding the action input later on in the network
            if ( add_layers_after_action ):
                networkA = lasagne.layers.DenseLayer(
                        inputAction, num_units=32,
                        nonlinearity=activation_type)
                network = lasagne.layers.ConcatLayer([network, networkA])
            else:
                network = lasagne.layers.ConcatLayer([network, inputAction])
        
        network = lasagne.layers.DenseLayer(
                network, num_units=64,
                nonlinearity=activation_type)
        # network = weight_norm(network)
        network = lasagne.layers.DropoutLayer(network, p=self._dropout_p, rescale=True)
        
        # layersAct.append(network)
        # network = lasagne.layers.ConcatLayer([layersAct[1], layersAct[0]])
        
        network = lasagne.layers.DenseLayer(
                network, num_units=32,
                nonlinearity=activation_type)
        # network = weight_norm(network)
        network = lasagne.layers.DropoutLayer(network, p=self._dropout_p, rescale=True)
        
        
        # layersAct.append(network)
        # network = lasagne.layers.ConcatLayer([layersAct[2], layersAct[1], layersAct[0]])
        # network = lasagne.layers.DropoutLayer(network, p=self._dropout_p, rescale=True)
        network = lasagne.layers.DenseLayer(
                network, num_units=8,
                nonlinearity=activation_type)
        network = lasagne.layers.DropoutLayer(network, p=self._dropout_p, rescale=True)
        """
        network = lasagne.layers.DenseLayer(
                network, num_units=8,
                nonlinearity=activation_type)
        """
        ## This can be used to model the reward function
        self._reward_net = lasagne.layers.DenseLayer(
                network, num_units=1,
                nonlinearity=lasagne.nonlinearities.linear)
                # print ("Initial W " + str(self._w_o.get_value()) )
        
          # print "Initial W " + str(self._w_o.get_value()) 
        
        self._states_shared = theano.shared(
            np.zeros((self._batch_size, self._state_length),
                     dtype=theano.config.floatX))

        self._next_states_shared = theano.shared(
            np.zeros((self._batch_size, self._result_state_length),
                     dtype=theano.config.floatX))

        self._rewards_shared = theano.shared(
            np.zeros((self._batch_size, 1), dtype=theano.config.floatX),
            broadcastable=(False, True))

        self._actions_shared = theano.shared(
            np.zeros((self._batch_size, self._action_length), dtype=theano.config.floatX),
            )
