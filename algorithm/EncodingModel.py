import theano
from theano import tensor as T
import numpy as np
import lasagne
import sys
sys.path.append('../')
from model.ModelUtil import *
from model.LearningUtil import loglikelihood, kl, entropy, flatgrad, zipsame, get_params_flat, setFromFlat, likelihood, loglikelihoodMEAN

# For debugging
# theano.config.mode='FAST_COMPILE'
from algorithm.AlgorithmInterface import AlgorithmInterface

class EncodingModel(AlgorithmInterface):
    
    def __init__(self, model, state_length, action_length, state_bounds, action_bounds, settings_):

        super(EncodingModel,self).__init__(model, state_length, action_length, state_bounds, action_bounds, 0, settings_)
        self._model = model
        batch_size=32
        self._learning_rate = self.getSettings()["fd_learning_rate"]
        # data types for model
        # create a small convolutional neural network
        
        # inputs_ = self._model.getStateSymbolicVariable()
        self._forward = lasagne.layers.get_output(self._model.getEncodeNet(), self._model.getStateSymbolicVariable(), deterministic=True)[:,:self._state_length]
        ## This drops to ~ 0 so fast.
        self._forward_std = (lasagne.layers.get_output(self._model.getEncodeNet(), self._model.getStateSymbolicVariable(), deterministic=True)[:,self._state_length:] * self.getSettings()['exploration_rate'] )+ 1e-4
        self._forward_std_drop = (lasagne.layers.get_output(self._model.getEncodeNet(), self._model.getStateSymbolicVariable(), deterministic=True)[:,self._state_length:] * self.getSettings()['exploration_rate']) + 1e-4
        self._forward_drop = lasagne.layers.get_output(self._model.getEncodeNet(), self._model.getStateSymbolicVariable(), deterministic=False)[:,:self._state_length]
        
        l2_loss = True
        
        # self._target = (Reward + self._discount_factor * self._q_valsB)
        self._diff = self._model.getStateSymbolicVariable() - self._forward_drop
        ## mean across each sate
        if (l2_loss):
            self._loss = T.mean(T.pow(self._diff, 2),axis=1)
        else:
            self._loss = T.mean(T.abs_(self._diff),axis=1)
        ## mean over batch
        self._loss = T.mean(self._loss)
        ## Another version that does not have dropout
        self._diff_NoDrop = self._model.getStateSymbolicVariable() - self._forward
        ## mean across each sate
        if (l2_loss):
            self._loss_NoDrop = T.mean(T.pow(self._diff_NoDrop, 2),axis=1)
        else:
            self._loss_NoDrop = T.mean(T.abs_(self._diff_NoDrop),axis=1)
        ## mean over batch
        self._loss_NoDrop = T.mean(self._loss_NoDrop)
        
        
        self._params = lasagne.layers.helper.get_all_params(self._model.getEncodeNet())
        self._givens_ = {
            self._model.getStateSymbolicVariable() : self._model.getStates(),
        }
        
        # SGD update
        if (self.getSettings()['optimizer'] == 'rmsprop'):
            print ("Optimizing Encoding Model with ", self.getSettings()['optimizer'], " method")
            self._updates_ = lasagne.updates.rmsprop(self._loss + (self._regularization_weight * lasagne.regularization.regularize_network_params(
                self._model.getEncodeNet(), lasagne.regularization.l2)), self._params, self._learning_rate, self._rho, self._rms_epsilon)
        elif (self.getSettings()['optimizer'] == 'momentum'):
            print ("Optimizing Encoding Model with ", self.getSettings()['optimizer'], " method")
            self._updates_ = lasagne.updates.momentum(self._loss + (self._regularization_weight * lasagne.regularization.regularize_network_params(
                self._model.getEncodeNet(), lasagne.regularization.l2)), self._params, self._learning_rate, momentum=self._rho)
        elif ( self.getSettings()['optimizer'] == 'adam'):
            print ("Optimizing Encoding Model with ", self.getSettings()['optimizer'], " method")
            self._updates_ = lasagne.updates.adam(self._loss + (self._regularization_weight * lasagne.regularization.regularize_network_params(
                self._model.getEncodeNet(), lasagne.regularization.l2)), self._params, self._learning_rate, beta1=0.9, beta2=0.999, epsilon=self._rms_epsilon)
        elif ( self.getSettings()['optimizer'] == 'adagrad'):
            print ("Optimizing Encoding Model with ", self.getSettings()['optimizer'], " method")
            self._updates_ = lasagne.updates.adagrad(self._loss + (self._regularization_weight * lasagne.regularization.regularize_network_params(
                self._model.getEncodeNet(), lasagne.regularization.l2)), self._params, self._learning_rate, epsilon=self._rms_epsilon)
        else:
            print ("Unknown optimization method: ", self.getSettings()['optimizer'])
        self._updates_ = lasagne.updates.rmsprop(self._loss + (self._regularization_weight * lasagne.regularization.regularize_network_params(
                self._model.getEncodeNet(), lasagne.regularization.l2)), self._params, self._learning_rate, self._rho,
                                            self._rms_epsilon)
        
        self._train = theano.function([], [self._loss], updates=self._updates_, givens=self._givens_)
        self._forwardDynamics = theano.function([], self._forward,
                                       givens={self._model.getStateSymbolicVariable() : self._model.getStates()
                                                })
        self._forwardDynamics_std = theano.function([], self._forward_std,
                                       givens={self._model.getStateSymbolicVariable() : self._model.getStates()
                                                })
        
        self._bellman_error = theano.function(inputs=[], outputs=self._diff, allow_input_downcast=True, givens=self._givens_)
        # self._diffs = theano.function(input=[State])
        # self._get_grad = theano.function([], outputs=lasagne.updates.get_or_compute_grads(self._loss_NoDrop, [lasagne.layers.get_all_layers(self._model.getForwardDynamicsNetwork())[self.getSettings()['action_input_layer_index']].input_var] + self._params), allow_input_downcast=True, givens=self._givens_)
        self._get_grad = theano.function([], outputs=lasagne.updates.get_or_compute_grads(self._loss_NoDrop, [self._model._stateInputVar] + self._params), allow_input_downcast=True, givens=self._givens_)
        # self._get_grad_reward = theano.function([], outputs=lasagne.updates.get_or_compute_grads((self._reward_loss_NoDrop), [lasagne.layers.get_all_layers(self._model.getRewardNetwork())[0].input_var] + self._reward_params), allow_input_downcast=True,

    def getNetworkParameters(self):
        params = []
        params.append(lasagne.layers.helper.get_all_param_values(self._model.getForwardDynamicsNetwork()))
        return params
    
    def setNetworkParameters(self, params):
        lasagne.layers.helper.set_all_param_values(self._model.getForwardDynamicsNetwork(), params[0])
        
    def setData(self, states, actions, result_states, rewards=[]):
        self._model.setStates(states)
        
    """
    def getGrads(self, states, actions, result_states):
        states = np.array(norm_state(states, self._state_bounds), dtype=self.getSettings()['float_type'])
        self.setData(states, actions, result_states)
        return self._get_grad()
    """
    def train(self, states, actions, result_states, rewards):
        # rewards = rewards * (1.0/(1.0-self.getSettings()['discount_factor'])) # scale rewards
        self.setData(states, actions, result_states, rewards)
        # print ("Performing Critic trainning update")
        #if (( self._updates % self._weight_update_steps) == 0):
        #    self.updateTargetModel()
        self._updates += 1
        # all_paramsActA = lasagne.layers.helper.get_all_param_values(self._l_outActA)
        loss = self._train()

        # This undoes the Actor parameter updates as a result of the Critic update.
        # print (diff_)
        return loss
    
    def predict(self, state, action):
        # states = np.zeros((self._batch_size, self._self._state_length), dtype=theano.config.floatX)
        # states[0, ...] = state
        state = np.array(norm_state(state, self._state_bounds), dtype=self.getSettings()['float_type'])
        # print ("fd state: ", state)
        self._model.setStates(state)
        # print ("State bounds: ", self._state_bounds)
        # print ("fd output: ", self._forwardDynamics()[0])
        state_ = scale_state(self._forwardDynamics()[0], self._state_bounds)
        return state_
    
    def predict_std(self, state, action):
        # states = np.zeros((self._batch_size, self._self._state_length), dtype=theano.config.floatX)
        # states[0, ...] = state
        state = np.array(norm_state(state, self._state_bounds), dtype=self.getSettings()['float_type'])
        # print ("fd state: ", state)
        self._model.setStates(state)
        state_ = scale_state(self._forwardDynamics_std()[0], self._state_bounds)
        return state_
    
    def predict_batch(self, states, actions):
        ## These input should already be normalized.
        self._model.setStates(states)
        return self._forwardDynamics()

    def bellman_error(self, states, actions, result_states, rewards):
        self.setData(states, actions, result_states, rewards)
        return self._bellman_error()
