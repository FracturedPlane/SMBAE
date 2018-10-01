import theano
from theano import tensor as T
import numpy as np
import lasagne
import copy

import sys
sys.path.append('../')
from model.ModelUtil import *

from algorithm.AlgorithmInterface import AlgorithmInterface

class DoubleDeepQNetwork(AlgorithmInterface):
    
    def __init__(self, model, n_in, n_out, state_bounds, action_bounds, reward_bound, settings_):
        
        super(DoubleDeepQNetwork,self).__init__(model, n_in, n_out, state_bounds, action_bounds, reward_bound, settings_)

        batch_size=32
        # data types for model
        model._Action = T.lmatrix("Action")
        model._Action.tag.test_value = np.zeros((batch_size,1),dtype=np.dtype('int64'))
        # create a small convolutional neural network
        model._actions_shared = theano.shared(
            np.zeros((batch_size, 1), dtype='int64')
            ,broadcastable=(False, True)
            )
        
        self._model = model
        self._modelTarget = copy.deepcopy(model)
        # print ("Initial W " + str(self._w_o.get_value()) )
        
        self._q_valsA = lasagne.layers.get_output(self._model.getActorNetwork(), self._model.getStateSymbolicVariable())
        self._q_valsB = lasagne.layers.get_output(self._modelTarget.getActorNetwork(), self._model.getStateSymbolicVariable())
        self._q_valsA_B = lasagne.layers.get_output(self._model.getActorNetwork(), self._model.getResultStateSymbolicVariable())
        self._q_valsB_A = lasagne.layers.get_output(self._modelTarget.getActorNetwork(), self._model.getResultStateSymbolicVariable())
        
        # This forces every batch to be of size batch_size...
        self._q_func = self._q_valsA[T.arange(len(self._model.getStateValues())), self._model.getActionSymbolicVariable().reshape((-1,))].reshape((-1, 1))
        self._q_funcB = self._q_valsB[T.arange(len(self._modelTarget.getStateValues())), self._model.getActionSymbolicVariable().reshape((-1,))].reshape((-1, 1))
        
        target = (self._model.getRewardSymbolicVariable() +
                #(T.ones_like(terminals) - terminals) *
                  self._discount_factor * T.max(self._q_valsB_A, axis=1, keepdims=True))
        diff = target - self._q_valsA[T.arange(len(self._model.getStateValues())),
                               self._model.getActionSymbolicVariable().reshape((-1,))].reshape((-1, 1))# Does some fancy indexing to get the column of interest
                               
        loss = 0.5 * diff ** 2 + (1e-5 * lasagne.regularization.regularize_network_params(
                                            self._model.getActorNetwork(), lasagne.regularization.l2))
        loss = T.mean(loss)
        
        targetB = (self._model.getRewardSymbolicVariable() +
                #(T.ones_like(terminals) - terminals) *
                  self._discount_factor * T.max(self._q_valsA_B, axis=1, keepdims=True))
        diffB = targetB - self._q_valsB[T.arange(len(self._modelTarget.getStateValues())),
                               self._model.getActionSymbolicVariable().reshape((-1,))].reshape((-1, 1))# Does some fancy indexing to get the column of interest
                               
        lossB = 0.5 * diffB ** 2 + (1e-5 * lasagne.regularization.regularize_network_params(
                                            self._modelTarget.getActorNetwork(), lasagne.regularization.l2))
        lossB = T.mean(lossB)
        
        params = lasagne.layers.helper.get_all_params(self._model.getActorNetwork())
        paramsB = lasagne.layers.helper.get_all_params(self._modelTarget.getActorNetwork())
        
        givens = {
            self._model.getStateSymbolicVariable(): self._model.getStates(),
            self._model.getResultStateSymbolicVariable(): self._model.getResultStates(),
            self._model.getRewardSymbolicVariable(): self._model.getRewards(),
            self._model.getActionSymbolicVariable(): self._model.getActions(),
        }
        
        givensB = {
            self._model.getStateSymbolicVariable(): self._modelTarget.getStates(),
            self._model.getResultStateSymbolicVariable(): self._modelTarget.getResultStates(),
            self._model.getRewardSymbolicVariable(): self._modelTarget.getRewards(),
            self._model.getActionSymbolicVariable(): self._modelTarget.getActions(),
        }
        
        # SGD update
        
        updates = lasagne.updates.rmsprop(loss, params, self._learning_rate, self._rho,
                                        self._rms_epsilon)
        updatesB = lasagne.updates.rmsprop(lossB, paramsB, self._learning_rate, self._rho,
                                        self._rms_epsilon)
        # TD update
        """
        updates = lasagne.updates.rmsprop(T.mean(self._q_func) + (1e-5 * lasagne.regularization.regularize_network_params(
        self._model.getActorNetwork(), lasagne.regularization.l2)), params, 
                     self._learning_rate * -T.mean(diff), self._rho, self._rms_epsilon)
        updatesB = lasagne.updates.rmsprop(T.mean(self._q_funcB) + (1e-5 * lasagne.regularization.regularize_network_params(
        self._modelTarget.getActorNetwork(), lasagne.regularization.l2)), params, 
                     self._learning_rate * -T.mean(diffB), self._rho, self._rms_epsilon)
        """
        
        
        self._train = theano.function([], [loss, self._q_valsA], updates=updates,
                                      givens=givens)
        self._trainB = theano.function([], [lossB, self._q_valsB], updates=updatesB,
                                      givens=givensB)
        
        self._q_vals = theano.function([], self._q_valsA,
                                       givens={self._model.getStateSymbolicVariable(): self._model.getStates()})
        self._q_valsB = theano.function([], self._q_valsB,
                                       givens={self._model.getStateSymbolicVariable(): self._modelTarget.getStates()})
        
        self._bellman_error = theano.function(inputs=[self._model.getStateSymbolicVariable(), self._model.getActionSymbolicVariable(), 
            self._model.getRewardSymbolicVariable(), self._model.getResultStateSymbolicVariable()], 
                                              outputs=diff, allow_input_downcast=True)
        self._bellman_error2 = theano.function(inputs=[], outputs=diff, allow_input_downcast=True, givens=givens)
        self._bellman_error2B = theano.function(inputs=[], outputs=diffB, allow_input_downcast=True, givens=givensB)
        
    def updateTargetModel(self):
        print ("Updating target Model")
        """
            Double Q learning now
        """
        all_paramsA = lasagne.layers.helper.get_all_param_values(self._model.getActorNetwork())
        # all_paramsActA = lasagne.layers.helper.get_all_param_values(self._model.getActorNetwork())
        lasagne.layers.helper.set_all_param_values(self._modelTarget.getActorNetwork(), all_paramsA)
        # lasagne.layers.helper.set_all_param_values(self._modelB.getActorNetwork(), all_paramsActA)
    
    def setData(self, states, actions, rewards, result_states):
        self._model.setStates(states)
        self._model.setResultStates(result_states)
        self._model.setActions(actions)
        self._model.setRewards(rewards)
        self._modelTarget.setStates(states)
        self._modelTarget.setResultStates(result_states)
        self._modelTarget.setActions(actions)
        self._modelTarget.setRewards(rewards)
        
    def train(self, states, actions, rewards, result_states):
        self.setData(states, actions, rewards, result_states)

        if (( self._updates % self._weight_update_steps) == 0):
            self.updateTargetModel()
        self._updates += 1
        import random
        r = random.choice([0,1])
        if r == 0:
            loss, _ = self._train()
            
            # diff_ = self.bellman_error(states, actions, rewards, result_states)
        else:
            loss, _ = self._trainB()
            
            # diff_ = self.bellman_errorB(states, actions, rewards, result_states)
        return loss
    
        return np.sqrt(loss)
    
    def q_value(self, state):
        """
            Don't normalize here it is done in q_values
        """
        # state = [norm_state(state, self._state_bounds)]
        q_values = self.q_values(state)
        action_ = self.predict(state)
        # print ("q_values: " + str(q_values) + " Action: " + str(action_))
        original_val = q_values[action_]
        return original_val
    
    def predict(self, state):
        """
            Don't normalize here it is done in q_values
        """
        # state = [norm_state(state, self._state_bounds)]
        q_vals = self.q_values(state)
        return np.argmax(q_vals)
    
    def q_values(self, state):
        state = [norm_state(state, self._state_bounds)]
        # print ("Q value: ", state)
        self._model.setStates(state)
        return self._q_vals()[0]
    
    def bellman_error(self, states, actions, rewards, result_states):
        # print ("Bellman error 2 actions: ", len(actions) , " rewards ", len(rewards), " states ", len(states), " result_states: ", len(result_states))
        self.setData(states, actions, rewards, result_states)
        return self._bellman_error2()
        # return self._bellman_error(state, action, reward, result_state)
