import theano
from theano import tensor as T
import numpy as np
import lasagne

import sys
sys.path.append('../')
from model.ModelUtil import *

from model.AgentInterface import AgentInterface

class DeepRLNet3(AgentInterface):
    
    def __init__(self, n_in, n_out, state_bounds, action_bounds, reward_bound):
        
        super(DeepRLNet3,self).__init__(n_in, n_out, state_bounds, action_bounds, reward_bound)

        batch_size=32
        # data types for model
        State = T.dmatrix("State")
        State.tag.test_value = np.random.rand(batch_size,self._state_length)
        ResultState = T.dmatrix("ResultState")
        ResultState.tag.test_value = np.random.rand(batch_size,self._state_length)
        Reward = T.col("Reward")
        Reward.tag.test_value = np.random.rand(batch_size,1)
        Action = T.icol("Action")
        Action.tag.test_value = np.zeros((batch_size,1),dtype=np.dtype('int64'))
        # create a small convolutional neural network
        inputLayerA = lasagne.layers.InputLayer((None, self._state_length), State)

        l_hid1A = lasagne.layers.DenseLayer(
                inputLayerA, num_units=256,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        l_hid2A = lasagne.layers.DenseLayer(
                l_hid1A, num_units=128,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        
        l_hid3A = lasagne.layers.DenseLayer(
                l_hid2A, num_units=64,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
    
        self._l_outA = lasagne.layers.DenseLayer(
                l_hid3A, num_units=n_out,
                nonlinearity=lasagne.nonlinearities.linear)
        # self._b_o = init_b_weights((n_out,))
        
        # self.updateTargetModel()
        inputLayerB = lasagne.layers.InputLayer((None, self._state_length), State)

        l_hid1B = lasagne.layers.DenseLayer(
                inputLayerB, num_units=256,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        
        l_hid2B = lasagne.layers.DenseLayer(
                l_hid1B, num_units=128,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
    
        l_hid3B = lasagne.layers.DenseLayer(
                l_hid2B, num_units=64,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        
        self._l_outB = lasagne.layers.DenseLayer(
                l_hid3B, num_units=n_out,
                nonlinearity=lasagne.nonlinearities.linear)

        
        # print ("Initial W " + str(self._w_o.get_value()) )
        
        self._learning_rate = 0.0002
        self._discount_factor= 0.8
        self._rho = 0.95
        self._rms_epsilon = 0.001
        
        self._weight_update_steps=8000
        self._updates=0
        
        self._states_shared = theano.shared(
            np.zeros((batch_size, self._state_length),
                     dtype=theano.config.floatX))

        self._next_states_shared = theano.shared(
            np.zeros((batch_size, self._state_length),
                     dtype=theano.config.floatX))

        self._rewards_shared = theano.shared(
            np.zeros((batch_size, 1), dtype=theano.config.floatX),
            broadcastable=(False, True))

        self._actions_shared = theano.shared(
            np.zeros((batch_size, 1), dtype='int64'),
            broadcastable=(False, True))
        
        self._q_valsA = lasagne.layers.get_output(self._l_outA, State)
        self._q_valsB = lasagne.layers.get_output(self._l_outB, ResultState)
        
        self._q_func = self._q_valsA[T.arange(batch_size), Action.reshape((-1,))].reshape((-1, 1))
        
        target = (Reward +
                #(T.ones_like(terminals) - terminals) *
                  self._discount_factor * T.max(self._q_valsB, axis=1, keepdims=True))
        diff = target - self._q_valsA[T.arange(batch_size),
                               Action.reshape((-1,))].reshape((-1, 1))# Does some fancy indexing to get the column of interest
                               
        loss = 0.5 * diff ** 2 + (1e-6 * lasagne.regularization.regularize_network_params(
                                            self._l_outA, lasagne.regularization.l2))
        loss = T.mean(loss)
        
        params = lasagne.layers.helper.get_all_params(self._l_outA)
        
        givens = {
            State: self._states_shared,
            ResultState: self._next_states_shared,
            Reward: self._rewards_shared,
            Action: self._actions_shared,
        }
        
        # SGD update
        
        updates = lasagne.updates.rmsprop(loss, params, self._learning_rate, self._rho,
                                         self._rms_epsilon)
        # TD update
        # updates = lasagne.updates.rmsprop(T.mean(self._q_func) + (1e-5 * lasagne.regularization.regularize_network_params(
        # self._l_outA, lasagne.regularization.l2)), params, 
        #              self._learning_rate * -T.mean(diff), self._rho, self._rms_epsilon)
        
        
        
        self._train = theano.function([], [loss, self._q_valsA], updates=updates,
                                      givens=givens)
        self._q_vals = theano.function([], self._q_valsA,
                                       givens={State: self._states_shared})
        
        self._bellman_error = theano.function(inputs=[State, Action, Reward, ResultState], outputs=diff, allow_input_downcast=True)
        
    def updateTargetModel(self):
        print ("Updating target Model")
        """
            Double Q learning now
        """
        all_paramsA = lasagne.layers.helper.get_all_param_values(self._l_outA)
        # all_paramsB = lasagne.layers.helper.get_all_param_values(self._l_outB)
        lasagne.layers.helper.set_all_param_values(self._l_outB, all_paramsA)
        # lasagne.layers.helper.set_all_param_values(self._l_outA, all_paramsB) 
    
    def train(self, states, actions, rewards, result_states):
        self._states_shared.set_value(states)
        self._next_states_shared.set_value(result_states)
        self._actions_shared.set_value(actions)
        self._rewards_shared.set_value(rewards)

        if (( self._updates % self._weight_update_steps) == 0):
            self.updateTargetModel()
        self._updates += 1
        loss, _ = self._train()
        return np.sqrt(loss)
    
    def q_value(self, state):
        # states = np.zeros((self._batch_size, self._self._state_length), dtype=theano.config.floatX)
        # states[0, ...] = state
        # state = [norm_state(state, self._state_bounds)]
        q_values = self.q_values(state)
        action_ = self.predict(state)
        # print ("q_values: " + str(q_values) + " Action: " + str(action_))
        original_val = q_values[action_]
        return original_val
    
    def predict(self, state):
        # state = [norm_state(state, self._state_bounds)]
        q_vals = self.q_values(state)
        return np.argmax(q_vals)
    
    def q_values(self, state):
        # states = np.zeros((self._batch_size, self._self._state_length), dtype=theano.config.floatX)
        # states[0, ...] = state
        state = [norm_state(state, self._state_bounds)]
        self._states_shared.set_value(state)
        return self._q_vals()[0]
    
    def bellman_error(self, state, action, reward, result_state):
        return self._bellman_error(state, action, reward, result_state)
