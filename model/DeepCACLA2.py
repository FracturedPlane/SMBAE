import theano
from theano import tensor as T
import numpy as np
import lasagne
import sys
sys.path.append('../')
from model.ModelUtil import *

# For debugging
# theano.config.mode='FAST_COMPILE'
from model.DeepCACLA import DeepCACLA

class DeepCACLA2(DeepCACLA):
    """
        This network uses the DeepCALCA algorithm but the network design is differnt
        Instead of sing two different networks this network shares the same 
        initial structure. Maybe this will help the network learn and share a 
        representation of the state.
    """
    
    def __init__(self, n_in, n_out, state_bounds, action_bounds, reward_bound, settings_):

        super(DeepCACLA,self).__init__(n_in, n_out, state_bounds, action_bounds, reward_bound, settings_)
        
        batch_size=self.getSettings()['batch_size']
        # data types for model
        State = T.dmatrix("State")
        State.tag.test_value = np.random.rand(batch_size,self._state_length)
        ResultState = T.dmatrix("ResultState")
        ResultState.tag.test_value = np.random.rand(batch_size,self._state_length)
        Reward = T.col("Reward")
        Reward.tag.test_value = np.random.rand(batch_size,1)
        Action = T.dmatrix("Action")
        Action.tag.test_value = np.random.rand(batch_size, self._action_length)
        # create a small convolutional neural network
        inputLayerA = lasagne.layers.InputLayer((None, self._state_length), State)

        l_hid1A = lasagne.layers.DenseLayer(
                inputLayerA, num_units=128,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        
        l_hid2A = lasagne.layers.DenseLayer(
                l_hid1A, num_units=64,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        
        l_hid3A = lasagne.layers.DenseLayer(
                l_hid2A, num_units=32,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        
        l_hid4A = lasagne.layers.DenseLayer(
                l_hid3A, num_units=32,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        
        l_hid5A = lasagne.layers.DenseLayer(
                l_hid4A, num_units=16,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
    
        self._l_outA = lasagne.layers.DenseLayer(
                l_hid5A, num_units=1,
                nonlinearity=lasagne.nonlinearities.linear)
        # self._b_o = init_b_weights((n_out,))
        
        l_hid4A = lasagne.layers.DenseLayer(
                l_hid3A, num_units=64,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        
        l_hid3ActA = lasagne.layers.DenseLayer(
                l_hid4A, num_units=32,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
    
        self._l_outActA = lasagne.layers.DenseLayer(
                l_hid3ActA, num_units=self._action_length,
                nonlinearity=lasagne.nonlinearities.linear)
        # self._b_o = init_b_weights((n_out,))
        
        # self.updateTargetModel()
        inputLayerB = lasagne.layers.InputLayer((None, self._state_length), State)

        l_hid1B = lasagne.layers.DenseLayer(
                inputLayerB, num_units=128,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        
        l_hid2B = lasagne.layers.DenseLayer(
                l_hid1B, num_units=64,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        
        l_hid3B = lasagne.layers.DenseLayer(
                l_hid2B, num_units=32,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        
        l_hid4B = lasagne.layers.DenseLayer(
                l_hid3B, num_units=32,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        
        l_hid5B = lasagne.layers.DenseLayer(
                l_hid4B, num_units=16,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
    
        self._l_outB = lasagne.layers.DenseLayer(
                l_hid5B, num_units=1,
                nonlinearity=lasagne.nonlinearities.linear)
        # self._b_o = init_b_weights((n_out,))
        
        l_hid4B = lasagne.layers.DenseLayer(
                l_hid3B, num_units=64,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        
        l_hid3ActB = lasagne.layers.DenseLayer(
                l_hid4B, num_units=32,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
    
        self._l_outActB = lasagne.layers.DenseLayer(
                l_hid3ActB, num_units=self._action_length,
                nonlinearity=lasagne.nonlinearities.linear)
        
          # print ("Initial W " + str(self._w_o.get_value()) )
        
        self._learning_rate = self.getSettings()['learning_rate']
        self._discount_factor= self.getSettings()['discount_factor']
        self._rho = self.getSettings()['rho']
        self._rms_epsilon = self.getSettings()['rms_epsilon']
        
        self._weight_update_steps=self.getSettings()['steps_until_target_network_update']
        self._updates=0
        self._decay_weight=self.getSettings()['regularization_weight']
        
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
            np.zeros((batch_size, self._action_length), dtype=theano.config.floatX),
            )
        
        self._q_valsA = lasagne.layers.get_output(self._l_outA, State)
        self._q_valsB = lasagne.layers.get_output(self._l_outB, ResultState)
        
        self._q_valsActA = lasagne.layers.get_output(self._l_outActA, State)
        self._q_valsActB = lasagne.layers.get_output(self._l_outActB, State)
        
        self._q_func = self._q_valsA
        self._q_funcAct = self._q_valsActA
        # self._q_funcAct = theano.function(inputs=[State], outputs=self._q_valsActA, allow_input_downcast=True)
        
        target = (Reward + self._discount_factor * self._q_valsB)
        diff = target - self._q_valsA
        loss = 0.5 * diff ** 2 + (self._decay_weight * lasagne.regularization.regularize_network_params(
        self._l_outA, lasagne.regularization.l2))
        loss = T.mean(loss)
        
        params = lasagne.layers.helper.get_all_params(self._l_outA)
        actionParams = lasagne.layers.helper.get_all_params(self._l_outActA)
        givens_ = {
            State: self._states_shared,
            ResultState: self._next_states_shared,
            Reward: self._rewards_shared,
            # Action: self._actions_shared,
        }
        actGivens = {
            State: self._states_shared,
            # ResultState: self._next_states_shared,
            # Reward: self._rewards_shared,
            Action: self._actions_shared,
        }
        
        # SGD update
        #updates_ = lasagne.updates.rmsprop(loss, params, self._learning_rate, self._rho,
        #                                    self._rms_epsilon)
        # TD update
        updates_ = lasagne.updates.rmsprop(T.mean(self._q_func) + (self._decay_weight * lasagne.regularization.regularize_network_params(
        self._l_outA, lasagne.regularization.l2)), params, 
                    self._learning_rate * -T.mean(diff), self._rho, self._rms_epsilon)
        
        
        # actDiff1 = (Action - self._q_valsActB) #TODO is this correct?
        # actDiff = (actDiff1 - (Action - self._q_valsActA))
        actDiff = ((Action - self._q_valsActA)) # Target network does not work well here?
        actLoss = 0.5 * actDiff ** 2 + (self._decay_weight * lasagne.regularization.regularize_network_params( self._l_outActA, lasagne.regularization.l2))
        actLoss = T.sum(actLoss)/float(batch_size)
        
        # actionUpdates = lasagne.updates.rmsprop(actLoss + 
        #    (1e-4 * lasagne.regularization.regularize_network_params(
        #        self._l_outActA, lasagne.regularization.l2)), actionParams, 
        #            self._learning_rate * 0.01 * (-actLoss), self._rho, self._rms_epsilon)
        
        actionUpdates = lasagne.updates.rmsprop(T.mean(self._q_funcAct) + 
          (self._decay_weight * lasagne.regularization.regularize_network_params(
              self._l_outActA, lasagne.regularization.l2)), actionParams, 
                  self._learning_rate * 0.5 * (-T.sum(actDiff)/float(batch_size)), self._rho, self._rms_epsilon)
        
        
        
        self._train = theano.function([], [loss, self._q_valsA], updates=updates_, givens=givens_)
        self._trainActor = theano.function([], [actLoss, self._q_valsActA], updates=actionUpdates, givens=actGivens)
        self._q_val = theano.function([], self._q_valsA,
                                       givens={State: self._states_shared})
        self._q_action = theano.function([], self._q_valsActA,
                                       givens={State: self._states_shared})
        self._bellman_error = theano.function(inputs=[State, Reward, ResultState], outputs=diff, allow_input_downcast=True)
        # self._diffs = theano.function(input=[State])
        
   