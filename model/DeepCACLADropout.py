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

class DeepCACLADropout(DeepCACLA):
    
    def __init__(self, n_in, n_out, state_bounds, action_bounds, reward_bound, settings_):

        super(DeepCACLADropout,self).__init__(n_in, n_out, state_bounds, action_bounds, reward_bound, settings_)
        
        batch_size=self.getSettings()['batch_size']
        dropout_p=self.getSettings()['dropout_p']
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
        inputLayerA = lasagne.layers.DropoutLayer(inputLayerA, p=dropout_p, rescale=True)
        
        l_hid1A = lasagne.layers.DenseLayer(
                inputLayerA, num_units=128,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        l_hid1A = lasagne.layers.DropoutLayer(l_hid1A, p=dropout_p, rescale=True)
        
        l_hid2A = lasagne.layers.DenseLayer(
                l_hid1A, num_units=64,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        l_hid2A = lasagne.layers.DropoutLayer(l_hid2A, p=dropout_p, rescale=True)
        
        l_hid3A = lasagne.layers.DenseLayer(
                l_hid2A, num_units=32,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        l_hid3A = lasagne.layers.DropoutLayer(l_hid3A, p=dropout_p, rescale=True)
    
        self._l_outA = lasagne.layers.DenseLayer(
                l_hid3A, num_units=1,
                nonlinearity=lasagne.nonlinearities.linear)
        # self._b_o = init_b_weights((n_out,))
        inputLayerActA = lasagne.layers.InputLayer((None, self._state_length), State)
        inputLayerActA = lasagne.layers.DropoutLayer(inputLayerActA, p=dropout_p, rescale=True)
        
        l_hid1ActA = lasagne.layers.DenseLayer(
                inputLayerActA, num_units=128,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        l_hid1ActA = lasagne.layers.DropoutLayer(l_hid1ActA, p=dropout_p, rescale=True)
        
        l_hid2ActA = lasagne.layers.DenseLayer(
                l_hid1ActA, num_units=64,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        l_hid2ActA = lasagne.layers.DropoutLayer(l_hid2ActA, p=dropout_p, rescale=True)
        
        l_hid3ActA = lasagne.layers.DenseLayer(
                l_hid2ActA, num_units=32,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        l_hid3ActA = lasagne.layers.DropoutLayer(l_hid3ActA, p=dropout_p, rescale=True)
    
        self._l_outActA = lasagne.layers.DenseLayer(
                l_hid3ActA, num_units=self._action_length,
                nonlinearity=lasagne.nonlinearities.linear)
        # self._b_o = init_b_weights((n_out,))
        
        # self.updateTargetModel()
        inputLayerB = lasagne.layers.InputLayer((None, self._state_length), State)
        inputLayerB = lasagne.layers.DropoutLayer(inputLayerB, p=dropout_p, rescale=True)
        
        l_hid1B = lasagne.layers.DenseLayer(
                inputLayerB, num_units=128,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        l_hid1B = lasagne.layers.DropoutLayer(l_hid1B, p=dropout_p, rescale=True)
        
        l_hid2B = lasagne.layers.DenseLayer(
                l_hid1B, num_units=64,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        l_hid2B = lasagne.layers.DropoutLayer(l_hid2B, p=dropout_p, rescale=True)
    
        l_hid3B = lasagne.layers.DenseLayer(
                l_hid2B, num_units=32,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        l_hid3B = lasagne.layers.DropoutLayer(l_hid3B, p=dropout_p, rescale=True)
        
        self._l_outB = lasagne.layers.DenseLayer(
                l_hid3B, num_units=1,
                nonlinearity=lasagne.nonlinearities.linear)
        
        inputLayerActB = lasagne.layers.InputLayer((None, self._state_length), State)
        inputLayerActB = lasagne.layers.DropoutLayer(inputLayerActB, p=dropout_p, rescale=True)
        
        l_hid1ActB = lasagne.layers.DenseLayer(
                inputLayerActB, num_units=128,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        l_hid1ActB = lasagne.layers.DropoutLayer(l_hid1ActB, p=dropout_p, rescale=True)
        
        l_hid2ActB = lasagne.layers.DenseLayer(
                l_hid1ActB, num_units=64,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        l_hid2ActB = lasagne.layers.DropoutLayer(l_hid2ActB, p=dropout_p, rescale=True)
    
        l_hid3ActB = lasagne.layers.DenseLayer(
                l_hid2ActB, num_units=32,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        l_hid3ActB = lasagne.layers.DropoutLayer(l_hid3ActB, p=dropout_p, rescale=True)
        
        self._l_outActB = lasagne.layers.DenseLayer(
                l_hid3ActB, num_units=self._action_length,
                nonlinearity=lasagne.nonlinearities.linear)
        
          # print ("Initial W " + str(self._w_o.get_value()) )
        
        self._learning_rate = self.getSettings()['learning_rate']
        self._discount_factor= self.getSettings()['discount_factor']
        self._rho = self.getSettings()['rho']
        self._rms_epsilon = self.getSettings()['rms_epsilon']
        
        self._weight_update_steps= self.getSettings()['steps_until_target_network_update']
        self._regularization_weight= self.getSettings()['regularization_weight']
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
            np.zeros((batch_size, self._action_length), dtype=theano.config.floatX),
            )
        
        self._q_valsA_drop = lasagne.layers.get_output(self._l_outA, State)
        self._q_valsA = lasagne.layers.get_output(self._l_outA, State, deterministic=True)
        self._q_valsB = lasagne.layers.get_output(self._l_outB, ResultState)
        
        self._q_valsActA_drop = lasagne.layers.get_output(self._l_outActA, State, deterministic=False)
        self._q_valsActA = lasagne.layers.get_output(self._l_outActA, State, deterministic=True)
        self._q_valsActB = lasagne.layers.get_output(self._l_outActB, State)
        
        self._q_func = self._q_valsA
        self._q_func_drop = self._q_valsA_drop
        self._q_funcAct = self._q_valsActA
        self._q_funcAct_drop = self._q_valsActA_drop
        # self._q_funcAct = theano.function(inputs=[State], outputs=self._q_valsActA, allow_input_downcast=True)
        
        target = (Reward + self._discount_factor * self._q_valsB)
        diff = target - self._q_valsA
        diff_drop = target - self._q_valsA_drop 
        loss = 0.5 * diff ** 2 + (self._regularization_weight * lasagne.regularization.regularize_network_params(
        self._l_outA, lasagne.regularization.l2))
        loss = T.mean(loss)
        loss_drop = T.mean(0.5 * diff_drop ** 2 + (self._regularization_weight * lasagne.regularization.regularize_network_params(
        self._l_outA, lasagne.regularization.l2)))
        
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
        updates_ = lasagne.updates.rmsprop(T.mean(self._q_func_drop) + (self._regularization_weight * lasagne.regularization.regularize_network_params(
        self._l_outA, lasagne.regularization.l2)), params, 
                    self._learning_rate * -T.mean(diff_drop), self._rho, self._rms_epsilon)
        
        
        # actDiff1 = (Action - self._q_valsActB) #TODO is this correct?
        # actDiff = (actDiff1 - (Action - self._q_valsActA))
        actDiff = ((Action - self._q_valsActA)) # Target network does not work well here?
        actDiff_drop = ((Action - self._q_valsActA_drop)) # Target network does not work well here?
        actLoss = 0.5 * actDiff ** 2 + (self._regularization_weight * lasagne.regularization.regularize_network_params( self._l_outActA, lasagne.regularization.l2))
        actLoss = T.sum(actLoss)/float(batch_size)
        actLoss_drop = (T.sum(0.5 * actDiff_drop ** 2 + (self._regularization_weight * 
                            lasagne.regularization.regularize_network_params( self._l_outActA, lasagne.regularization.l2)))/
                            float(batch_size))
        
        # actionUpdates = lasagne.updates.rmsprop(actLoss + 
        #    (1e-4 * lasagne.regularization.regularize_network_params(
        #        self._l_outActA, lasagne.regularization.l2)), actionParams, 
        #            self._learning_rate * 0.01 * (-actLoss), self._rho, self._rms_epsilon)
        
        actionUpdates = lasagne.updates.rmsprop(T.mean(self._q_funcAct_drop) + 
          (self._regularization_weight * lasagne.regularization.regularize_network_params(
              self._l_outActA, lasagne.regularization.l2)), actionParams, 
                  self._learning_rate * 0.5 * (-T.sum(actDiff_drop)/float(batch_size)), self._rho, self._rms_epsilon)
        
        
        
        self._train = theano.function([], [loss_drop, self._q_valsA_drop], updates=updates_, givens=givens_)
        self._trainActor = theano.function([], [actLoss_drop, self._q_valsActA_drop], updates=actionUpdates, givens=actGivens)
        self._q_val = theano.function([], self._q_valsA,
                                       givens={State: self._states_shared})

        self._q_action_drop = theano.function([], self._q_valsActA_drop,
                                       givens={State: self._states_shared})
        self._q_action = theano.function([], self._q_valsActA,
                                       givens={State: self._states_shared})
        self._bellman_error_drop = theano.function(inputs=[State, Reward, ResultState], outputs=diff_drop, allow_input_downcast=True)
        
        self._bellman_error = theano.function(inputs=[State, Reward, ResultState], outputs=diff, allow_input_downcast=True)
        # self._diffs = theano.function(input=[State])
        
    def updateTargetModel(self):
        print ("Updating target Model")
        """
            Target model updates
        """
        all_paramsA = lasagne.layers.helper.get_all_param_values(self._l_outA)
        all_paramsActA = lasagne.layers.helper.get_all_param_values(self._l_outActA)
        lasagne.layers.helper.set_all_param_values(self._l_outB, all_paramsA)
        lasagne.layers.helper.set_all_param_values(self._l_outActB, all_paramsActA) 
        

    def getNetworkParameters(self):
        params = []
        params.append(lasagne.layers.helper.get_all_param_values(self._l_outA))
        params.append(lasagne.layers.helper.get_all_param_values(self._l_outActA))
        params.append(lasagne.layers.helper.get_all_param_values(self._l_outB))
        params.append(lasagne.layers.helper.get_all_param_values(self._l_outActB))
        return params
    
    def setNetworkParameters(self, params):
        lasagne.layers.helper.set_all_param_values(self._l_outA, params[0])
        lasagne.layers.helper.set_all_param_values(self._l_outActA, params[1])
        lasagne.layers.helper.set_all_param_values(self._l_outB, params[2])
        lasagne.layers.helper.set_all_param_values(self._l_outActB, params[3])
    
    
    def predict(self, state, deterministic_=True):
        # states = np.zeros((self._batch_size, self._state_length), dtype=theano.config.floatX)
        # states[0, ...] = state
        state = [norm_state(state, self._state_bounds)]
        self._states_shared.set_value(state)
        # action_ = lasagne.layers.get_output(self._l_outActA, state, deterministic=deterministic_).mean()
        # action_ = scale_action(self._q_action()[0], self._action_bounds)
        # if deterministic_:
        action_ = scale_action(self._q_action()[0], self._action_bounds)
        # else:
        # action_ = scale_action(self._q_action()[0], self._action_bounds)
        # action_ = q_valsActA[0]
        return action_
    
    def predictWithDropout(self, state, deterministic_=True):
        # states = np.zeros((self._batch_size, self._state_length), dtype=theano.config.floatX)
        # states[0, ...] = state
        state = [norm_state(state, self._state_bounds)]
        self._states_shared.set_value(state)
        # action_ = lasagne.layers.get_output(self._l_outActA, state, deterministic=deterministic_).mean()
        # action_ = scale_action(self._q_action()[0], self._action_bounds)
        # if deterministic_:
        action_ = scale_action(self._q_action_drop()[0], self._action_bounds)
        # else:
        # action_ = scale_action(self._q_action()[0], self._action_bounds)
        # action_ = q_valsActA[0]
        return action_
    
    def q_value(self, state):
        # states = np.zeros((self._batch_size, self._state_length), dtype=theano.config.floatX)
        # states[0, ...] = state
        state = [norm_state(state, self._state_bounds)]
        self._states_shared.set_value(state)
        return self._q_val()[0]
    
    def bellman_error(self, state, action, reward, result_state):
        return self._bellman_error(state, reward, result_state)
