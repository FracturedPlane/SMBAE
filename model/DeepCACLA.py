import theano
from theano import tensor as T
import numpy as np
import lasagne
import sys
sys.path.append('../')
from model.ModelUtil import *

# For debugging
# theano.config.mode='FAST_COMPILE'
from model.AgentInterface import AgentInterface

class DeepCACLA(AgentInterface):
    
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

        num_additional_layers=4
        l_hid1A = inputLayerA
        for i in range(num_additional_layers):
            l_hid1A = lasagne.layers.DenseLayer(
                    l_hid1A, num_units=128,
                    nonlinearity=lasagne.nonlinearities.leaky_rectify)
        
        l_hid2A = lasagne.layers.DenseLayer(
                l_hid1A, num_units=64,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        
        l_hid3A = lasagne.layers.DenseLayer(
                l_hid2A, num_units=32,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
    
        self._l_outA = lasagne.layers.DenseLayer(
                l_hid3A, num_units=1,
                nonlinearity=lasagne.nonlinearities.linear)
        # self._b_o = init_b_weights((n_out,))
        inputLayerActA = lasagne.layers.InputLayer((None, self._state_length), State)
        
        l_hid1ActA = inputLayerActA 
        for i in range(num_additional_layers):
            l_hid1ActA = lasagne.layers.DenseLayer(
                    l_hid1ActA, num_units=128,
                    nonlinearity=lasagne.nonlinearities.leaky_rectify)
        
        l_hid2ActA = lasagne.layers.DenseLayer(
                l_hid1ActA, num_units=64,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        
        l_hid3ActA = lasagne.layers.DenseLayer(
                l_hid2ActA, num_units=32,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
    
        self._l_outActA = lasagne.layers.DenseLayer(
                l_hid3ActA, num_units=self._action_length,
                nonlinearity=lasagne.nonlinearities.linear)
        # self._b_o = init_b_weights((n_out,))
        
        # self.updateTargetModel()
        inputLayerB = lasagne.layers.InputLayer((None, self._state_length), State)
        
        l_hid1B = inputLayerB 
        for i in range(num_additional_layers):
            l_hid1B = lasagne.layers.DenseLayer(
                    l_hid1B, num_units=128,
                    nonlinearity=lasagne.nonlinearities.leaky_rectify)
        
        l_hid2B = lasagne.layers.DenseLayer(
                l_hid1B, num_units=64,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
    
        l_hid3B = lasagne.layers.DenseLayer(
                l_hid2B, num_units=32,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        
        self._l_outB = lasagne.layers.DenseLayer(
                l_hid3B, num_units=1,
                nonlinearity=lasagne.nonlinearities.linear)
        
        inputLayerActB = lasagne.layers.InputLayer((None, self._state_length), State)
        
        l_hid1ActB = inputLayerActB 
        for i in range(num_additional_layers):
            l_hid1ActB = lasagne.layers.DenseLayer(
                    l_hid1ActB, num_units=128,
                    nonlinearity=lasagne.nonlinearities.leaky_rectify)
        
        l_hid2ActB = lasagne.layers.DenseLayer(
                l_hid1ActB, num_units=64,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
    
        l_hid3ActB = lasagne.layers.DenseLayer(
                l_hid2ActB, num_units=32,
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
        # updates_ = lasagne.updates.rmsprop(loss, params, self._learning_rate, self._rho,
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
        
        actionUpdates = lasagne.updates.rmsprop(actLoss + 
           (self._decay_weight * lasagne.regularization.regularize_network_params(
               self._l_outActA, lasagne.regularization.l2)), actionParams, 
                   self._learning_rate * (actLoss), self._rho, self._rms_epsilon)
        
        # actionUpdates = lasagne.updates.rmsprop(T.mean(self._q_funcAct) + 
        # (self._decay_weight * lasagne.regularization.regularize_network_params(
        #      self._l_outActA, lasagne.regularization.l2)), actionParams, 
        #          self._learning_rate * 0.5 * (-T.sum(actDiff)/float(batch_size)), self._rho, self._rms_epsilon)
        
        
        
        self._train = theano.function([], [loss, self._q_valsA], updates=updates_, givens=givens_)
        self._trainActor = theano.function([], [actLoss, self._q_valsActA], updates=actionUpdates, givens=actGivens)
        self._q_val = theano.function([], self._q_valsA,
                                       givens={State: self._states_shared})
        self._q_action = theano.function([], self._q_valsActA,
                                       givens={State: self._states_shared})
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
        
    def trainCritic(self, states, actions, rewards, result_states):
        self._states_shared.set_value(states)
        self._next_states_shared.set_value(result_states)
        self._actions_shared.set_value(actions)
        self._rewards_shared.set_value(rewards)
        # print ("Performing Critic trainning update")
        if (( self._updates % self._weight_update_steps) == 0):
            self.updateTargetModel()
        self._updates += 1
        loss, _ = self._train()
        
        return loss
    
    def trainActor(self, states, actions, rewards, result_states):
        self._states_shared.set_value(states)
        self._next_states_shared.set_value(result_states)
        self._actions_shared.set_value(actions)
        self._rewards_shared.set_value(rewards)
        # print ("Performing Critic trainning update")
        if (( self._updates % self._weight_update_steps) == 0):
            self.updateTargetModel()
        self._updates += 1
        # loss, _ = self._train()
        lossActor = 0
        
        diff_ = self._bellman_error(states, rewards, result_states)
        # print ("Diff")
        # print (diff_)
        tmp_states=[]
        tmp_result_states=[]
        tmp_actions=[]
        tmp_rewards=[]
        for i in range(len(diff_)):
            # print ("Performing Actor trainning update")
            
            if ( diff_[i] > 0.0):
                # print (states[i])
                tmp_states.append(states[i])
                tmp_result_states.append(result_states[i])
                tmp_actions.append(actions[i])
                tmp_rewards.append(rewards[i])
                
        if (len(tmp_actions) > 0):
            self._states_shared.set_value(np.array(tmp_states))
            self._next_states_shared.set_value(tmp_result_states)
            self._actions_shared.set_value(tmp_actions)
            self._rewards_shared.set_value(tmp_rewards)
            lossActor, _ = self._trainActor()
            # print ("Length of positive actions: " + str(len(tmp_actions)))
            # return np.sqrt(lossActor);
        return lossActor
    
    def train(self, states, actions, rewards, result_states):
        loss = self.trainCritic(states, actions, rewards, result_states)
        lossActor = self.trainActor(states, actions, rewards, result_states)
        return loss
    
    def predict(self, state):
        # states = np.zeros((self._batch_size, self._state_length), dtype=theano.config.floatX)
        # states[0, ...] = state
        state = [norm_state(state, self._state_bounds)]
        self._states_shared.set_value(state)
        # q_valsActA = lasagne.layers.get_output(self._l_outActA, state).eval()
        action_ = scale_action(self._q_action()[0], self._action_bounds)
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
