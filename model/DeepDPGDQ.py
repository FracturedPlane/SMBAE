import theano
from theano import tensor as T
import numpy as np
import lasagne
import sys
sys.path.append('../')
from model.ModelUtil import *
from model.DeepDPG import DeepDPG
from model.DeepDPG import rmsprop

# For debugging
# theano.config.mode='FAST_COMPILE'
from collections import OrderedDict

class DeepDPGDQ(DeepDPG):
    
    def __init__(self, n_in, n_out, state_bounds, action_bounds, reward_bound, settings_):
        """
            In order to get this to work we need to be careful not to update the actor parameters
            when updating the critic. This can be an issue when concatenating networks together.
            The first first network becomes a part of the second. However you can still access the first
            network by itself but an updates on the second network will effect the first network.
            Care needs to be taken to make sure only the parameters of the second network are updated.
        """
        
        super(DeepDPGDQ,self).__init__(n_in, n_out, state_bounds, action_bounds, reward_bound, settings_)

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
        inputLayerActA = lasagne.layers.InputLayer((None, self._state_length), State)
        
        l_hid1ActA = lasagne.layers.DenseLayer(
                inputLayerActA, num_units=128,
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
        
        inputLayerA = lasagne.layers.InputLayer((None, self._state_length), State)

        concatLayer = lasagne.layers.ConcatLayer([inputLayerA, self._l_outActA])
        
        l_hid1A = lasagne.layers.DenseLayer(
                concatLayer, num_units=128,
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
        # self._b_o = init_b_weights((self._action_length,))

        # self.updateTargetModel()
        inputLayerActB = lasagne.layers.InputLayer((None, self._state_length), State)
        
        l_hid1ActB = lasagne.layers.DenseLayer(
                inputLayerActB, num_units=128,
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

        inputLayerB = lasagne.layers.InputLayer((None, self._state_length), State)
        concatLayerB = lasagne.layers.ConcatLayer([inputLayerB, self._l_outActB])
        
        l_hid1B = lasagne.layers.DenseLayer(
                concatLayerB, num_units=128,
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
        
        ################################################################################\
        inputLayerActA = lasagne.layers.InputLayer((None, self._state_length), State)
        
        l_hid1ActA = lasagne.layers.DenseLayer(
                inputLayerActA, num_units=128,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        
        l_hid2ActA = lasagne.layers.DenseLayer(
                l_hid1ActA, num_units=64,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        
        l_hid3ActA = lasagne.layers.DenseLayer(
                l_hid2ActA, num_units=32,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
    
        self._l_outActATarget = lasagne.layers.DenseLayer(
                l_hid3ActA, num_units=self._action_length,
                nonlinearity=lasagne.nonlinearities.linear)
        
        inputLayerA = lasagne.layers.InputLayer((None, self._state_length), State)

        concatLayer = lasagne.layers.ConcatLayer([inputLayerA, self._l_outActA])
        
        l_hid1A = lasagne.layers.DenseLayer(
                concatLayer, num_units=128,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        
        l_hid2A = lasagne.layers.DenseLayer(
                l_hid1A, num_units=64,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        
        l_hid3A = lasagne.layers.DenseLayer(
                l_hid2A, num_units=32,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
    
        self._l_outATarget = lasagne.layers.DenseLayer(
                l_hid3A, num_units=1,
                nonlinearity=lasagne.nonlinearities.linear)
        # self._b_o = init_b_weights((self._action_length,))

        # self.updateTargetModel()
        inputLayerActB = lasagne.layers.InputLayer((None, self._state_length), State)
        
        l_hid1ActB = lasagne.layers.DenseLayer(
                inputLayerActB, num_units=128,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        
        l_hid2ActB = lasagne.layers.DenseLayer(
                l_hid1ActB, num_units=64,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
    
        l_hid3ActB = lasagne.layers.DenseLayer(
                l_hid2ActB, num_units=32,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        
        self._l_outActBTarget = lasagne.layers.DenseLayer(
                l_hid3ActB, num_units=self._action_length,
                nonlinearity=lasagne.nonlinearities.linear)

        inputLayerB = lasagne.layers.InputLayer((None, self._state_length), State)
        concatLayerB = lasagne.layers.ConcatLayer([inputLayerB, self._l_outActB])
        
        l_hid1B = lasagne.layers.DenseLayer(
                concatLayerB, num_units=128,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        
        l_hid2B = lasagne.layers.DenseLayer(
                l_hid1B, num_units=64,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        
        l_hid3B = lasagne.layers.DenseLayer(
                l_hid2B, num_units=32,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
    
        self._l_outBTarget = lasagne.layers.DenseLayer(
                l_hid3B, num_units=1,
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
        
        self._q_valsActA = lasagne.layers.get_output(self._l_outActA, State)
        self._q_valsActB = lasagne.layers.get_output(self._l_outActB, State)
        # self._q_valsActB2 = lasagne.layers.get_output(self._l_outActB, State)
       
        
        inputs_ = {
            State: self._states_shared,
            Action: self._q_valsActA,
        }
        self._q_valsA = lasagne.layers.get_output(self._l_outA, inputs_)
        inputs_ = {
            ResultState: self._next_states_shared,
            Action: self._q_valsActB,
        }
        self._q_valsA_B = lasagne.layers.get_output(self._l_outBTarget, inputs_)
        inputs_ = {
            State: self._states_shared,
            Action: self._q_valsActB,
        }
        self._q_valsB = lasagne.layers.get_output(self._l_outB, inputs_)
        inputs_ = {
            State: self._next_states_shared,
            Action: self._q_valsActA,
        }
        self._q_valsB_A = lasagne.layers.get_output(self._l_outATarget, inputs_)
        
        
        self._q_func = self._q_valsA
        self._q_funcAct = self._q_valsActA
        self._q_funcB = self._q_valsB
        self._q_funcActB = self._q_valsActB2
        
        # self._q_funcAct = theano.function(inputs=[State], outputs=self._q_valsActA, allow_input_downcast=True)
        
        self._target = (Reward + self._discount_factor * self._q_valsA_B)
        self._diff = self._target - self._q_valsA
        
        self._targetB = (Reward + self._discount_factor * self._q_valsB_A)
        self._diffB = self._target - self._q_valsB
        
        self._loss = 0.5 * self._diff ** 2 + (self._decay_weight * lasagne.regularization.regularize_network_params(
        self._l_outA, lasagne.regularization.l2))
        self._loss = T.mean(self._loss)
        
        self._lossB = 0.5 * self._diffB ** 2 + (self._decay_weight * lasagne.regularization.regularize_network_params(
        self._l_outB, lasagne.regularization.l2))
        self._lossB = T.mean(self._lossB)
    
        # assert len(lasagne.layers.helper.get_all_params(self._l_outA)) == 16
        # Need to remove the action layers from these params
        self._params = lasagne.layers.helper.get_all_params(self._l_outA)[-len(lasagne.layers.helper.get_all_params(self._l_outActA)):]
        self._paramsB = lasagne.layers.helper.get_all_params(self._l_outB)[-len(lasagne.layers.helper.get_all_params(self._l_outActB)):] 
        print ("******Number of Layers is: " + str(len(lasagne.layers.helper.get_all_params(self._l_outA))))
        print ("******Number of Action Layers is: " + str(len(lasagne.layers.helper.get_all_params(self._l_outActA))))
        self._actionParams = lasagne.layers.helper.get_all_params(self._l_outActA)
        self._actionParamsB = lasagne.layers.helper.get_all_params(self._l_outActB)
        self._givens_ = {
            State: self._states_shared,
            # ResultState: self._next_states_shared,
            Reward: self._rewards_shared,
            # Action: self._actions_shared,
        }
        self._actGivens = {
            State: self._states_shared,
            # ResultState: self._next_states_shared,
            # Reward: self._rewards_shared,
            # Action: self._actions_shared,
        }
        
        # SGD update
        #updates_ = rmsprop(loss, params, self._learning_rate, self._rho,
        #                                    self._rms_epsilon)
        # TD update
        # minimize Value function error
        self._updates_ = rmsprop(T.mean(self._q_func) + (self._decay_weight * lasagne.regularization.regularize_network_params(
        self._l_outA, lasagne.regularization.l2)), self._params, 
                    self._learning_rate * -T.mean(self._diff), self._rho, self._rms_epsilon)
        
        self._updates_B = rmsprop(T.mean(self._q_funcB) + (self._decay_weight * lasagne.regularization.regularize_network_params(
        self._l_outB, lasagne.regularization.l2)), self._paramsB, 
                    self._learning_rate * -T.mean(self._diffB), self._rho, self._rms_epsilon)
        
        
        # actDiff1 = (Action - self._q_valsActB) #TODO is this correct?
        # actDiff = (actDiff1 - (Action - self._q_valsActA))
        # actDiff = ((Action - self._q_valsActB2)) # Target network does not work well here?
        #self._actDiff = ((Action - self._q_valsActA)) # Target network does not work well here?
        #self._actLoss = 0.5 * self._actDiff ** 2 + (1e-4 * lasagne.regularization.regularize_network_params( self._l_outActA, lasagne.regularization.l2))
        #self._actLoss = T.mean(self._actLoss)
        
        # actionUpdates = rmsprop(actLoss + 
        #    (1e-4 * lasagne.regularization.regularize_network_params(
        #        self._l_outActA, lasagne.regularization.l2)), actionParams, 
        #            self._learning_rate * 0.01 * (-actLoss), self._rho, self._rms_epsilon)
        
        # Maximize wrt q function
        
        # theano.gradient.grad_clip(x, lower_bound, upper_bound) # // TODO
        self._actionUpdates = rmsprop(T.mean(self._q_func) + 
          (self._decay_weight * lasagne.regularization.regularize_network_params(
              self._l_outActA, lasagne.regularization.l2)), self._actionParams, 
                  self._learning_rate * 0.1, self._rho, self._rms_epsilon)
        
        
        self._actionUpdatesB = rmsprop(T.mean(self._q_funcB) + 
          (self._decay_weight * lasagne.regularization.regularize_network_params(
              self._l_outActB, lasagne.regularization.l2)), self._actionParamsB, 
                  self._learning_rate * 0.1, self._rho, self._rms_epsilon)
        
        
        
        
        self._train = theano.function([], [self._loss, self._q_valsA], updates=self._updates_, givens=self._givens_)
        self._trainB = theano.function([], [self._lossB, self._q_valsB], updates=self._updates_B, givens=self._givens_)
        self._trainActor = theano.function([], [self._q_valsA], updates=self._actionUpdates, givens=self._actGivens)
        self._trainActorB = theano.function([], [self._q_valsB], updates=self._actionUpdatesB, givens=self._actGivens)
        self._q_val = theano.function([], self._q_valsA, givens={State: self._states_shared})
        self._q_valB = theano.function([], self._q_valsB, givens={State: self._states_shared})
        self._q_action = theano.function([], self._q_valsActA,
                                       givens={State: self._states_shared})
        self._q_actionB = theano.function([], self._q_valsActB,
                                       givens={State: self._states_shared})
        # self._q_actionB = theano.function([], self._q_valsActB, givens={State: self._states_shared})
        
        inputs_ = [
                   State, 
                   Reward, 
                   # ResultState
                   ]
        self._bellman_error = theano.function(inputs=inputs_, outputs=self._diff, allow_input_downcast=True)
        self._bellman_errorB = theano.function(inputs=inputs_, outputs=self._diffB, allow_input_downcast=True)
        # self._diffs = theano.function(input=[State])
        
        # x = T.matrices('x')
        # z_lazy = ifelse(T.gt(self._q_val()[0][0], self._q_valB()[0][0]), self._q_action(), self._q_actionB())
        # self._f_lazyifelse = theano.function([], z_lazy,
        #                        mode=theano.Mode(linker='vm'))
        
            
    def updateTargetModel(self):
        # print ("Updating target Model")
        """
            Target model updates
        """
        
        all_paramsA = lasagne.layers.helper.get_all_param_values(self._l_outA)
        all_paramsATarget = lasagne.layers.helper.get_all_param_values(self._l_outATarget)
        lerp_weight = 0.001
        # vals = lasagne.layers.helper.get_all_param_values(self._l_outActA)
        
        all_params = []
        for paramsA, paramsATarget in zip(all_paramsA, all_paramsATarget):
            # print ("paramsA: " + str(paramsA))
            # print ("paramsB: " + str(paramsB))
            params = (lerp_weight * paramsA) + ((1.0 - lerp_weight) * paramsATarget)
            all_params.append(params)
        lasagne.layers.helper.set_all_param_values(self._l_outATarget, all_params)
        # lasagne.layers.helper.set_all_param_values(self._l_outActB, all_paramsAct) 
        
        all_paramsB = lasagne.layers.helper.get_all_param_values(self._l_outB)
        all_paramsBTarget = lasagne.layers.helper.get_all_param_values(self._l_outBTarget)
        lerp_weight = 0.001
        # vals = lasagne.layers.helper.get_all_param_values(self._l_outActA)
        
        all_params = []
        for paramsB, paramsBTarget in zip(all_paramsB, all_paramsBTarget):
            # print ("paramsA: " + str(paramsA))
            # print ("paramsB: " + str(paramsB))
            params = (lerp_weight * paramsB) + ((1.0 - lerp_weight) * paramsBTarget)
            all_params.append(params)
        lasagne.layers.helper.set_all_param_values(self._l_outBTarget, all_params)
        
    def getNetworkParameters(self):
        params = []
        params.append(lasagne.layers.helper.get_all_param_values(self._l_outA))
        params.append(lasagne.layers.helper.get_all_param_values(self._l_outActA))
        params.append(lasagne.layers.helper.get_all_param_values(self._l_outB))
        params.append(lasagne.layers.helper.get_all_param_values(self._l_outActB))
        params.append(lasagne.layers.helper.get_all_param_values(self._l_outATarget))
        params.append(lasagne.layers.helper.get_all_param_values(self._l_outActATarget))
        params.append(lasagne.layers.helper.get_all_param_values(self._l_outBTarget))
        params.append(lasagne.layers.helper.get_all_param_values(self._l_outActBTarget))
        return params
        
    def setNetworkParameters(self, params):
        lasagne.layers.helper.set_all_param_values(self._l_outA, params[0])
        lasagne.layers.helper.set_all_param_values(self._l_outActA, params[1])
        lasagne.layers.helper.set_all_param_values(self._l_outB, params[2])
        lasagne.layers.helper.set_all_param_values(self._l_outActB, params[3])
        lasagne.layers.helper.set_all_param_values(self._l_outATarget, params[4])
        lasagne.layers.helper.set_all_param_values(self._l_outActATarget, params[5])
        lasagne.layers.helper.set_all_param_values(self._l_outBTarget, params[6])
        lasagne.layers.helper.set_all_param_values(self._l_outActBTarget, params[7])
        
    def trainCritic(self, states, actions, rewards, result_states):
        self._states_shared.set_value(states)
        self._next_states_shared.set_value(result_states)
        self._actions_shared.set_value(actions)
        self._rewards_shared.set_value(rewards)
        
        
        if (( self._updates % self._weight_update_steps) == 0):
            self.updateTargetModel()
        self._updates += 1
        # loss, _ = self._train()
        loss = 0
        
        
        import random
        r = random.choice([0,1])
        if r == 0:
            loss, _ = self._train()
            
            diff_ = self._bellman_error(states, rewards)
        else:
            loss, _ = self._trainB()
            
            diff_ = self._bellman_errorB(states, rewards)
        return loss
        
    def trainActor(self, states, actions, rewards, result_states):
        self._states_shared.set_value(states)
        self._next_states_shared.set_value(result_states)
        self._actions_shared.set_value(actions)
        self._rewards_shared.set_value(rewards)
        
        import random
        loss = 0
        diff_ = 0
        
        r = random.choice([0,1])
        if r == 0:
            loss, _ = self._train()
            loss = self._trainActor()
            
            diff_ = self._bellman_error(states, rewards)
        else:
            loss, _ = self._trainB()
            loss = self._trainActorB()
            diff_ = self._bellman_errorB(states, rewards)
        
        if not all(np.isfinite(diff_)):
            print ("States: " + str(states) + " ResultsStates: " + str(result_states) + " Rewards: " + str(rewards) + " Actions: " + str(actions))
            print ("Bellman Error in DeepDPGDQ is Nan: " + str(diff_))
            sys.exit()
            
        return loss
        
    def train(self, states, actions, rewards, result_states):
        loss = self.trainCritic(states, actions, rewards, result_states)
        lossActor = self.trainActor(states, actions, rewards, result_states)
        return loss
    
    def predict(self, state):
        # states = np.zeros((self._batch_size, self._state_length), dtype=theano.config.floatX)
        # states[0, ...] = state
        # q_valsActA = lasagne.layers.get_output(self._l_outActA, state).eval()
        action = 0
        state = [norm_state(state, self._state_bounds)]
        self._states_shared.set_value(state)
        # print ("Q Value: " + str(self._q_val()[0]) )
    
        if self._q_val()[0][0] > self._q_valB()[0][0]:
            action = self._q_action()[0]
        else:
            action = self._q_actionB()[0]
        
        action_ = scale_action(action, self._action_bounds)
        # action_ = q_valsActA[0]
        return action_
    def q_value(self, state):
        # states = np.zeros((self._batch_size, self._state_length), dtype=theano.config.floatX)
        # states[0, ...] = state
        state = [norm_state(state, self._state_bounds)]
        self._states_shared.set_value(state)
        if self._q_val()[0][0] > self._q_valB()[0][0]:
            return self._q_val()[0]
        else:
            return self._q_valB()[0]
    
    def bellman_error(self, state, action, reward, result_state):
        # return self._bellman_error(state, reward, result_state)
        return self._bellman_error(state, reward)
