import theano
from theano import tensor as T
import numpy as np
import lasagne
import sys
import copy
sys.path.append('../')
from model.ModelUtil import *
from algorithm.CACLA import CACLA

# For debugging
# theano.config.mode='FAST_COMPILE'

class CACLADV(CACLA):
    
    def __init__(self, model, n_in, n_out, state_bounds, action_bounds, reward_bound, settings_):

        super(CACLADV,self).__init__(model, n_in, n_out, state_bounds, action_bounds, reward_bound, settings_)
        
        # create a small convolutional neural network
        
        self._q_valsNextState = lasagne.layers.get_output(self._model.getCriticNetwork(), self._model.getResultStateSymbolicVariable())
        

        targetTarget = (self._model.getRewardSymbolicVariable() + (self._discount_factor * self._q_valsNextState))
        self._diffTarget = targetTarget - self._q_funcTarget
        
        lossTarget = 0.5 * self._diffTarget ** 2
        self._lossTarget = T.mean(lossTarget)
        
        self._paramsTarget = lasagne.layers.helper.get_all_params(self._modelTarget.getCriticNetwork())
        self._actionParamsTarget = lasagne.layers.helper.get_all_params(self._modelTarget.getActorNetwork())
        self._givens_Target = {
            self._model.getStateSymbolicVariable(): self._modelTarget.getStates(),
            self._model.getResultStateSymbolicVariable(): self._modelTarget.getResultStates(),
            self._model.getRewardSymbolicVariable(): self._modelTarget.getRewards()
            # self._model.getActionSymbolicVariable(): self._actions_shared,
        }
        self._actGivensTarget = {
            self._model.getStateSymbolicVariable(): self._modelTarget.getStates(),
            # self._model.getResultStateSymbolicVariable(): self._next_states_shared,
            # self._model.getRewardSymbolicVariable(): self._rewards_shared,
            self._model.getActionSymbolicVariable(): self._modelTarget.getActions()
        }
        
        # SGD update
        self._updates_Target = lasagne.updates.rmsprop(self._lossTarget + (self._regularization_weight * lasagne.regularization.regularize_network_params(
                    self._modelTarget.getCriticNetwork(), lasagne.regularization.l2)), self._paramsTarget, self._learning_rate, self._rho,
                                          self._rms_epsilon)
        
        # TD update
        # self._updates_Target = lasagne.updates.rmsprop(T.mean(self._q_funcTarget) + (self._regularization_weight * lasagne.regularization.regularize_network_params(
        # self._modelTarget.getCriticNetwork(), lasagne.regularization.l2)), paramsTarget, 
        #            self._learning_rate * -T.mean(diffTarget), self._rho, self._rms_epsilon)
        
        
        actDiff_Target = ((self._model.getActionSymbolicVariable() - self._q_valsActTarget)) # Target network does not work well here?
        actLoss_Target = 0.5 * actDiff_Target ** 2
        self._actLoss_Target = T.sum(actLoss_Target)/float(self._batch_size)
        
        self._actionUpdatesTarget = lasagne.updates.rmsprop(self._actLoss_Target + 
          (self._regularization_weight * lasagne.regularization.regularize_network_params(
              self._modelTarget.getActorNetwork(), lasagne.regularization.l2)), self._actionParamsTarget, 
                  self._learning_rate, self._rho, self._rms_epsilon)
        
        CACLADV.compile(self)
        
        
    def compile(self):
        super(CACLADV, self).compile()  
        #### Functions ####
        self._trainTarget = theano.function([], [self._lossTarget, self._q_funcTarget], updates=self._updates_Target, givens=self._givens_Target)
        self._trainActorTarget = theano.function([], [self._q_valsTarget], updates=self._actionUpdatesTarget, givens=self._actGivensTarget)
        self._q_actionTarget = theano.function([], self._q_valsActTarget, givens={self._model.getStateSymbolicVariable(): self._modelTarget.getStates()})
        self._bellman_errorTarget = theano.function(inputs=[], outputs=self._diffTarget, allow_input_downcast=True, givens=self._givens_Target)
        # self._diffs = theano.function(input=[self._model.getStateSymbolicVariable()])
        
        # x = T.matrices('x')
        # z_lazy = ifelse(T.gt(self._q_val()[0][0], self._q_valTarget()[0][0]), self._q_action(), self._q_actionTarget())
        # self._f_lazyifelse = theano.function([], z_lazy,
        #                        mode=theano.Mode(linker='vm'))
        
    def updateTargetModel(self):
        """
            The target model should not be updated in the way for douvle value learning.
        """
        pass
        

    def trainCritic(self, states, actions, rewards, result_states):
        self.setData(states, actions, rewards, result_states)
        # print ("Performing Critic trainning update")
        
        if (( self._updates % self._weight_update_steps) == 0):
            self.updateTargetModel()
        
        self._updates += 1
        loss = 0
        
        
        import random
        r = random.choice([0,1])
        if r == 0:
            loss, _ = self._train()
            
            # diff_ = self.bellman_error(states, actions, rewards, result_states)
        else:
            loss, _ = self._trainTarget()
            
            # diff_ = self.bellman_errorTarget(states, actions, rewards, result_states)
        return loss
    
    def trainActor(self, states, actions, rewards, result_states):
        self.setData(states, actions, rewards, result_states)
        # print ("Performing Critic trainning update")
        """
        if (( self._updates % self._weight_update_steps) == 0):
            self.updateTargetModel()
        """
        self._updates += 1
        lossActor = 0
        diff_=None
        
        import random
        r = random.choice([0,1])
        if r == 0:
            loss, _ = self._train()
            diff_ = self.bellman_error(states, actions, rewards, result_states)
        else:
            loss, _ = self._trainTarget()
            diff_ = self.bellman_errorTarget(states, actions, rewards, result_states)
        
        # print ("Diff")
        # print (diff_)
        tmp_states=[]
        tmp_result_states=[]
        tmp_actions=[]
        tmp_rewards=[]
        for i in range(len(diff_)):
            # selecting the tuples that are an improvement over the current V()
            if ( diff_[i] > 0.0):
                tmp_states.append(states[i])
                tmp_result_states.append(result_states[i])
                tmp_actions.append(actions[i])
                tmp_rewards.append(rewards[i])
                
        if (len(tmp_actions) > 0):
            self.setData(tmp_states, tmp_actions, tmp_rewards, tmp_result_states)
            if r == 0:
                lossActor = self._trainActor()
            else:
                lossActor = self._trainActorTarget()
        return lossActor
