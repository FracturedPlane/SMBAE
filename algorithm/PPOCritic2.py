## Nothing for now...

# from modular_rl import *

# ================================================================
# Proximal Policy Optimization
# ================================================================
"""
    This version uses a value function to compute the advantage
    This method also uses some techniques to reduce the KL divergence iff the 
    divergence goes above the a threshold.
"""


import theano
from theano import tensor as T
from lasagne.layers import get_all_params
import numpy as np
import lasagne
import sys
import copy
sys.path.append('../')
from model.ModelUtil import norm_state, scale_state, norm_action, scale_action, action_bound_std
from model.LearningUtil import loglikelihood, kl, entropy, likelihood
from algorithm.AlgorithmInterface import AlgorithmInterface



class PPOCritic2(AlgorithmInterface):
    
    def __init__(self, model, n_in, n_out, state_bounds, action_bounds, reward_bound, settings_):

        super(PPOCritic2,self).__init__(model, n_in, n_out, state_bounds, action_bounds, reward_bound, settings_)
        
        # create a small convolutional neural network
        
        self._Fallen = T.bcol("Fallen")
        ## because float64 <= float32 * int32, need to use int16 or int8
        self._Fallen.tag.test_value = np.zeros((self._batch_size,1),dtype=np.dtype('int8'))
        
        self._fallen_shared = theano.shared(
            np.zeros((self._batch_size, 1), dtype='int8'),
            broadcastable=(False, True))
        
        self._advantage = T.col("Advantage")
        self._advantage.tag.test_value = np.zeros((self._batch_size,1),dtype=np.dtype(self.getSettings()['float_type']))
        
        self._advantage_shared = theano.shared(
            np.zeros((self._batch_size, 1), dtype=self.getSettings()['float_type']),
            broadcastable=(False, True))
        
        self._KL_Weight = T.scalar("KL_Weight")
        self._KL_Weight.tag.test_value = np.zeros((1),dtype=np.dtype(self.getSettings()['float_type']))[0]
        
        self._kl_weight_shared = theano.shared(
            np.ones((1), dtype=self.getSettings()['float_type'])[0])
        self._kl_weight_shared.set_value(self.getSettings()['previous_value_regularization_weight'])
        
        """
        self._target_shared = theano.shared(
            np.zeros((self._batch_size, 1), dtype='float64'),
            broadcastable=(False, True))
        """
        self._critic_regularization_weight = self.getSettings()["critic_regularization_weight"]
        self._critic_learning_rate = self.getSettings()["critic_learning_rate"]
        # primary network
        self._model = model
        # Target network
        self._modelTarget = copy.deepcopy(model)
        
        self._q_valsA = lasagne.layers.get_output(self._model.getCriticNetwork(), self._model.getStateSymbolicVariable(), deterministic=True)
        self._q_valsA_drop = lasagne.layers.get_output(self._model.getCriticNetwork(), self._model.getStateSymbolicVariable(), deterministic=False)
        self._q_valsTargetNextState = lasagne.layers.get_output(self._modelTarget.getCriticNetwork(), self._model.getResultStateSymbolicVariable(), deterministic=True)
        self._q_valsTarget = lasagne.layers.get_output(self._modelTarget.getCriticNetwork(), self._model.getStateSymbolicVariable(), deterministic=True)
        self._q_valsTarget_drop = lasagne.layers.get_output(self._modelTarget.getCriticNetwork(), self._model.getStateSymbolicVariable(), deterministic=False)
        
        self._q_valsActA = lasagne.layers.get_output(self._model.getActorNetwork(), self._model.getStateSymbolicVariable(), deterministic=True)[:,:self._action_length]
        self._q_valsActASTD = lasagne.layers.get_output(self._model.getActorNetwork(), self._model.getStateSymbolicVariable(), deterministic=True)[:,self._action_length:]
        
        ## prevent value from being 0
        self._q_valsActASTD = (self._q_valsActASTD * self.getSettings()['exploration_rate']) + 5e-2
        self._q_valsActTarget = lasagne.layers.get_output(self._modelTarget.getActorNetwork(), self._model.getStateSymbolicVariable())[:,:self._action_length]
        self._q_valsActTargetSTD = lasagne.layers.get_output(self._modelTarget.getActorNetwork(), self._model.getStateSymbolicVariable())[:,self._action_length:]
        self._q_valsActTargetSTD = (self._q_valsActTargetSTD  * self.getSettings()['exploration_rate']) + 5e-2
        self._q_valsActA_drop = lasagne.layers.get_output(self._model.getActorNetwork(), self._model.getStateSymbolicVariable(), deterministic=False)
        
        self._q_func = self._q_valsA
        self._q_funcTarget = self._q_valsTarget
        self._q_func_drop = self._q_valsA_drop
        self._q_funcTarget_drop = self._q_valsTarget_drop
        self._q_funcAct = self._q_valsActA
        self._q_funcAct_drop = self._q_valsActA_drop
        
        # self._target = (self._model.getRewardSymbolicVariable() + (np.array([self._discount_factor] ,dtype=np.dtype(self.getSettings()['float_type']))[0] * self._q_valsTargetNextState )) * self._Fallen
        self._target = T.mul(T.add(self._model.getRewardSymbolicVariable(), T.mul(self._discount_factor, self._q_valsTargetNextState )), self._Fallen)
        self._diff = self._target - self._q_func
        self._diff_drop = self._target - self._q_func_drop 
        # loss = 0.5 * self._diff ** 2 
        loss = T.pow(self._diff, 2)
        self._loss = T.mean(loss)
        self._loss_drop = T.mean(0.5 * self._diff_drop ** 2)
        
        self._params = lasagne.layers.helper.get_all_params(self._model.getCriticNetwork())
        self._actionParams = lasagne.layers.helper.get_all_params(self._model.getActorNetwork())
        self._givens_ = {
            self._model.getStateSymbolicVariable(): self._model.getStates(),
            self._model.getResultStateSymbolicVariable(): self._model.getResultStates(),
            self._model.getRewardSymbolicVariable(): self._model.getRewards(),
            self._Fallen: self._fallen_shared
            # self._model.getActionSymbolicVariable(): self._actions_shared,
        }
        self._actGivens = {
            self._model.getStateSymbolicVariable(): self._model.getStates(),
            self._model.getResultStateSymbolicVariable(): self._model.getResultStates(),
            self._model.getRewardSymbolicVariable(): self._model.getRewards(),
            self._model.getActionSymbolicVariable(): self._model.getActions(),
            self._Fallen: self._fallen_shared,
            # self._advantage: self._advantage_shared,
            self._KL_Weight: self._kl_weight_shared
        }
        
        self._critic_regularization = (self._critic_regularization_weight * lasagne.regularization.regularize_network_params(
        self._model.getCriticNetwork(), lasagne.regularization.l2))
        # self._actor_regularization = ( (self._regularization_weight * lasagne.regularization.regularize_network_params(
        #         self._model.getActorNetwork(), lasagne.regularization.l2)) )
        self._kl_firstfixed = T.mean(kl(self._q_valsActTarget, self._q_valsActTargetSTD, self._q_valsActA, self._q_valsActASTD, self._action_length))
        # self._actor_regularization = (( self.getSettings()['previous_value_regularization_weight']) * self._kl_firstfixed )
        self._actor_regularization = (( self._KL_Weight ) * self._kl_firstfixed ) + (10*(self._kl_firstfixed>self.getSettings()['kl_divergence_threshold'])*
                                                                                     T.square(self._kl_firstfixed-self.getSettings()['kl_divergence_threshold']))
        
        # SGD update
        # self._updates_ = lasagne.updates.rmsprop(self._loss, self._params, self._learning_rate, self._rho,
        #                                    self._rms_epsilon)
        if (self.getSettings()['optimizer'] == 'rmsprop'):
            self._updates_ = lasagne.updates.rmsprop(self._loss # + self._critic_regularization
                                                     , self._params, self._learning_rate, self._rho,
                                           self._rms_epsilon)
        elif (self.getSettings()['optimizer'] == 'momentum'):
            self._updates_ = lasagne.updates.momentum(self._loss # + self._critic_regularization
                                                      , self._params, self._critic_learning_rate , momentum=self._rho)
        elif ( self.getSettings()['optimizer'] == 'adam'):
            self._updates_ = lasagne.updates.adam(self._loss # + self._critic_regularization 
                        , self._params, self._critic_learning_rate , beta1=0.9, beta2=0.999, epsilon=1e-08)
        else:
            print ("Unknown optimization method: ", self.getSettings()['optimizer'])
            sys.exit(-1)
        ## TD update
        """
        if (self.getSettings()['optimizer'] == 'rmsprop'):
            self._updates_ = lasagne.updates.rmsprop(T.mean(self._q_func) + self._critic_regularization, self._params, 
                        self._critic_learning_rate * -T.mean(self._diff), self._rho, self._rms_epsilon)
        elif (self.getSettings()['optimizer'] == 'momentum'):
            self._updates_ = lasagne.updates.momentum(T.mean(self._q_func) + self._critic_regularization, self._params, 
                        self._critic_learning_rate * -T.mean(self._diff), momentum=self._rho)
        elif ( self.getSettings()['optimizer'] == 'adam'):
            self._updates_ = lasagne.updates.adam(T.mean(self._q_func), self._params, 
                        self._critic_learning_rate * -T.mean(self._diff), beta1=0.9, beta2=0.999, epsilon=1e-08)
        else:
            print ("Unknown optimization method: ", self.getSettings()['optimizer'])
            sys.exit(-1)
        """
        ## Need to perform an element wise operation or replicate _diff for this to work properly.
        # self._actDiff = theano.tensor.elemwise.Elemwise(theano.scalar.mul)((self._model.getActionSymbolicVariable() - self._q_valsActA), 
        #                                                                    theano.tensor.tile((self._advantage * (1.0/(1.0-self._discount_factor))), self._action_length)) # Target network does not work well here?
        
        ## advantage = Q(a,s) - V(s) = (r + gamma*V(s')) - V(s) 
        # self._advantage = (((self._model.getRewardSymbolicVariable() + (self._discount_factor * self._q_valsTargetNextState)) * self._Fallen)) - self._q_func
        
        self._Advantage = self._diff * (1.0/(1.0-self._discount_factor)) ## scale back to same as rewards
        self._log_prob = loglikelihood(self._model.getActionSymbolicVariable(), self._q_valsActA, self._q_valsActASTD, self._action_length)
        self._log_prob_target = loglikelihood(self._model.getActionSymbolicVariable(), self._q_valsActTarget, self._q_valsActTargetSTD, self._action_length)
        # self._prob = likelihood(self._model.getActionSymbolicVariable(), self._q_valsActA, self._q_valsActASTD, self._action_length)
        # self._prob_target = likelihood(self._model.getActionSymbolicVariable(), self._q_valsActTarget, self._q_valsActTargetSTD, self._action_length)
        # self._actLoss_ = ( (T.exp(self._log_prob - self._log_prob_target).dot(self._Advantage)) )
        # self._actLoss_ = ( (T.exp(self._log_prob - self._log_prob_target) * (self._Advantage)) )
        # self._actLoss_ = ( ((self._log_prob) * self._Advantage) )
        # self._actLoss_ = ( ((self._log_prob)) )
        ## This does the sum already
        # self._actLoss_ =  ( (self._log_prob).dot( self._Advantage) )
        # self._actLoss_ = theano.tensor.elemwise.Elemwise(theano.scalar.mul)((self._prob / self._prob_target), self._Advantage)
        self._actLoss_ = theano.tensor.elemwise.Elemwise(theano.scalar.mul)(T.exp(self._log_prob - self._log_prob_target), self._Advantage)
        
        # self._actLoss_ = theano.tensor.elemwise.Elemwise(theano.scalar.mul)((self._log_prob), self._Advantage)
        # self._actLoss_ = T.mean(self._log_prob) 
        # self._policy_entropy = 0.5 * T.mean(T.log(2 * np.pi * self._q_valsActASTD ) + 1 )
        ## - because update computes gradient DESCENT updates
        # self._actLoss = -1.0 * ((T.mean(self._actLoss_)) + (self._actor_regularization ))
        # self._entropy = -1. * T.sum(T.log(self._q_valsActA + 1e-8) * self._q_valsActA, axis=1, keepdims=True)
        ## - because update computes gradient DESCENT updates
        self._actLoss = (-1.0 * T.mean(self._actLoss_)) + (1.0 *self._actor_regularization) + (-1e-3 * entropy(self._q_valsActASTD))
        # self._actLoss_drop = (T.sum(0.5 * self._actDiff_drop ** 2)/float(self._batch_size)) # because the number of rows can shrink
        # self._actLoss_drop = (T.mean(0.5 * self._actDiff_drop ** 2))
        self._policy_grad = T.grad(self._actLoss ,  self._actionParams)
        if (self.getSettings()['optimizer'] == 'rmsprop'):
            self._actionUpdates = lasagne.updates.rmsprop(self._policy_grad, self._actionParams, 
                    self._learning_rate , self._rho, self._rms_epsilon)
        elif (self.getSettings()['optimizer'] == 'momentum'):
            self._actionUpdates = lasagne.updates.momentum(self._policy_grad, self._actionParams, 
                    self._learning_rate , momentum=self._rho)
        elif ( self.getSettings()['optimizer'] == 'adam'):
            self._actionUpdates = lasagne.updates.adam(self._policy_grad, self._actionParams, 
                    self._learning_rate , beta1=0.9, beta2=0.999, epsilon=1e-08)
        else:
            print ("Unknown optimization method: ", self.getSettings()['optimizer'])
            
            
        
        # actionUpdates = lasagne.updates.rmsprop(T.mean(self._q_funcAct_drop) + 
        #   (self._regularization_weight * lasagne.regularization.regularize_network_params(
        #       self._model.getActorNetwork(), lasagne.regularization.l2)), actionParams, 
        #           self._learning_rate * 0.5 * (-T.sum(actDiff_drop)/float(self._batch_size)), self._rho, self._rms_epsilon)
        self._givens_grad = {
            self._model.getStateSymbolicVariable(): self._model.getStates(),
            # self._model.getResultStateSymbolicVariable(): self._model.getResultStates(),
            # self._model.getRewardSymbolicVariable(): self._model.getRewards(),
            # self._model.getActionSymbolicVariable(): self._actions_shared,
        }
        
        ## Bellman error
        self._bellman = self._target - self._q_funcTarget
        
        PPOCritic2.compile(self)
        
    def compile(self):
        
        #### Stuff for Debugging #####
        #### Stuff for Debugging #####
        self._get_diff = theano.function([], [self._diff], givens=self._givens_)
        # self._get_advantage = theano.function([], [self._advantage], givens=self._givens_)
        # self._get_advantage = theano.function([], [self._advantage])
        self._get_target = theano.function([], [self._target], givens={
            # self._model.getStateSymbolicVariable(): self._model.getStates(),
            self._model.getResultStateSymbolicVariable(): self._model.getResultStates(),
            self._model.getRewardSymbolicVariable(): self._model.getRewards(),
            self._Fallen: self._fallen_shared
            # self._model.getActionSymbolicVariable(): self._actions_shared,
        })
        self._get_critic_regularization = theano.function([], [self._critic_regularization])
        self._get_critic_loss = theano.function([], [self._loss], givens=self._givens_)
        
        self._get_actor_regularization = theano.function([], [self._actor_regularization],
                                                            givens={self._model.getStateSymbolicVariable(): self._model.getStates(),
                                                                    self._KL_Weight: self._kl_weight_shared})
        self._get_actor_loss = theano.function([], [self._actLoss], givens=self._actGivens)
        # self._get_actor_diff_ = theano.function([], [self._actDiff], givens= self._actGivens)
        """{
            self._model.getStateSymbolicVariable(): self._model.getStates(),
            self._model.getResultStateSymbolicVariable(): self._model.getResultStates(),
            self._model.getRewardSymbolicVariable(): self._model.getRewards(),
            self._model.getActionSymbolicVariable(): self._model.getActions(),
            self._Fallen: self._fallen_shared
        }) """
        
        self._get_action_diff = theano.function([], [self._actLoss_], givens={
            self._model.getStateSymbolicVariable(): self._model.getStates(),
            self._model.getResultStateSymbolicVariable(): self._model.getResultStates(),
            self._model.getRewardSymbolicVariable(): self._model.getRewards(),
            self._model.getActionSymbolicVariable(): self._model.getActions(),
            self._Fallen: self._fallen_shared,
            # self._advantage: self._advantage_shared,
            # self._KL_Weight: self._kl_weight_shared
        })
        
        
        self._train = theano.function([], [self._loss, self._q_func], updates=self._updates_, givens=self._givens_)
        self._trainActor = theano.function([], [self._actLoss, self._q_func_drop], updates=self._actionUpdates, givens=self._actGivens)
        self._q_val = theano.function([], self._q_func,
                                       givens={self._model.getStateSymbolicVariable(): self._model.getStates()})
        self._q_valTarget = theano.function([], self._q_funcTarget,
                                       givens={self._model.getStateSymbolicVariable(): self._modelTarget.getStates()})
        self._q_val_drop = theano.function([], self._q_func_drop,
                                       givens={self._model.getStateSymbolicVariable(): self._model.getStates()})
        self._q_action_drop = theano.function([], self._q_valsActA_drop,
                                       givens={self._model.getStateSymbolicVariable(): self._model.getStates()})
        self._q_action = theano.function([], self._q_valsActA,
                                       givens={self._model.getStateSymbolicVariable(): self._model.getStates()})
        self._q_action_std = theano.function([], self._q_valsActASTD,
                                       givens={self._model.getStateSymbolicVariable(): self._model.getStates()})
        self._get_log_prob = theano.function([], self._log_prob,
                                       givens={self._model.getStateSymbolicVariable(): self._model.getStates(),
                                               self._model.getActionSymbolicVariable(): self._model.getActions(),})
        self._get_log_prob_target = theano.function([], self._log_prob_target,
                                       givens={self._model.getStateSymbolicVariable(): self._model.getStates(),
                                               self._model.getActionSymbolicVariable(): self._model.getActions(),})
        
        self._q_action_target = theano.function([], self._q_valsActTarget,
                                       givens={self._model.getStateSymbolicVariable(): self._model.getStates()})
        # self._bellman_error_drop = theano.function(inputs=[self._model.getStateSymbolicVariable(), self._model.getRewardSymbolicVariable(), self._model.getResultStateSymbolicVariable()], outputs=self._diff_drop, allow_input_downcast=True)
        self._bellman_error_drop2 = theano.function(inputs=[], outputs=self._diff_drop, allow_input_downcast=True, givens=self._givens_)
        
        # self._bellman_error = theano.function(inputs=[self._model.getStateSymbolicVariable(), self._model.getResultStateSymbolicVariable(), self._model.getRewardSymbolicVariable()], outputs=self._diff, allow_input_downcast=True)
        self._bellman_error2 = theano.function(inputs=[], outputs=self._diff, allow_input_downcast=True, givens=self._givens_)
        self._bellman_errorTarget = theano.function(inputs=[], outputs=self._bellman, allow_input_downcast=True, givens=self._givens_)
        # self._diffs = theano.function(input=[self._model.getStateSymbolicVariable()])
        self._get_grad = theano.function([], outputs=lasagne.updates.get_or_compute_grads(T.mean(self._q_func), [lasagne.layers.get_all_layers(self._model.getCriticNetwork())[0].input_var] + self._params), allow_input_downcast=True, givens=self._givens_grad)
        # self._get_grad2 = theano.gof.graph.inputs(lasagne.updates.rmsprop(loss, params, self._learning_rate, self._rho, self._rms_epsilon))
        
        # self._compute_fisher_vector_product = theano.function([flat_tangent] + args, fvp, **FNOPTS)
        self.kl_divergence = theano.function([], self._kl_firstfixed,
                                             givens={self._model.getStateSymbolicVariable(): self._model.getStates()})
        
    def updateTargetModel(self):
        print ("Updating target Model")
        """
            Target model updates
        """
        all_paramsA = lasagne.layers.helper.get_all_param_values(self._model.getCriticNetwork())
        all_paramsActA = lasagne.layers.helper.get_all_param_values(self._model.getActorNetwork())
        lasagne.layers.helper.set_all_param_values(self._modelTarget.getCriticNetwork(), all_paramsA)
        lasagne.layers.helper.set_all_param_values(self._modelTarget.getActorNetwork(), all_paramsActA) 
    
    def setData(self, states, actions, rewards, result_states, fallen):
        self._model.setStates(states)
        self._model.setResultStates(result_states)
        self._model.setActions(actions)
        self._model.setRewards(rewards)
        self._modelTarget.setStates(states)
        self._modelTarget.setResultStates(result_states)
        self._modelTarget.setActions(actions)
        self._modelTarget.setRewards(rewards)
        # print ("Falls: ", fallen)
        self._fallen_shared.set_value(fallen)
        # diff_ = self.bellman_error(states, actions, rewards, result_states, falls)
        ## Easy fix for computing actor loss
        diff = self._bellman_error2()
        self._advantage_shared.set_value(diff)
        
        # _targets = rewards + (self._discount_factor * self._q_valsTargetNextState )
        
    def trainCritic(self, states, actions, rewards, result_states, falls):
        self.setData(states, actions, rewards, result_states, falls)
        # print ("Performing Critic trainning update")
        if (( self._updates % self._weight_update_steps) == 0):
            self.updateTargetModel()
        self._updates += 1
        # print ("Falls:", falls)
        # print ("Ceilinged Rewards: ", np.ceil(rewards))
        # print ("Target Values: ", self._get_target())
        # print ("V Values: ", np.mean(self._q_val()))
        # print ("diff Values: ", np.mean(self._get_diff()))
        # data = np.append(falls, self._get_target()[0], axis=1)
        # print ("Rewards, Falls, Targets:", np.append(rewards, data, axis=1))
        # print ("Rewards, Falls, Targets:", [rewards, falls, self._get_target()])
        # print ("Actions: ", actions)
        loss, _ = self._train()
        print(" Critic loss: ", loss)
        
        return loss
    
    def trainActor(self, states, actions, rewards, result_states, falls, advantage):
        
        self.setData(states, actions, rewards, result_states, falls)
        # advantage = self._get_diff()[0]
        # self._advantage_shared.set_value(advantage)
        ## Update the network parameters of the target network
        all_paramsActA = lasagne.layers.helper.get_all_param_values(self._model.getActorNetwork())
        lasagne.layers.helper.set_all_param_values(self._modelTarget.getActorNetwork(), all_paramsActA)
        # print ("Performing Critic trainning update")
        # if (( self._updates % self._weight_update_steps) == 0):
        #     self.updateTargetModel()
        # self._updates += 1
        # loss, _ = self._train()
        # print( "Actor loss: ", self._get_action_diff())
        lossActor = 0
        
        # diff_ = self.bellman_error(states, actions, rewards, result_states, falls)
        # print("Advantage: ", np.mean(self._get_advantage()))
        print("Advantage: ", np.mean(advantage), " std: ", np.std(advantage))
        print("Actions:     ", np.mean(actions, axis=0))
        print("Policy mean: ", np.mean(self._q_action(), axis=0))
        # print("Actions std:  ", np.mean(np.sqrt( (np.square(np.abs(actions - np.mean(actions, axis=0))))/1.0), axis=0) )
        print("Actions std:  ", np.std((actions - self._q_action()), axis=0) )
        print("Policy   std: ", np.mean(self._q_action_std(), axis=0))
        print("Policy log prob target: ", np.mean(self._get_log_prob_target(), axis=0))
        print( "Actor loss: ", np.mean(self._get_action_diff()))
        # print ("Actor diff: ", np.mean(np.array(self._get_diff()) / (1.0/(1.0-self._discount_factor))))
        ## Sometimes really HUGE losses appear, ocasionally
        if (np.abs(np.mean(self._get_action_diff())) < 10): 
            lossActor, _ = self._trainActor()
        print("Policy log prob after: ", np.mean(self._get_log_prob(), axis=0))
        # print( "Length of positive actions: " , str(len(tmp_actions)), " Actor loss: ", lossActor)
        print( " Actor loss: ", lossActor)
        # print( "Policy loss: ", lossActor)
        # self._advantage_shared.set_value(diff_)
        # lossActor, _ = self._trainActor()
        kl_after = self.kl_divergence()
        ### kl is too high
        if (kl_after > self.getSettings()['kl_divergence_threshold']):
        # if ( True ):
            ### perform line search to find better step for this update
            iters = 10
            ## Best is closest to threshold without going over
            best = [-10000000, 0]
            current_params = np.array(lasagne.layers.helper.get_all_param_values(self._model.getActorNetwork()))
            old_params = np.array(lasagne.layers.helper.get_all_param_values(self._modelTarget.getActorNetwork()))
            param_direction = current_params - old_params
            alpha_ = 0.5
            for alpha in range(iters):
                # alpha_ = float(alpha)/iters
                tmp_params = (old_params) + (param_direction * alpha_)
                tmp_params_list = [i for i in tmp_params]
                lasagne.layers.helper.set_all_param_values(self._model.getActorNetwork(), tmp_params_list)
                act_loss = np.mean(self._get_action_diff())
                print ("New kl: ", self.kl_divergence(), " act_loss: ", act_loss, " alpha: ", alpha_)
                if ( (self.kl_divergence() < self.getSettings()['kl_divergence_threshold']) and 
                    # ( self.kl_divergence() > best[0] ) ):
                    (act_loss > best[0]) ):
                    best[0] = self.kl_divergence()
                    best[1] = alpha_
                if ( (self.kl_divergence() < self.getSettings()['kl_divergence_threshold']) ):
                    alpha_ = (alpha_ * 1.25)
                else:
                    alpha_ = (alpha_ / 1.35)
                if ( alpha_ > 1.0 ):
                    ## Want to try and keep alpha between 0 and 1
                    alpha_ = (alpha_ / 1.2)
            alpha_ = best[1]
            tmp_params = (old_params) + (param_direction * alpha_)
            tmp_params_list = [i for i in tmp_params]
            lasagne.layers.helper.set_all_param_values(self._model.getActorNetwork(), tmp_params_list)
        """
        if kl_d > self.getSettings()['kl_divergence_threshold']:
            self._kl_weight_shared.set_value(self._kl_weight_shared.get_value()*2.0)
        else:
            self._kl_weight_shared.set_value(self._kl_weight_shared.get_value()/2.0)
        """  
    
        kl_coeff = self._kl_weight_shared.get_value()
        if kl_after > 1.3*self.getSettings()['kl_divergence_threshold']: 
            kl_coeff *= 1.5
            if (kl_coeff < 1e-4):
                self._kl_weight_shared.set_value(kl_coeff)
                print "Got KL=%.3f (target %.3f). Increasing penalty coeff => %.3f."%(kl_after, self.getSettings()['kl_divergence_threshold'], kl_coeff)
        elif kl_after < 0.7*self.getSettings()['kl_divergence_threshold']: 
            kl_coeff /= 1.5
            if ( kl_coeff > 1e-8 ):
                self._kl_weight_shared.set_value(kl_coeff)
                print "Got KL=%.3f (target %.3f). Decreasing penalty coeff => %.3f."%(kl_after, self.getSettings()['kl_divergence_threshold'], kl_coeff)
        else:
            print ("KL=%.3f is close enough to target %.3f."%(kl_after, self.getSettings()['kl_divergence_threshold']))
        print ("KL_divergence: ", self.kl_divergence(), " kl_weight: ", self._kl_weight_shared.get_value())
        
        # print ("Diff")
        # print (diff_)
        """
        tmp_states=[]
        tmp_result_states=[]
        tmp_actions=[]
        tmp_rewards=[]
        tmp_falls=[]
        tmp_diff=[]
        for i in range(len(diff_)):
            if ( diff_[i] > 0.0):
                tmp_diff.append(diff_[i])
                tmp_states.append(states[i])
                tmp_result_states.append(result_states[i])
                tmp_actions.append(actions[i])
                tmp_rewards.append(rewards[i])
                tmp_falls.append(falls[i])
                
        if (len(tmp_actions) > 0):
            self._advantage_shared.set_value(tmp_diff)
            self.setData(tmp_states, tmp_actions, tmp_rewards, tmp_result_states, tmp_falls)
        
            # print ("Actor diff: ", np.mean(np.array(self._get_diff()) / (1.0/(1.0-self._discount_factor))))
            lossActor, _ = self._trainActor()
            print( "Length of positive actions: " , str(len(tmp_actions)), " Actor loss: ", lossActor)
            # print( " Actor loss: ", lossActor)
            # print("Diff for actor: ", self._get_diff())
            # print ("Tmp_diff: ", tmp_diff)
            # print ( "Action before diff: ", self._get_actor_diff_())
            # print( "Action diff: ", self._get_action_diff())
            # return np.sqrt(lossActor);
        """
        return lossActor
    
    def train(self, states, actions, rewards, result_states, falls):
        loss = self.trainCritic(states, actions, rewards, result_states, falls)
        lossActor = self.trainActor(states, actions, rewards, result_states, falls)
        return loss


"""
class TrpoUpdater(EzFlat, EzPickle):
    
    options = [
        ("cg_damping", float, 1e-3, "Add multiple of the identity to Fisher matrix during CG"),
        ("max_kl", float, 1e-2, "KL divergence between old and new policy (averaged over state-space)"),
    ]

    def __init__(self, stochpol, usercfg):
        EzPickle.__init__(self, stochpol, usercfg)
        cfg = update_default_config(self.options, usercfg)

        self.stochpol = stochpol
        self.cfg = cfg

        probtype = stochpol.probtype
        params = stochpol.trainable_variables
        EzFlat.__init__(self, params)

        ob_no = stochpol.input
        act_na = probtype.sampled_variable()
        adv_n = T.vector("adv_n")

        # Probability distribution:
        prob_np = stochpol.get_output()
        oldprob_np = probtype.prob_variable()

        logp_n = probtype.loglikelihood(act_na, prob_np)
        oldlogp_n = probtype.loglikelihood(act_na, oldprob_np)
        N = ob_no.shape[0]

        # Policy gradient:
        surr = (-1.0 / N) * T.exp(logp_n - oldlogp_n).dot(adv_n)
        pg = flatgrad(surr, params)

        prob_np_fixed = theano.gradient.disconnected_grad(prob_np)
        kl_firstfixed = probtype.kl(prob_np_fixed, prob_np).sum()/N
        ## first derivative of kl
        grads = T.grad(kl_firstfixed, params)
        flat_tangent = T.fvector(name="flat_tan")
        shapes = [var.get_value(borrow=True).shape for var in params]
        start = 0
        ## Collect the current tangents
        tangents = []
        for shape in shapes:
            size = np.prod(shape)
            tangents.append(T.reshape(flat_tangent[start:start+size], shape))
            start += size
        ## fisher vector product  (gvp: gradient vector product, fvp: fisher vector product)
        ## grad * tangent = jacobian    
        gvp = T.add(*[T.sum(g*tangent) for (g, tangent) in zipsame(grads, tangents)]) #pylint: disable=E1111
        # Fisher-vector product
        ## I think this computes the jacobian over the dot product between two gradients, resulting in the Hessian
        fvp = flatgrad(gvp, params)

        ent = probtype.entropy(prob_np).mean()
        kl = probtype.kl(oldprob_np, prob_np).mean()

        losses = [surr, kl, ent]
        self.loss_names = ["surr", "kl", "ent"]

        args = [ob_no, act_na, adv_n, oldprob_np]

        self.compute_policy_gradient = theano.function(args, pg, **FNOPTS)
        self.compute_losses = theano.function(args, losses, **FNOPTS)
        self.compute_fisher_vector_product = theano.function([flat_tangent] + args, fvp, **FNOPTS)

    def __call__(self, paths):
        cfg = self.cfg
        prob_np = concat([path["prob"] for path in paths])
        ob_no = concat([path["observation"] for path in paths])
        action_na = concat([path["action"] for path in paths])
        advantage_n = concat([path["advantage"] for path in paths])
        args = (ob_no, action_na, advantage_n, prob_np)

        thprev = self.get_params_flat()
        def fisher_vector_product(p):
            return self.compute_fisher_vector_product(p, *args)+cfg["cg_damping"]*p #pylint: disable=E1101,W0640
        g = self.compute_policy_gradient(*args)
        losses_before = self.compute_losses(*args)
        if np.allclose(g, 0):
            print "got zero gradient. not updating"
        else:
            stepdir = cg(fisher_vector_product, -g)
            ## preconditioner, I think
            shs = .5*stepdir.dot(fisher_vector_product(stepdir))
            lm = np.sqrt(shs / cfg["max_kl"])
            print "lagrange multiplier:", lm, "gnorm:", np.linalg.norm(g)
            fullstep = stepdir / lm
            neggdotstepdir = -g.dot(stepdir)
            def loss(th):
                self.set_params_flat(th)
                ## Returns surrogate loss
                return self.compute_losses(*args)[0] #pylint: disable=W0640
            ## line searcho ver surrogate loss
            success, theta = linesearch(loss, thprev, fullstep, neggdotstepdir/lm)
            print "success", success
            self.set_params_flat(theta)
        losses_after = self.compute_losses(*args)

        out = OrderedDict()
        for (lname, lbefore, lafter) in zipsame(self.loss_names, losses_before, losses_after):
            out[lname+"_before"] = lbefore
            out[lname+"_after"] = lafter
        return out

def linesearch(f, x, fullstep, expected_improve_rate, max_backtracks=10, accept_ratio=.1):
    
    fval = f(x)
    print "fval before", fval
    for (_n_backtracks, stepfrac) in enumerate(.5**np.arange(max_backtracks)):
        xnew = x + stepfrac*fullstep
        newfval = f(xnew)
        actual_improve = fval - newfval
        expected_improve = expected_improve_rate*stepfrac
        ratio = actual_improve/expected_improve
        print "a/e/r", actual_improve, expected_improve, ratio
        if ratio > accept_ratio and actual_improve > 0:
            print "fval after", newfval
            return True, xnew
    return False, x

def cg(f_Ax, b, cg_iters=10, callback=None, verbose=False, residual_tol=1e-10):
    
    ## Copies of policy gradient
    p = b.copy()
    r = b.copy()
    x = np.zeros_like(b)
    rdotr = r.dot(r)

    fmtstr =  "%10i %10.3g %10.3g"
    titlestr =  "%10s %10s %10s"
    if verbose: print titlestr % ("iter", "residual norm", "soln norm")

    for i in xrange(cg_iters):
        if callback is not None:
            callback(x)
        if verbose: print fmtstr % (i, rdotr, np.linalg.norm(x))
        ## fisher vector product of policy gradient
        z = f_Ax(p)
        ## 
        v = rdotr / p.dot(z)
        x += v*p
        r -= v*z
        newrdotr = r.dot(r)
        mu = newrdotr/rdotr
        p = r + mu*p

        rdotr = newrdotr
        if rdotr < residual_tol:
            break

    if callback is not None:
        callback(x)
    if verbose: print fmtstr % (i+1, rdotr, np.linalg.norm(x))  # pylint: disable=W0631
    return x
"""