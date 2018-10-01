## Nothing for now...

# from modular_rl import *

# ================================================================
# Trust Region Policy Optimization
# ================================================================


import theano
from theano import tensor as T
from lasagne.layers import get_all_params
from collections import OrderedDict
import numpy as np
import lasagne
import sys
import copy
sys.path.append('../')
from model.ModelUtil import norm_state, scale_state, norm_action, scale_action, action_bound_std
from model.LearningUtil import loglikelihood, kl, entropy, flatgrad, zipsame, get_params_flat, setFromFlat
from algorithm.AlgorithmInterface import AlgorithmInterface


# For debugging
# theano.config.mode='FAST_COMPILE'
# from DeepCACLA import DeepCACLA

class TRPOCritic(AlgorithmInterface):
    
    def __init__(self, model, n_in, n_out, state_bounds, action_bounds, reward_bound, settings_):

        super(TRPOCritic,self).__init__(model, n_in, n_out, state_bounds, action_bounds, reward_bound, settings_)
        
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
        self._q_valsActASTD = (self._q_valsActASTD * self.getSettings()['exploration_rate']) + 1e-3
        self._q_valsActTarget_ = lasagne.layers.get_output(self._modelTarget.getActorNetwork(), self._model.getStateSymbolicVariable())
        self._q_valsActTarget = self._q_valsActTarget_[:,:self._action_length]
        self._q_valsActTargetSTD = self._q_valsActTarget_[:,self._action_length:]
        self._q_valsActTargetSTD = (self._q_valsActTargetSTD  * self.getSettings()['exploration_rate']) + 1e-3
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
            # self._KL_Weight: self._kl_weight_shared
        }
        
        self._critic_regularization = (self._critic_regularization_weight * lasagne.regularization.regularize_network_params(
        self._model.getCriticNetwork(), lasagne.regularization.l2))
        # self._actor_regularization = ( (self._regularization_weight * lasagne.regularization.regularize_network_params(
        #         self._model.getActorNetwork(), lasagne.regularization.l2)) )
        self._kl_firstfixed = kl(self._q_valsActTarget, self._q_valsActTargetSTD, self._q_valsActA, self._q_valsActASTD, self._action_length).mean()
        # self._actor_regularization = (( self.getSettings()['previous_value_regularization_weight']) * self._kl_firstfixed )
        self._actor_regularization = (( self._KL_Weight ) * self._kl_firstfixed ) + ((self._kl_firstfixed>self.getSettings()['kl_divergence_threshold'])*
                                                                                     T.square(self._kl_firstfixed-self.getSettings()['kl_divergence_threshold']))
        
        # SGD update
        # self._updates_ = lasagne.updates.rmsprop(self._loss + (self._regularization_weight * lasagne.regularization.regularize_network_params(
        # self._model.getCriticNetwork(), lasagne.regularization.l2)), self._params, self._learning_rate, self._rho,
        #                                    self._rms_epsilon)
        # TD update
        if (self.getSettings()['optimizer'] == 'rmsprop'):
            self._updates_ = lasagne.updates.rmsprop(T.mean(self._q_func) + self._critic_regularization, self._params, 
                        self._critic_learning_rate * -T.mean(self._diff), self._rho, self._rms_epsilon)
        elif (self.getSettings()['optimizer'] == 'momentum'):
            self._updates_ = lasagne.updates.momentum(T.mean(self._q_func) + self._critic_regularization, self._params, 
                        self._critic_learning_rate * -T.mean(self._diff), momentum=self._rho)
        elif ( self.getSettings()['optimizer'] == 'adam'):
            self._updates_ = lasagne.updates.adam(T.mean(self._q_func) + self._critic_regularization, self._params, 
                        self._critic_learning_rate * -T.mean(self._diff), beta1=0.9, beta2=0.999, epsilon=1e-08)
        else:
            print ("Unknown optimization method: ", self.getSettings()['optimizer'])
            sys.exit(-1)
        ## Need to perform an element wise operation or replicate _diff for this to work properly.
        # self._actDiff = theano.tensor.elemwise.Elemwise(theano.scalar.mul)((self._model.getActionSymbolicVariable() - self._q_valsActA), 
        #                                                                    theano.tensor.tile((self._advantage * (1.0/(1.0-self._discount_factor))), self._action_length)) # Target network does not work well here?
        
        ## advantage = Q(a,s) - V(s) = (r + gamma*V(s')) - V(s) 
        # self._advantage = (((self._model.getRewardSymbolicVariable() + (self._discount_factor * self._q_valsTargetNextState)) * self._Fallen)) - self._q_func
        
        self._Advantage = self._diff # * (1.0/(1.0-self._discount_factor)) ## scale back to same as rewards
        self._log_prob = loglikelihood(self._model.getActionSymbolicVariable(), self._q_valsActA, self._q_valsActASTD, self._action_length)
        self._log_prob_target = loglikelihood(self._model.getActionSymbolicVariable(), self._q_valsActTarget, self._q_valsActTargetSTD, self._action_length)
        # self._actLoss_ = ( (T.exp(self._log_prob - self._log_prob_target).dot(self._Advantage)) )
        # self._actLoss_ = ( (T.exp(self._log_prob - self._log_prob_target) * (self._Advantage)) )
        # self._actLoss_ = ( ((self._log_prob) * self._Advantage) )
        # self._actLoss_ = ( ((self._log_prob)) )
        ## This does the sum already
        # self._actLoss_ =  ( (self._log_prob).dot( self._Advantage) )
        self._actLoss_ = theano.tensor.elemwise.Elemwise(theano.scalar.mul)(T.exp(self._log_prob - self._log_prob_target), self._Advantage)
        # self._actLoss_ = theano.tensor.elemwise.Elemwise(theano.scalar.mul)(T.exp(self._log_prob - self._log_prob_target), self._advantage)
        
        # self._actLoss_ = theano.tensor.elemwise.Elemwise(theano.scalar.mul)((self._log_prob), self._Advantage)
        # self._actLoss_ = T.mean(self._log_prob) 
        # self._policy_entropy = 0.5 * T.mean(T.log(2 * np.pi * self._q_valsActASTD ) + 1 )
        ## - because update computes gradient DESCENT updates
        # self._actLoss = -1.0 * ((T.mean(self._actLoss_)) + (self._actor_regularization ))
        # self._entropy = -1. * T.sum(T.log(self._q_valsActA + 1e-8) * self._q_valsActA, axis=1, keepdims=True)
        ## - because update computes gradient DESCENT updates
        self._actLoss = (T.mean(self._actLoss_)) 
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
        N = self._model.getStateSymbolicVariable().shape[0]
        params = self._actionParams
        surr = self._actLoss * (-1.0)
        self.pg = flatgrad(surr, params)

        prob_mean_fixed = theano.gradient.disconnected_grad(self._q_valsActA)
        prob_std_fixed = theano.gradient.disconnected_grad(self._q_valsActASTD)
        kl_firstfixed = kl(prob_mean_fixed, prob_std_fixed, self._q_valsActA, self._q_valsActASTD, self._action_length).sum()/N
        grads = T.grad(kl_firstfixed, params)
        self.flat_tangent = T.vector(name="flat_tan")
        shapes = [var.get_value(borrow=True).shape for var in params]
        start = 0
        tangents = []
        for shape in shapes:
            size = np.prod(shape)
            tangents.append(T.reshape(self.flat_tangent[start:start+size], shape))
            start += size
        self.gvp = T.add(*[T.sum(g*tangent) for (g, tangent) in zipsame(grads, tangents)]) #pylint: disable=E1111
        # Fisher-vector product
        self.fvp = flatgrad(self.gvp, params)
        
        self.ent = entropy(self._q_valsActASTD).mean()
        self.kl = kl(self._q_valsActTarget, self._q_valsActTargetSTD, self._q_valsActA, self._q_valsActASTD, self._action_length).mean()
        
        self.losses = [surr, self.kl, self.ent]
        self.loss_names = ["surr", "kl", "ent"]

        self.args = [self._model.getStateSymbolicVariable(), 
                     self._model.getActionSymbolicVariable(),
                     self._model.getResultStateSymbolicVariable(), 
                     self._model.getRewardSymbolicVariable(),
                     self._Fallen
                     # self._advantage
                     # self._q_valsActTarget_
                     ]
        
        
        self.args_fvp = [self._model.getStateSymbolicVariable(), 
                     # self._model.getActionSymbolicVariable()
                     # self._advantage,
                     # self._q_valsActTarget_
                     ]
        
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
        
        TRPOCritic.compile(self)
        
    def compile(self):
        
        #### Stuff for Debugging #####
        #### Stuff for Debugging #####
        self._get_diff = theano.function([], [self._diff], givens=self._givens_)
        self._get_advantage = theano.function([], [self._Advantage], givens=self._givens_)
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
            #self._advantage: self._advantage_shared,
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
        
        self.compute_policy_gradient = theano.function(self.args, self.pg)
        self.compute_losses = theano.function(self.args, self.losses)
        self.compute_fisher_vector_product = theano.function([self.flat_tangent] + self.args_fvp, self.fvp)
        
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
        print("Advantage: ", np.mean(self._get_advantage()))
        # print("Advantage: ", np.mean(advantage))
        # print("Advantage, reward: ", np.concatenate((self._get_advantage(), rewards), axis=1))
        print("Actions:     ", np.mean(actions, axis=0))
        print("Policy mean: ", np.mean(self._q_action(), axis=0))
        # print("Actions std:  ", np.mean(np.sqrt( (np.square(np.abs(actions - np.mean(actions, axis=0))))/1.0), axis=0) )
        print("Actions std:  ", np.std((actions - self._q_action()), axis=0) )
        print("Policy   std: ", np.mean(self._q_action_std(), axis=0))
        print("Policy log prob target: ", np.mean(self._get_log_prob_target(), axis=0))
        # print( "Actor loss: ", np.mean(self._get_action_diff()))
        # print ("Actor diff: ", np.mean(np.array(self._get_diff()) / (1.0/(1.0-self._discount_factor))))
        ## Sometimes really HUGE losses appear, ocasionally
        # if (np.abs(np.mean(self._get_action_diff())) < 10): 
        #     lossActor, _ = self._trainActor()
        
        self.getSettings()['cg_damping'] = 1e-3
        """
        cfg = self.cfg
        prob_np = concat([path["prob"] for path in paths])
        ob_no = concat([path["observation"] for path in paths])
        action_na = concat([path["action"] for path in paths])
        advantage_n = concat([path["advantage"] for path in paths])
        """
        args = (states, actions, result_states, rewards, falls)
        args_fvp = (states)

        thprev = get_params_flat(lasagne.layers.helper.get_all_param_values(self._model.getActorNetwork()))
        def fisher_vector_product(p):
            return self.compute_fisher_vector_product(p, states)+np.float32(self.getSettings()['cg_damping'])*p #pylint: disable=E1101,W0640
        g = self.compute_policy_gradient(*args)
        losses_before = self.compute_losses(*args)
        if np.allclose(g, 0):
            print "got zero gradient. not updating"
        else:
            stepdir = cg(fisher_vector_product, -g)
            shs = .5*stepdir.dot(fisher_vector_product(stepdir))
            lm = np.sqrt(shs / np.float32(self.getSettings()['kl_divergence_threshold']))
            print "lagrange multiplier:", lm, "gnorm:", np.linalg.norm(g)
            fullstep = stepdir / lm
            neggdotstepdir = -g.dot(stepdir)
            def loss(th):
                # self.set_params_flat(th)
                params_tmp = setFromFlat(all_paramsActA, th)
                lasagne.layers.helper.set_all_param_values(self._model.getActorNetwork(), params_tmp)
                return self.compute_losses(*args)[0] #pylint: disable=W0640
            success, theta = linesearch(loss, thprev, fullstep, neggdotstepdir/lm)
            print "success", success
            params_tmp = setFromFlat(all_paramsActA, theta)
            lasagne.layers.helper.set_all_param_values(self._model.getActorNetwork(), params_tmp)
            # self.set_params_flat(theta)
        losses_after = self.compute_losses(*args)

        out = OrderedDict()
        for (lname, lbefore, lafter) in zipsame(self.loss_names, losses_before, losses_after):
            out[lname+"_before"] = lbefore
            out[lname+"_after"] = lafter
        return out
    
        print( "Losses before: ", self.loss_names, ", ", losses_before)
        print( "Losses after: ", self.loss_names, ", ", losses_after)
        
        # print("Policy log prob after: ", np.mean(self._get_log_prob(), axis=0))
        # print( "Length of positive actions: " , str(len(tmp_actions)), " Actor loss: ", lossActor)
        # print( " Actor loss: ", lossActor)
        # self._advantage_shared.set_value(diff_)
        # lossActor, _ = self._trainActor()
        # kl_after = self.kl_divergence()
        """
        if kl_d > self.getSettings()['kl_divergence_threshold']:
            self._kl_weight_shared.set_value(self._kl_weight_shared.get_value()*2.0)
        else:
            self._kl_weight_shared.set_value(self._kl_weight_shared.get_value()/2.0)
        """  
    
        # return lossActor
    
    def train(self, states, actions, rewards, result_states, falls):
        loss = self.trainCritic(states, actions, rewards, result_states, falls)
        lossActor = self.trainActor(states, actions, rewards, result_states, falls)
        return loss
    
def linesearch(f, x, fullstep, expected_improve_rate, max_backtracks=10, accept_ratio=.1):
    """
    Backtracking linesearch, where expected_improve_rate is the slope dy/dx at the initial point
    """
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
    """
    Demmel p 312
    """
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
        z = f_Ax(p)
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
    if verbose: print fmtstr % (i+1, rdotr, np.linalg.norm(x)) # pylint: disable=W0631
    return x