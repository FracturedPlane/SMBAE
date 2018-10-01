import theano
from theano import tensor as T
import numpy as np
import lasagne
import sys
import copy
sys.path.append('../')
from model.ModelUtil import *
from algorithm.AlgorithmInterface import AlgorithmInterface
from collections import OrderedDict

# For debugging
# theano.config.optimizer='fast_compile'
# from DeepCACLA import DeepCACLA
def loglikelihood(a, mean0, std0, d):
    """
        d is the number of action dimensions
    """
    
    # exp[ -(a - mu)^2/(2*sigma^2) ] / sqrt(2*pi*sigma^2)
    return T.reshape(- 0.5 * (T.square(a - mean0) / std0).sum(axis=1) - 0.5 * T.log(2.0 * np.pi) * d - T.log(std0).sum(axis=1), newshape=(-1, 1))
    # return (- 0.5 * T.square((a - mean0) / std0).sum(axis=1) - 0.5 * T.log(2.0 * np.pi) * d - T.log(std0).sum(axis=1))


def likelihood(a, mean0, std0, d):
    return T.exp(loglikelihood(a, mean0, std0, d))

class A3C2(AlgorithmInterface):
    
    def __init__(self, model, n_in, n_out, state_bounds, action_bounds, reward_bound, settings_):

        super(A3C2,self).__init__(model, n_in, n_out, state_bounds, action_bounds, reward_bound, settings_)
        
        # create a small convolutional neural network
        
        self._Fallen = T.bcol("Fallen")
        ## because float64 <= float32 * int32, need to use int16 or int8
        self._Fallen.tag.test_value = np.zeros((self._batch_size,1),dtype=np.dtype('int8'))
        
        self._fallen_shared = theano.shared(
            np.zeros((self._batch_size, 1), dtype='int8'),
            broadcastable=(False, True))
        
        self._advantage = T.col("Tmp_Diff")
        self._advantage.tag.test_value = np.zeros((self._batch_size,1),dtype=np.dtype(self.getSettings()['float_type']))
        
        self._advantage_shared = theano.shared(
            np.zeros((self._batch_size, 1), dtype=self.getSettings()['float_type']),
            broadcastable=(False, True))
        
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
        self._q_valsTargetNextState = lasagne.layers.get_output(self._modelTarget.getCriticNetwork(), self._model.getResultStateSymbolicVariable())
        self._q_valsTarget = lasagne.layers.get_output(self._modelTarget.getCriticNetwork(), self._model.getStateSymbolicVariable())
        self._q_valsTarget_drop = lasagne.layers.get_output(self._modelTarget.getCriticNetwork(), self._model.getStateSymbolicVariable(), deterministic=False)
        
        self._q_valsActA = lasagne.layers.get_output(self._model.getActorNetwork(), self._model.getStateSymbolicVariable(), deterministic=True)[:,:self._action_length]
        self._q_valsActASTD = lasagne.layers.get_output(self._model.getActorNetwork(), self._model.getStateSymbolicVariable(), deterministic=True)[:,self._action_length:]
        
        ## prevent value from being 0
        self._q_valsActASTD = self._q_valsActASTD + 1e-3
        self._q_valsActTarget = lasagne.layers.get_output(self._modelTarget.getActorNetwork(), self._model.getStateSymbolicVariable())[:,:self._action_length]
        self._q_valsActTargetSTD = lasagne.layers.get_output(self._modelTarget.getActorNetwork(), self._model.getStateSymbolicVariable())[:,self._action_length:]
        self._q_valsActTargetSTD = self._q_valsActTargetSTD + 1e-3
        self._q_valsActA_drop = lasagne.layers.get_output(self._model.getActorNetwork(), self._model.getStateSymbolicVariable(), deterministic=False)
        
        self._q_func = self._q_valsA
        self._q_funcTarget = self._q_valsTarget
        self._q_func_drop = self._q_valsA_drop
        self._q_funcTarget_drop = self._q_valsTarget_drop
        self._q_funcAct = self._q_valsActA
        self._q_funcAct_drop = self._q_valsActA_drop
        
        N = self._model.getStateSymbolicVariable().shape[0]
        
        # self._target = (self._model.getRewardSymbolicVariable() + (np.array([self._discount_factor] ,dtype=np.dtype(self.getSettings()['float_type']))[0] * self._q_valsTargetNextState )) * self._Fallen
        self._target = T.mul(T.add(self._model.getRewardSymbolicVariable(), T.mul(self._discount_factor, self._q_valsTargetNextState )), self._Fallen)
        self._diff = self._target - self._q_func
        # self._Advantage = self._diff
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
            self._Fallen: self._fallen_shared
            # self._advantage: self._advantage_shared
        }
        
        self._critic_regularization = (self._critic_regularization_weight * lasagne.regularization.regularize_network_params(
        self._model.getCriticNetwork(), lasagne.regularization.l2))
        self._actor_regularization = (self._regularization_weight * lasagne.regularization.regularize_network_params(
                self._model.getActorNetwork(), lasagne.regularization.l2))
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
        #                                                                theano.tensor.tile((self._advantage * (1.0/(1.0-self._discount_factor))), self._action_length)) # Target network does not work well here?
        # self._actLoss
        # self._actDiff = (self._model.getActionSymbolicVariable() - self._q_valsActA)
        # self._actDiff = ((self._model.getActionSymbolicVariable() - self._q_valsActA)) # Target network does not work well here?
        # self._actDiff_drop = ((self._model.getActionSymbolicVariable() - self._q_valsActA_drop)) # Target network does not work well here?
        ## This should be a single column vector
        # self._actLoss_ = theano.tensor.elemwise.Elemwise(theano.scalar.mul)(( (T.mean(T.pow(self._actDiff, 2),axis=1) )), (self._diff * (1.0/(1.0-self._discount_factor))))
        # self._actLoss_ = theano.tensor.elemwise.Elemwise(theano.scalar.mul)(( T.reshape(T.sum(T.pow(self._actDiff, 2),axis=1), (self._batch_size, 1) )), 
        #                                                                        (self._advantage * (1.0/(1.0-self._discount_factor)))
        # self._actLoss_ = (T.mean(T.pow(self._actDiff, 2),axis=1))
        # self._Advantage = theano.tensor.tile(theano.gradient.disconnected_grad(self._diff), self._action_length)
        # self._Advantage = theano.gradient.disconnected_grad(self._diff)
        # self._Advantage = theano.tensor.clip(self._diff * (1.0/(1.0-self._discount_factor)), 0, 100000.0) ## scale back to same as rewards
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
        # self._actLoss_ = theano.tensor.elemwise.Elemwise(theano.scalar.mul)((self._log_prob), self._Advantage)
        # self._actLoss_ = T.mean(self._log_prob) 
        self._policy_entropy = 0.5 * T.mean(T.log(2 * np.pi * self._q_valsActASTD ) + 1 )
        ## - because update computes gradient DESCENT updates
        self._actLoss = -1.0 * ((T.mean(self._actLoss_)) + (self._policy_entropy * 1e-2))
        # self._actLoss_drop = (T.sum(0.5 * self._actDiff_drop ** 2)/float(self._batch_size)) # because the number of rows can shrink
        # self._actLoss_drop = (T.mean(0.5 * self._actDiff_drop ** 2))
        # self._policy_grad = T.grad(self._actLoss ,  self._actionParams)
        # self._policy_grad = self._actLoss
        # self._policy_grad = lasagne.updates.total_norm_constraint(self._policy_grad, 5)
        # steps, self._actionUpdates = get_adam_steps_and_updates(self._policy_grad, self._actionParams, self._learning_rate)
        # self._actionUpdates = adam_updates(self._actLoss, self._actionParams, self._learning_rate)
        
        if (self.getSettings()['optimizer'] == 'rmsprop'):
            self._actionUpdates = lasagne.updates.rmsprop(self._actLoss , self._actionParams, 
                    self._learning_rate , self._rho, self._rms_epsilon)
        elif (self.getSettings()['optimizer'] == 'momentum'):
            self._actionUpdates = lasagne.updates.momentum(self._actLoss , self._actionParams, 
                    self._learning_rate , momentum=self._rho)
        elif ( self.getSettings()['optimizer'] == 'adam'):
            self._actionUpdates = lasagne.updates.adam(self._actLoss , self._actionParams, 
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
        A3C2.compile(self)
        
    def compile(self):
        
        #### Stuff for Debugging #####
        self._get_diff = theano.function([], [self._diff], givens=self._givens_)
        self._get_advantage = theano.function([], [self._Advantage], givens=self._givens_)
        self._get_target = theano.function([], [self._target], givens={
            # self._model.getStateSymbolicVariable(): self._model.getStates(),
            self._model.getResultStateSymbolicVariable(): self._model.getResultStates(),
            self._model.getRewardSymbolicVariable(): self._model.getRewards(),
            self._Fallen: self._fallen_shared
            # self._model.getActionSymbolicVariable(): self._actions_shared,
        })
        self._get_critic_regularization = theano.function([], [self._critic_regularization])
        self._get_critic_loss = theano.function([], [self._loss], givens=self._givens_)
        
        self._get_actor_regularization = theano.function([], [self._actor_regularization])
        self._get_actor_loss = theano.function([], [self._actLoss], givens=self._actGivens)
        # self._get_actor_diff_ = theano.function([], [self._actDiff], givens= self._actGivens)
        """{
            self._model.getStateSymbolicVariable(): self._model.getStates(),
            self._model.getResultStateSymbolicVariable(): self._model.getResultStates(),
            self._model.getRewardSymbolicVariable(): self._model.getRewards(),
            self._model.getActionSymbolicVariable(): self._model.getActions(),
            self._Fallen: self._fallen_shared
        }) """
        
        self._get_action_diff = theano.function([], [self._actLoss_], givens=self._actGivens)
        
        
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
        # print ("Performing Critic trainning update")
        # if (( self._updates % self._weight_update_steps) == 0):
        all_paramsActA = lasagne.layers.helper.get_all_param_values(self._model.getActorNetwork())
        lasagne.layers.helper.set_all_param_values(self._modelTarget.getActorNetwork(), all_paramsActA)
        #     self.updateTargetModel()
        # self._updates += 1
        # loss, _ = self._train()
        """
        print( "Advantage: ", np.mean(self._get_advantage()))
        print("Actions: ", np.mean(actions, axis=0))
        print("Actions: std", np.std(actions, axis=0))
        print("Policy mean: ", np.mean(self._q_action(), axis=0))
        print("Policy std: ", np.mean(self._q_action_std(), axis=0))
        print("Policy log prob: ", np.mean(self._get_log_prob(), axis=0))
        print("Policy log prob target: ", np.mean(self._get_log_prob_target(), axis=0))
        print( "Actor loss: ", np.mean(self._get_action_diff()))
        """
        
        """
        if ( np.mean(self._get_action_diff()) > 0 ): ## should be impossible
            print("Advantage: ", (self._get_advantage()))
            print("Policy log prob: ", (self._get_log_prob()))
            print("Actions: ", (actions))
            print("Policy mean: ", (self._q_action()))
            print("Policy std: ", (self._q_action_std()))
        """
        # self.setData(states, actions, rewards, result_states, falls)
        # print ("Performing Critic trainning update")
        # if (( self._updates % self._weight_update_steps) == 0):
        #     self.updateTargetModel()
        # self._updates += 1
        # loss, _ = self._train()
        # print( "Actor loss: ", self._get_action_diff())
        lossActor = 0
        """
        diff_ = self.bellman_error(states, actions, rewards, result_states, falls)
        # print ("Diff")
        # print (diff_)
        tmp_states=[]
        tmp_result_states=[]
        tmp_actions=[]
        tmp_rewards=[]
        tmp_falls=[]
        tmp_diff=[]
        for i in range(len(diff_)):
            if ( diff_[i] > 0.0): #positive temporal difference
                tmp_diff.append(diff_[i])
                tmp_states.append(states[i])
                tmp_result_states.append(result_states[i])
                tmp_actions.append(actions[i])
                tmp_rewards.append(rewards[i])
                tmp_falls.append(falls[i])
                
        if (len(tmp_actions) > 0):
           """ 
        # self._advantage_shared.set_value(tmp_diff)
        # self.setData(tmp_states, tmp_actions, tmp_rewards, tmp_result_states, tmp_falls)
    
        print( "Advantage: ", np.mean(self._get_advantage()))
        print("Actions: ", np.mean(actions, axis=0))
        print("Actions: std", np.std(actions, axis=0))
        print("Policy mean: ", np.mean(self._q_action(), axis=0))
        print("Policy std: ", np.mean(self._q_action_std(), axis=0))
        print("Policy log prob target: ", np.mean(self._get_log_prob_target(), axis=0))
        print( "Actor loss: ", np.mean(self._get_action_diff()))
        # print ("Actor diff: ", np.mean(np.array(self._get_diff()) / (1.0/(1.0-self._discount_factor))))
        ## Sometimes really HUGE losses appear
        if (np.abs(np.mean(self._get_action_diff())) < 10): 
            lossActor, _ = self._trainActor()
        print("Policy log prob after: ", np.mean(self._get_log_prob(), axis=0))
        # print( "Length of positive actions: " , str(len(tmp_actions)), " Actor loss: ", lossActor)
        print( " Actor loss: ", lossActor)
            # print("Diff for actor: ", self._get_diff())
            # print ("Tmp_diff: ", tmp_diff)
            # print ( "Action before diff: ", self._get_actor_diff_())
            # print( "Action diff: ", self._get_action_diff())
            # return np.sqrt(lossActor);
        return lossActor
    
    def train(self, states, actions, rewards, result_states, falls):
        loss = self.trainCritic(states, actions, rewards, result_states, falls)
        lossActor = self.trainActor(states, actions, rewards, result_states, falls)
        return loss
    
    def predict(self, state, deterministic_=True):
        # states = np.zeros((self._batch_size, self._state_length), dtype=theano.config.floatX)
        # states[0, ...] = state
        state = norm_state(state, self._state_bounds)
        state = np.array(state, dtype=theano.config.floatX)
        self._model.setStates(state)
        # action_ = lasagne.layers.get_output(self._model.getActorNetwork(), state, deterministic=deterministic_).mean()
        # action_ = scale_action(self._q_action()[0], self._action_bounds)
        # if deterministic_:
        action_ = scale_action(self._q_action()[0], self._action_bounds)
        # action_ = scale_action(self._q_action_target()[0], self._action_bounds)
        # else:
        # action_ = scale_action(self._q_action()[0], self._action_bounds)
        # action_ = q_valsActA[0]
        return action_
