import theano
from theano import tensor as T
import numpy as np
import lasagne
import sys
import copy
sys.path.append('../')
from model.ModelUtil import *
from algorithm.AlgorithmInterface import AlgorithmInterface

# For debugging
# theano.config.mode='FAST_COMPILE'
# from DeepCACLA import DeepCACLA

# @profile(precision=5)
class CACLA(AlgorithmInterface):
    
    # @profile(precision=5)
    def __init__(self, model, n_in, n_out, state_bounds, action_bounds, reward_bound, settings_):

        super(CACLA,self).__init__(model, n_in, n_out, state_bounds, action_bounds, reward_bound, settings_)
        
        # create a small convolutional neural network
        
        self._Fallen = T.bcol("Fallen")
        self._Fallen.tag.test_value = np.zeros((self._batch_size,1),dtype=np.dtype('int8'))
        
        self._fallen_shared = theano.shared(
            np.zeros((self._batch_size, 1), dtype='int8'),
            broadcastable=(False, True))
        self._usingDropout = True
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
        
        self._q_valsActA = lasagne.layers.get_output(self._model.getActorNetwork(), self._model.getStateSymbolicVariable(), deterministic=True)
        self._q_valsActTarget = lasagne.layers.get_output(self._modelTarget.getActorNetwork(), self._model.getStateSymbolicVariable(), deterministic=True)
        self._q_valsActA_drop = lasagne.layers.get_output(self._model.getActorNetwork(), self._model.getStateSymbolicVariable(), deterministic=False)
        
        self._q_func = self._q_valsA
        self._q_funcTarget = self._q_valsTarget
        self._q_func_drop = self._q_valsA_drop
        self._q_funcTarget_drop = self._q_valsTarget_drop
        self._q_funcAct = self._q_valsActA
        self._q_funcAct_drop = self._q_valsActA_drop
        # self._q_funcAct = theano.function(inputs=[self._model.getStateSymbolicVariable()], outputs=self._q_valsActA, allow_input_downcast=True)
        
        # self._target = (self._model.getRewardSymbolicVariable() + (self._discount_factor * self._q_valsTargetNextState )) * theano.tensor.maximum(1.0, theano.tensor.ceil(self._model.getRewardSymbolicVariable())) # Did not understand how the maximum was working
        # self._target = (self._model.getRewardSymbolicVariable() + (self._discount_factor * self._q_valsTargetNextState )) * theano.tensor.ceil(self._model.getRewardSymbolicVariable())
        ## Don't need to use dropout for the target network
        self._target = (self._model.getRewardSymbolicVariable() + (self._discount_factor * self._q_valsTargetNextState )) * self._Fallen
        # self._target = self._model.getTargetSymbolicVariable()
        ## When there is no dropout in the network it will have no affect here
        self._diff = self._target - self._q_func
        self._diff_drop = self._target - self._q_func_drop 
        loss = 0.5 * self._diff ** 2 
        self._loss = T.mean(loss)
        # self._loss_drop = T.mean(0.5 * (self._diff_drop ** 2))
        
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
            # self._model.getResultStateSymbolicVariable(): self._next_states_shared,
            # self._model.getRewardSymbolicVariable(): self._rewards_shared,
            self._model.getActionSymbolicVariable(): self._model.getActions()
        }
        
        self._critic_regularization = (self._critic_regularization_weight * lasagne.regularization.regularize_network_params(
        self._model.getCriticNetwork(), lasagne.regularization.l2))
        self._actor_regularization = (self._regularization_weight * lasagne.regularization.regularize_network_params(
                self._model.getActorNetwork(), lasagne.regularization.l2))
        # SGD update
        # self._updates_ = lasagne.updates.rmsprop(self._loss + self._critic_regularization, self._params, 
        #                         self._learning_rate, self._rho, self._rms_epsilon)
        # TD update
        self._updates_ = lasagne.updates.rmsprop(T.mean(self._q_func) + self._critic_regularization, self._params, 
                    self._critic_learning_rate * -T.mean(self._diff), self._rho, self._rms_epsilon)
        
        
        # actDiff1 = (self._model.getActionSymbolicVariable() - self._q_valsActTarget) #TODO is this correct?
        # actDiff = (actDiff1 - (self._model.getActionSymbolicVariable() - self._q_valsActA))
        # self._actDiff = ((self._model.getActionSymbolicVariable() - self._q_valsActA)) # Target network does not work well here?
        self._actDiff = ((self._model.getActionSymbolicVariable() - self._q_valsActA_drop)) # Target network does not work well here?
        # self._actLoss = 0.5 * (self._actDiff ** 2) 
        ## Should produce a single column vector or costs for each sample in the batch
        self._actLoss_ = T.mean(T.pow(self._actDiff, 2),axis=1) 
        # self._actLoss = T.sum(self._actLoss)/float(self._batch_size) 
        self._actLoss = T.mean(self._actLoss_)
        # self._actLoss_drop = (T.sum(0.5 * self._actDiff_drop ** 2)/float(self._batch_size)) # because the number of rows can shrink
        # self._actLoss_drop = (T.mean(0.5 * self._actDiff_drop ** 2))
        ## Computes the distance between actions weighted by the distances between the states that result in those actions
        """
        state_sum = T.mean(T.pow(self._model.getStateSymbolicVariable(),2), axis=1)
        Distance = ((((state_sum + T.reshape(state_sum, (1,-1)).T) - 2*T.dot(self._model.getStateSymbolicVariable(), self._model.getStateSymbolicVariable().T))))
        action_sum = T.mean(T.pow(self._q_valsActA_drop,2), axis=1)
        Distance_action = ((((action_sum + T.reshape(action_sum, (1,-1)).T) - 2*T.dot(self._q_valsActA_drop, self._q_valsActA_drop.T))))
        weighted_dist = theano.tensor.elemwise.Elemwise(theano.scalar.mul)(Distance, Distance_action)
        self._weighted_mean_dist = T.mean(weighted_dist, axis=1)
        """
        ## Entropy from A3C, make sure network is not producing same action for everything..
        # self.entropy = -T.mean(T.sum(self._q_valsActA_drop, axis=0))
        # self._weighted_entropy = -T.mean(self._weighted_mean_dist)
        self._weighted_entropy = 0
        
        
        self._actionUpdates = lasagne.updates.rmsprop(self._actLoss + self._actor_regularization + (0.00001 * self._weighted_entropy), 
                                self._actionParams, self._learning_rate , self._rho, self._rms_epsilon)
        
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
        CACLA.compile(self)
    
    # @profile(precision=5)   
    def compile(self):
        """
            Theano uses a lot of memory to compile functions...
        """
        
        #### Stuff for Debugging #####
        self._get_diff = theano.function([], [self._diff], givens=self._givens_)
        self._get_actDiff = theano.function([], [self._actDiff], givens=self._actGivens)
        self._get_target = theano.function([], [self._target], givens={
            # self._model.getStateSymbolicVariable(): self._model.getStates(),
            self._model.getResultStateSymbolicVariable(): self._model.getResultStates(),
            self._model.getRewardSymbolicVariable(): self._model.getRewards(),
            self._Fallen: self._fallen_shared
            # self._model.getActionSymbolicVariable(): self._actions_shared,
        })
        """
        self._get_weighted_mean_dist = theano.function([], [self._weighted_mean_dist], givens={
            self._model.getStateSymbolicVariable(): self._model.getStates()
            # self._model.getResultStateSymbolicVariable(): self._model.getResultStates(),
            # self._model.getRewardSymbolicVariable(): self._model.getRewards(),
            # self._Fallen: self._fallen_shared
            # self._model.getActionSymbolicVariable(): self._actions_shared,
        })
        """
        self._get_critic_regularization = theano.function([], [self._critic_regularization])
        self._get_critic_loss = theano.function([], [self._loss], givens=self._givens_)
        
        self._get_actor_regularization = theano.function([], [self._actor_regularization])
        self._get_actor_loss = theano.function([], [self._actLoss], givens=self._actGivens)
        self._get_actor_batch_loss = theano.function([], [self._actLoss_], givens=self._actGivens)
        
        
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
        self._q_action_target = theano.function([], self._q_valsActTarget,
                                       givens={self._model.getStateSymbolicVariable(): self._model.getStates()})
        # self._bellman_error_drop = theano.function(inputs=[self._model.getStateSymbolicVariable(), self._model.getRewardSymbolicVariable(), self._model.getResultStateSymbolicVariable()], outputs=self._diff_drop, allow_input_downcast=True)
        # self._bellman_error_drop2 = theano.function(inputs=[], outputs=self._diff_drop, allow_input_downcast=True, givens=self._givens_)
        
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
    
    def trainActor(self, states, actions, rewards, result_states, falls):
        self.setData(states, actions, rewards, result_states, falls)
        # print ("Performing Critic trainning update")
        # if (( self._updates % self._weight_update_steps) == 0):
        #     self.updateTargetModel()
        # self._updates += 1
        # loss, _ = self._train()
        lossActor = 0
        
        diff_ = self.bellman_error(states, actions, rewards, result_states, falls, advantage=None)
        # print ("Diff")
        # print (diff_)
        tmp_states=[]
        tmp_result_states=[]
        tmp_actions=[]
        tmp_rewards=[]
        tmp_falls=[]
        for i in range(len(diff_)):
            if ( diff_[i] > 0.0):
                tmp_states.append(states[i])
                tmp_result_states.append(result_states[i])
                tmp_actions.append(actions[i])
                tmp_rewards.append(rewards[i])
                tmp_falls.append(falls[i])
                
        if (len(tmp_actions) > 0):
            self.setData(tmp_states, tmp_actions, tmp_rewards, tmp_result_states, tmp_falls)
            # print ("Actions: ", self._q_action_drop())
            # print ("weighted dist: ", self._get_weighted_mean_dist())
            lossActor, _ = self._trainActor()
            print( "Length of positive actions: " , str(len(tmp_actions)), " Actor loss: ", lossActor)
            # print ( " Actions: ", tmp_actions)
            # print ( "Actor diff: " , self._get_actDiff())
            # return np.sqrt(lossActor);
            # print ("batch loss: ", self._get_actor_batch_loss())
        else:
            print ("Length of BAD positive actions: ", len(tmp_actions))
        return lossActor
    
    def train(self, states, actions, rewards, result_states, falls):
        loss = self.trainCritic(states, actions, rewards, result_states, falls)
        lossActor = self.trainActor(states, actions, rewards, result_states, falls)
        return loss
