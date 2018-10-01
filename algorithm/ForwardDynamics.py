import theano
from theano import tensor as T
import numpy as np
import lasagne
import sys
sys.path.append('../')
from model.ModelUtil import *
from model.LearningUtil import loglikelihood, loglikelihoodMEAN, kl, entropy, flatgrad, zipsame, get_params_flat, setFromFlat, likelihood, loglikelihoodMEAN

# For debugging
# theano.config.mode='FAST_COMPILE'
from algorithm.AlgorithmInterface import AlgorithmInterface

class ForwardDynamics(AlgorithmInterface):
    
    def __init__(self, model, state_length, action_length, state_bounds, action_bounds, settings_):

        super(ForwardDynamics,self).__init__(model, state_length, action_length, state_bounds, action_bounds, 0, settings_)
        self._model = model
        batch_size=32
        self._learning_rate = self.getSettings()["fd_learning_rate"]
        self._regularization_weight = 1e-6
        # data types for model
        # create a small convolutional neural network
        
        self._fd_grad_target = T.matrix("FD_Grad")
        self._fd_grad_target.tag.test_value = np.zeros((self._batch_size,self._state_length), dtype=np.dtype(self.getSettings()['float_type']))
        self._fd_grad_target_shared = theano.shared(
            np.zeros((self._batch_size, self._state_length),
                      dtype=self.getSettings()['float_type']))
        
        
        condition_reward_on_result_state = False
        self._train_combined_loss = False
        
        self._inputs_ = {
            self._model.getStateSymbolicVariable(): self._model.getStates(),
            self._model.getActionSymbolicVariable(): self._model.getActions(),
        }
        if (condition_reward_on_result_state):
            self._inputs_reward_ = {
                self._model.getStateSymbolicVariable(): self._model.getStates(),
                self._model.getActionSymbolicVariable(): self._model.getActions(),
                self._model.getResultStateSymbolicVariable() : self._model.getResultStates(),
            }
        else:
            self._inputs_reward_ = self._inputs_
        self._forward = lasagne.layers.get_output(self._model.getForwardDynamicsNetwork(), self._inputs_, deterministic=True)[:,:self._state_length]
        ## This drops to ~ 0 so fast.
        self._forward_std = (lasagne.layers.get_output(self._model.getForwardDynamicsNetwork(), self._inputs_, deterministic=True)[:,self._state_length:] * self.getSettings()['exploration_rate'] )+ 5e-3
        self._forward_std_drop = (lasagne.layers.get_output(self._model.getForwardDynamicsNetwork(), self._inputs_, deterministic=True)[:,self._state_length:] * self.getSettings()['exploration_rate']) + 5e-3
        self._forward_drop = lasagne.layers.get_output(self._model.getForwardDynamicsNetwork(), self._inputs_, deterministic=False)[:,:self._state_length]
        self._reward = lasagne.layers.get_output(self._model.getRewardNetwork(), self._inputs_reward_, deterministic=True)
        self._reward_drop = lasagne.layers.get_output(self._model.getRewardNetwork(), self._inputs_reward_, deterministic=False)
        
        if ( 'train_state_encoding' in self.getSettings() and (self.getSettings()['train_state_encoding'])):    
            self._forward_state_encode = lasagne.layers.get_output(self._model._state_encoding_net, self._model.getStates(), deterministic=False)
        
        l2_loss = True
        
        if ('use_stochastic_forward_dynamics' in self.getSettings() and 
            (self.getSettings()['use_stochastic_forward_dynamics'] == True)):
            
            self._diff = loglikelihood(self._model.getResultStateSymbolicVariable(), self._forward_drop, self._forward_std_drop, self._state_length)
            self._policy_entropy = 0.5 * T.mean(T.log(2 * np.pi * self._forward_std + 1 ) )
            self._loss = -1.0 * (T.mean(self._diff) + (self._policy_entropy * 1e-2))
            # self._loss = -1.0 * (T.mean(self._diff) ) 
            
            ### Not used dropout stuff
            self._diff_NoDrop = loglikelihoodMEAN(self._model.getResultStateSymbolicVariable(), self._forward, self._forward_std, self._state_length)
            # self._loss_NoDrop = -1.0 * (T.mean(self._diff_NoDrop) + (self._policy_entropy * 1e-4))
            self._loss_NoDrop = -1.0 * (T.mean(self._diff_NoDrop) )
        else:
            # self._target = (Reward + self._discount_factor * self._q_valsB)
            self._diff = self._model.getResultStateSymbolicVariable() - self._forward_drop
            ## mean across each sate
            if (l2_loss):
                self._loss = T.mean(T.pow(self._diff, 2),axis=1)
            else:
                self._loss = T.mean(T.abs_(self._diff),axis=1)
            ## mean over batch
            self._loss = T.mean(self._loss)
            ## Another version that does not have dropout
            self._diff_NoDrop = self._model.getResultStateSymbolicVariable() - self._forward
            ## mean across each sate
            if (l2_loss):
                self._loss_NoDrop = T.mean(T.pow(self._diff_NoDrop, 2),axis=1)
            else:
                self._loss_NoDrop = T.mean(T.abs_(self._diff_NoDrop),axis=1)
            ## mean over batch
            self._loss_NoDrop = T.mean(self._loss_NoDrop)
            
        ## Scale the reward value back to proper values.
        ## because rewards are noramlized then scaled by the discount factor to the value stay between -1,1.
        self._reward_diff = (self._model.getRewardSymbolicVariable() * (1.0 / (1.0 - self.getSettings()['discount_factor']))) - self._reward_drop
        self.__Reward = self._model.getRewardSymbolicVariable()
        print ("self.__Reward", self.__Reward)
        # self._reward_diff = (self._model.getRewardSymbolicVariable()) - self._reward_drop
        self._reward_loss_ = T.mean(T.pow(self._reward_diff, 2),axis=1)
        self._reward_loss = T.mean(self._reward_loss_)
        
        self._reward_diff_NoDrop = (self._model.getRewardSymbolicVariable()* (1.0 / (1.0- self.getSettings()['discount_factor']))) - self._reward
        # self._reward_diff_NoDrop = (self._model.getRewardSymbolicVariable()) - self._reward
        self._reward_loss_NoDrop_ = T.mean(T.pow(self._reward_diff_NoDrop, 2),axis=1)
        self._reward_loss_NoDrop = T.mean(self._reward_loss_NoDrop_)
        
        if ( 'train_state_encoding' in self.getSettings() and (self.getSettings()['train_state_encoding'])):
            self._state_encode_diff = self._forward_state_encode - self._model.getStateSymbolicVariable()
            self._state_encode_loss = T.mean(T.mean(T.pow(self._reward_diff, 2),axis=1))
        
        self._params = lasagne.layers.helper.get_all_params(self._model.getForwardDynamicsNetwork())
        self._reward_params = lasagne.layers.helper.get_all_params(self._model.getRewardNetwork())
        if ( 'train_state_encoding' in self.getSettings() and (self.getSettings()['train_state_encoding'])):
            self._encode_params = lasagne.layers.helper.get_all_params(self._model._state_encoding_net)
        self._givens_ = {
            self._model.getStateSymbolicVariable() : self._model.getStates(),
            self._model.getResultStateSymbolicVariable() : self._model.getResultStates(),
            self._model.getActionSymbolicVariable(): self._model.getActions(),
        }
        
        if (condition_reward_on_result_state):
            self._reward_givens_ = {
                self._model.getStateSymbolicVariable() : self._model.getStates(),
                self._model.getResultStateSymbolicVariable() : self._model.getResultStates(),
                self._model.getActionSymbolicVariable(): self._model.getActions(),
                self._model.getRewardSymbolicVariable() : self._model.getRewards(),
            }
        else:
            self._reward_givens_ = {
                self._model.getStateSymbolicVariable() : self._model.getStates(),
                # self._model.getResultStateSymbolicVariable() : self._model.getResultStates(),
                self._model.getActionSymbolicVariable(): self._model.getActions(),
                self._model.getRewardSymbolicVariable() : self._model.getRewards(),
            }
        if ( 'train_state_encoding' in self.getSettings() and (self.getSettings()['train_state_encoding'])):
            self._state_encode_givens_ = {
                self._model.getStateSymbolicVariable() : self._model.getStates(),
            }

        # SGD update
        print ("Optimizing Forward Dynamics with ", self.getSettings()['optimizer'], " method")
        self._updates_ = lasagne.updates.adam(self._loss + (self._regularization_weight * lasagne.regularization.regularize_network_params(
            self._model.getForwardDynamicsNetwork(), lasagne.regularization.l2)), self._params, self._learning_rate, beta1=0.9, beta2=0.999, epsilon=self._rms_epsilon)
        self._reward_updates_ = lasagne.updates.adam(self._reward_loss + (self._regularization_weight * lasagne.regularization.regularize_network_params(
            self._model.getRewardNetwork(), lasagne.regularization.l2)), self._reward_params, self._learning_rate, beta1=0.9, beta2=0.999, epsilon=self._rms_epsilon)
        self._combined_updates_ = lasagne.updates.adam(self._loss + self._reward_loss + (self._regularization_weight * lasagne.regularization.regularize_network_params(
            self._model.getForwardDynamicsNetwork(), lasagne.regularization.l2)), self._params + self._reward_params , self._learning_rate, beta1=0.9, beta2=0.999, epsilon=self._rms_epsilon)
        if ( 'train_state_encoding' in self.getSettings() and (self.getSettings()['train_state_encoding'])):
            self._state_encoding_updates_ = lasagne.updates.adam(self._encode_loss + (self._regularization_weight * lasagne.regularization.regularize_network_params(
                self._model._state_encoding_net, lasagne.regularization.l2)), self._encode_params, self._learning_rate, beta1=0.9, beta2=0.999, epsilon=self._rms_epsilon)
        
        self._combined_givens_ = {
            self._model.getStateSymbolicVariable() : self._model.getStates(),
            self._model.getResultStateSymbolicVariable() : self._model.getResultStates(),
            self._model.getActionSymbolicVariable(): self._model.getActions(),
            self._model.getRewardSymbolicVariable() : self._model.getRewards(),
        }
        
        self._combined_loss = self._reward_loss + self._loss
     
        self._train = theano.function([], [self._loss], updates=self._updates_, givens=self._givens_)
        self._train_reward = theano.function([], [self._reward_loss], updates=self._reward_updates_, givens=self._reward_givens_)
        self._train_combined = theano.function([], [self._reward_loss], updates=self._combined_updates_, givens=self._combined_givens_)
        if ( 'train_state_encoding' in self.getSettings() and (self.getSettings()['train_state_encoding'])):
            self._train_state_encoding = theano.function([], [self._encoding_loss], updates=self._state_encoding_updates_, givens=self._state_encode_givens_)
        self._forwardDynamics = theano.function([], self._forward,
                                       givens={self._model.getStateSymbolicVariable() : self._model.getStates(),
                                                self._model.getActionSymbolicVariable(): self._model.getActions()})
        self._forwardDynamics_drop = theano.function([], self._forward_drop,
                                       givens={self._model.getStateSymbolicVariable() : self._model.getStates(),
                                                self._model.getActionSymbolicVariable(): self._model.getActions()})
        self._forwardDynamics_std = theano.function([], self._forward_std,
                                       givens=self._inputs_)
        self._predict_reward = theano.function([], self._reward,
                                       givens=self._inputs_reward_)
        
        self._bellman_error = theano.function(inputs=[], outputs=self._loss, allow_input_downcast=True, givens=self._givens_)
        self._reward_error = theano.function(inputs=[], outputs=self._reward_diff, allow_input_downcast=True, givens=self._reward_givens_)
        self._reward_values = theano.function(inputs=[], outputs=self.__Reward, allow_input_downcast=True, givens={
                                # self._model.getStateSymbolicVariable() : self._model.getStates(),
                                # self._model.getResultStateSymbolicVariable() : self._model.getResultStates(),
                                # self._model.getActionSymbolicVariable(): self._model.getActions(),
                                self._model.getRewardSymbolicVariable() : self._model.getRewards(),
                            })
        
        # self._diffs = theano.function(input=[State])
        self._get_grad_old = theano.function([], outputs=lasagne.updates.get_or_compute_grads(self._loss_NoDrop, [self._model._actionInputVar] + self._params), allow_input_downcast=True, givens=self._givens_)
        self._get_grad = theano.function([], outputs=T.grad(cost=None, wrt=[self._model._actionInputVar] + self._params,
                                                            known_grads={self._forward: self._fd_grad_target_shared}), 
                                         allow_input_downcast=True, 
                                         givens= self._inputs_)
        
        """
        self._get_grad = theano.function([], outputs=lasagne.updates.get_or_compute_grads(loss_or_grads=self._fd_grad_target, 
                                                                                          params=[self._model._actionInputVar] + self._params,
                                                            ), 
                                         allow_input_downcast=True, 
                                         givens= {
            self._model.getStateSymbolicVariable() : self._model.getStates(),
            # self._model.getResultStateSymbolicVariable() : self._model.getResultStates(),
            self._model.getActionSymbolicVariable(): self._model.getActions(),
            # self._fd_grad_target : self._fd_grad_target_shared
        })
        """
        # self._get_grad_reward = theano.function([], outputs=lasagne.updates.get_or_compute_grads((self._reward_loss_NoDrop), [lasagne.layers.get_all_layers(self._model.getRewardNetwork())[0].input_var] + self._reward_params), allow_input_downcast=True,
        self._get_grad_reward = theano.function([], outputs=lasagne.updates.get_or_compute_grads(T.mean(self._reward), [self._model._actionInputVar] + self._reward_params), allow_input_downcast=True, 
                                                givens=self._inputs_reward_)

    def getNetworkParameters(self):
        params = []
        params.append(lasagne.layers.helper.get_all_param_values(self._model.getForwardDynamicsNetwork()))
        params.append(lasagne.layers.helper.get_all_param_values(self._model.getRewardNetwork()))
        return params
    
    def setNetworkParameters(self, params):
        lasagne.layers.helper.set_all_param_values(self._model.getForwardDynamicsNetwork(), params[0])
        lasagne.layers.helper.set_all_param_values(self._model.getRewardNetwork(), params[1])
        
    def setData(self, states, actions, result_states=None, rewards=None):
        self._model.setStates(states)
        if not (result_states is None):
            self._model.setResultStates(result_states)
        self._model.setActions(actions)
        if not (rewards is None):
            self._model.setRewards(rewards)
            
    def setGradTarget(self, grad):
        self._fd_grad_target_shared.set_value(grad)
        
    def getGrads(self, states, actions, result_states, v_grad=None, alreadyNormed=False):
        if ( alreadyNormed == False ):
            states = np.array(norm_state(states, self._state_bounds), dtype=self.getSettings()['float_type'])
            actions = np.array(norm_action(actions, self._action_bounds), dtype=self.getSettings()['float_type'])
            result_states = np.array(norm_state(result_states, self._state_bounds), dtype=self.getSettings()['float_type'])
        # result_states = np.array(result_states, dtype=self.getSettings()['float_type'])
        self.setData(states, actions, result_states)
        # if (v_grad != None):
        self.setGradTarget(v_grad)
        return self._get_grad()
    
    def getGradsOld(self, states, actions, result_states):
        states = np.array(norm_state(states, self._state_bounds), dtype=self.getSettings()['float_type'])
        actions = np.array(norm_action(actions, self._action_bounds), dtype=self.getSettings()['float_type'])
        result_states = np.array(norm_state(result_states, self._state_bounds), dtype=self.getSettings()['float_type'])
        # result_states = np.array(result_states, dtype=self.getSettings()['float_type'])
        self.setData(states, actions, result_states)
        return self._get_grad_old()
    
    def getRewardGrads(self, states, actions, alreadyNormed=False):
        # states = np.array(states, dtype=self.getSettings()['float_type'])
        # actions = np.array(actions, dtype=self.getSettings()['float_type'])
        if ( alreadyNormed is False ):
            states = np.array(norm_state(states, self._state_bounds), dtype=self.getSettings()['float_type'])
            actions = np.array(norm_action(actions, self._action_bounds), dtype=self.getSettings()['float_type'])
            # rewards = np.array(norm_state(rewards, self._reward_bounds), dtype=self.getSettings()['float_type'])
        self.setData(states, actions)
        return self._get_grad_reward()
                
    def train(self, states, actions, result_states, rewards):
        # rewards = rewards * (1.0/(1.0-self.getSettings()['discount_factor'])) # scale rewards
        self.setData(states, actions, result_states, rewards)
        # print ("Performing Critic trainning update")
        #if (( self._updates % self._weight_update_steps) == 0):
        #    self.updateTargetModel()
        self._updates += 1
        # all_paramsActA = lasagne.layers.helper.get_all_param_values(self._l_outActA)
        if ( self._train_combined_loss ):
            loss = self._train_combined()
            loss = self._train_combined()
        else:
            loss = self._train()
            if ( self.getSettings()['train_reward_predictor']):
                # print ("self._reward_bounds: ", self._reward_bounds)
                # print( "Rewards, predicted_reward, difference, model diff, model rewards: ", np.concatenate((rewards, self._predict_reward(), self._predict_reward() - rewards, self._reward_error(), self._reward_values()), axis=1))
                lossReward = self._train_reward()
                if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
                    print ("Loss Reward: ", lossReward)
            if ( 'train_state_encoding' in self.getSettings() and (self.getSettings()['train_state_encoding'])):
                
                lossEncoding = self._train_state_encoding()
                if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
                    print ("Loss Encoding: ", lossEncoding)     
        # This undoes the Actor parameter updates as a result of the Critic update.
        # print (diff_)
        return loss
    
    def predict(self, state, action):
        # states = np.zeros((self._batch_size, self._self._state_length), dtype=theano.config.floatX)
        # states[0, ...] = state
        state = np.array(norm_state(state, self._state_bounds), dtype=self.getSettings()['float_type'])
        # print ("fd state: ", state)
        action = np.array(norm_action(action, self._action_bounds), dtype=self.getSettings()['float_type'])
        self._model.setStates(state)
        self._model.setActions(action)
        # print ("State bounds: ", self._state_bounds)
        # print ("fd output: ", self._forwardDynamics()[0])
        state_ = scale_state(self._forwardDynamics()[0], self._state_bounds)
        return state_
    
    def predictWithDropout(self, state, action):
        # "dropout"
        state = np.array(norm_state(state, self._state_bounds), dtype=self.getSettings()['float_type'])
        action = np.array(norm_action(action, self._action_bounds), dtype=self.getSettings()['float_type'])
        self._model.setStates(state)
        self._model.setActions(action)
        state_ = scale_state(self._forwardDynamics_drop()[0], self._state_bounds)
        return state_
    
    def predict_std(self, state, action):
        # states = np.zeros((self._batch_size, self._self._state_length), dtype=theano.config.floatX)
        # states[0, ...] = state
        state = np.array(norm_state(state, self._state_bounds), dtype=self.getSettings()['float_type'])
        # print ("fd state: ", state)
        action = np.array(norm_action(action, self._action_bounds), dtype=self.getSettings()['float_type'])
        self._model.setStates(state)
        self._model.setActions(action)
        state_ = self._forwardDynamics_std() * (action_bound_std(self._state_bounds))
        return state_
    
    def predict_reward(self, state, action):
        # states = np.zeros((self._batch_size, self._self._state_length), dtype=theano.config.floatX)
        # states[0, ...] = state
        state = np.array(norm_state(state, self._state_bounds), dtype=self.getSettings()['float_type'])
        action = np.array(norm_action(action, self._action_bounds), dtype=self.getSettings()['float_type'])
        self._model.setStates(state)
        self._model.setActions(action)
        predicted_reward = self._predict_reward()
        reward_ = scale_reward(predicted_reward, self.getRewardBounds()) # * (1.0 / (1.0- self.getSettings()['discount_factor']))
        # reward_ = scale_reward(predicted_reward, self.getRewardBounds())[0] * (1.0 / (1.0- self.getSettings()['discount_factor']))
        # reward_ = scale_state(predicted_reward, self._reward_bounds)
        # print ("reward, predicted reward: ", reward_, predicted_reward)
        return reward_
    
    def predict_batch(self, states, actions):
        ## These input should already be normalized.
        self._model.setStates(states)
        self._model.setActions(actions)
        return self._forwardDynamics()
    
    def predict_reward_batch(self, states, actions):
        """
            This data should already be normalized
        """
        # states = np.zeros((self._batch_size, self._self._state_length), dtype=theano.config.floatX)
        # states[0, ...] = state
        # state = np.array(norm_state(state, self._state_bounds), dtype=self.getSettings()['float_type'])
        # action = np.array(norm_action(action, self._action_bounds), dtype=self.getSettings()['float_type'])
        self._model.setStates(states)
        self._model.setActions(actions)
        predicted_reward = self._predict_reward()
        # reward_ = scale_reward(predicted_reward, self.getRewardBounds())[0] # * (1.0 / (1.0- self.getSettings()['discount_factor']))
        # reward_ = scale_reward(predicted_reward, self.getRewardBounds())[0] * (1.0 / (1.0- self.getSettings()['discount_factor']))
        # reward_ = scale_state(predicted_reward, self._reward_bounds)
        # print ("reward, predicted reward: ", reward_, predicted_reward)
        return predicted_reward

    def bellman_error(self, states, actions, result_states, rewards):
        self.setData(states, actions, result_states, rewards)
        return self._bellman_error()
    
    def reward_error(self, states, actions, result_states, rewards):
        # rewards = rewards * (1.0/(1.0-self.getSettings()['discount_factor'])) # scale rewards
        self.setData(states, actions, result_states, rewards)
        return self._reward_error()
