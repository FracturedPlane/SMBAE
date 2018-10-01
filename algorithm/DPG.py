import theano
from theano import tensor as T
import numpy as np
import lasagne
import sys
sys.path.append('../')
from model.ModelUtil import *
from algorithm.AlgorithmInterface import AlgorithmInterface


# For debugging
# theano.config.mode='FAST_COMPILE'
from collections import OrderedDict

class DPG(AlgorithmInterface):
    
    def __init__(self, model, n_in, n_out, state_bounds, action_bounds, reward_bound, settings_):
        """
            In order to get this to work we need to be careful not to update the actor parameters
            when updating the critic. This can be an issue when the Concatenating networks together.
            The first first network becomes a part of the second. However you can still access the first
            network by itself but an updates on the second network will effect the first network.
            Care needs to be taken to make sure only the parameters of the second network are updated.
        """
        
        super(DPG,self).__init__( model, n_in, n_out, state_bounds, action_bounds, reward_bound, settings_)

        self._Fallen = T.bcol("Fallen")
        ## because float64 <= float32 * int32, need to use int16 or int8
        self._Fallen.tag.test_value = np.zeros((self._batch_size,1),dtype=np.dtype('int8'))
        
        self._fallen_shared = theano.shared(
            np.zeros((self._batch_size, 1), dtype='int8'),
            broadcastable=(False, True))

        self._Action = T.matrix("Action2")
        self._Action.tag.test_value = np.random.rand(self._batch_size, self._action_length)
                
        self._Tmp_Target = T.col("Tmp_Target")
        self._Tmp_Target.tag.test_value = np.zeros((self._batch_size,1),dtype=np.dtype(self.getSettings()['float_type']))
        
        self._tmp_target_shared = theano.shared(
            np.zeros((self._batch_size, 1), dtype=self.getSettings()['float_type']),
            broadcastable=(False, True))
        
        self._modelTarget = copy.deepcopy(model)
        
            
        # print ("Initial W " + str(self._w_o.get_value()) )
        
        self._learning_rate = self.getSettings()['learning_rate']
        self._discount_factor= self.getSettings()['discount_factor']
        self._rho = self.getSettings()['rho']
        self._rms_epsilon = self.getSettings()['rms_epsilon']
        
        self._weight_update_steps=self.getSettings()['steps_until_target_network_update']
        self._updates=0
        self._decay_weight=self.getSettings()['regularization_weight']
        self._critic_regularization_weight = self.getSettings()["critic_regularization_weight"]
        self._critic_learning_rate = self.getSettings()["critic_learning_rate"]
        
        # self._q_valsA = lasagne.layers.get_output(self._model.getCriticNetwork(), self._model.getStateSymbolicVariable(), deterministic=True)
        # self._q_valsA_drop = lasagne.layers.get_output(self._model.getCriticNetwork(), self._model.getStateSymbolicVariable(), deterministic=False)
        # self._q_valsNextState = lasagne.layers.get_output(self._model.getCriticNetwork(), self._model.getResultStateSymbolicVariable(), deterministic=True)
        # self._q_valsTargetNextState = lasagne.layers.get_output(self._modelTarget.getCriticNetwork(), self._model.getResultStateSymbolicVariable(), deterministic=True)
        # self._q_valsTarget = lasagne.layers.get_output(self._modelTarget.getCriticNetwork(), self._model.getStateSymbolicVariable(), deterministic=True)
        # self._q_valsTarget_drop = lasagne.layers.get_output(self._modelTarget.getCriticNetwork(), self._model.getStateSymbolicVariable(), deterministic=False)
        
        self._q_valsActA = lasagne.layers.get_output(self._model.getActorNetwork(), self._model.getStateSymbolicVariable(), deterministic=True)
        self._q_valsActTarget = lasagne.layers.get_output(self._modelTarget.getActorNetwork(), self._model.getResultStateSymbolicVariable(), deterministic=True)
        # self._q_valsActA_drop = lasagne.layers.get_output(self._model.getActorNetwork(), self._model.getStateSymbolicVariable(), deterministic=False)
        
        
        inputs_1 = {
            self._model.getStateSymbolicVariable(): self._model.getStates(),
            self._model.getActionSymbolicVariable(): self._model.getActions()
        }
        self._q_valsA = lasagne.layers.get_output(self._model.getCriticNetwork(), inputs_1)
        inputs_1_policy = {
            self._model.getStateSymbolicVariable(): self._model.getStates(),
            self._model.getActionSymbolicVariable(): self._q_valsActA
        }
        self._q_vals_train_policy = lasagne.layers.get_output(self._model.getCriticNetwork(), inputs_1_policy)
        inputs_2 = {
            self._modelTarget.getStateSymbolicVariable(): self._model.getResultStates(),
            self._modelTarget.getActionSymbolicVariable(): self._model.getActions()
        }
        self._q_valsB_ = lasagne.layers.get_output(self._modelTarget.getCriticNetwork(), inputs_2, deterministic=True)
        
        self._q_func = self._q_valsA
        self._q_funcB = self._q_valsB_
        # self._q_funcTarget = self._q_valsTarget
        # self._q_func_drop = self._q_valsA_drop
        # self._q_funcTarget_drop = self._q_valsTarget_drop
        self._q_funcAct = self._q_valsActA
        # self._q_funcAct_drop = self._q_valsActA_drop
        
        # self._q_funcAct = theano.function(inputs=[State], outputs=self._q_valsActA, allow_input_downcast=True)
        
        # self._target = T.mul(T.add(self._model.getRewardSymbolicVariable(), T.mul(self._discount_factor, self._q_valsB )), self._Fallen)
        self._diff = self._Tmp_Target - self._q_func
        # self._diff_drop = self._target - self._q_func_drop 
        # loss = 0.5 * self._diff ** 2 
        loss = T.pow(self._diff, 2)
        self._loss = T.mean(loss)
        # self._loss_drop = T.mean(0.5 * self._diff_drop ** 2)
    
        # assert len(lasagne.layers.helper.get_all_params(self._l_outA)) == 16
        # Need to remove the action layers from these params
        self._params = lasagne.layers.helper.get_all_params(self._model.getCriticNetwork()) 
        print ("******Number of Layers is: " + str(len(lasagne.layers.helper.get_all_params(self._model.getCriticNetwork()))))
        print ("******Number of Action Layers is: " + str(len(lasagne.layers.helper.get_all_params(self._model.getActorNetwork()))))
        self._actionParams = lasagne.layers.helper.get_all_params(self._model.getActorNetwork())
        self._givens_ = {
            self._model.getStateSymbolicVariable(): self._model.getStates(),
            self._model.getActionSymbolicVariable():  self._model.getActions(),
            # self._Action:  self._q_valsActTarget,
            # self._model.getResultStateSymbolicVariable(): self._model.getResultStates(),
            # self._model.getRewardSymbolicVariable(): self._model.getRewards(),
            # self._Fallen: self._fallen_shared
            self._Tmp_Target: self._tmp_target_shared
        }
        self._actGivens = {
            self._model.getStateSymbolicVariable(): self._model.getStates(),
            # self._model.getResultStateSymbolicVariable(): self._model.getResultStates(),
            # self._model.getRewardSymbolicVariable(): self._model.getRewards(),
            # self._model.getActionSymbolicVariable(): self._model.getActions(),
            # self._Fallen: self._fallen_shared
            # self._tmp_diff: self._tmp_diff_shared
        }
        
        self._critic_regularization = (self._critic_regularization_weight * 
                                       lasagne.regularization.regularize_network_params(
                                            self._model.getCriticNetwork(), lasagne.regularization.l2))
        
        ## MSE update
        self._value_grad = T.grad(self._loss + self._critic_regularization
                                                     , self._params)
        print ("Optimizing Value Function with ", self.getSettings()['optimizer'], " method")
        self._updates_ = lasagne.updates.adam(self._value_grad
                    , self._params, self._critic_learning_rate , beta1=0.9, beta2=0.9, epsilon=self._rms_epsilon)
        
        self._givens_grad = {
            self._model.getStateSymbolicVariable(): self._model.getStates(),
            # self._model.getResultStateSymbolicVariable(): self._model.getResultStates(),
            # self._model.getRewardSymbolicVariable(): self._model.getRewards(),
            self._model.getActionSymbolicVariable(): self._model.getActions(),
        }
        
        ## Some cool stuff to backprop action gradients
        
        self._action_grad = T.matrix("Action_Grad")
        self._action_grad.tag.test_value = np.zeros((self._batch_size,self._action_length), dtype=np.dtype(self.getSettings()['float_type']))
        
        self._action_grad_shared = theano.shared(
            np.zeros((self._batch_size, self._action_length),
                      dtype=self.getSettings()['float_type']))
        
        ### Maximize wrt q function
        
        self._action_mean_grads = T.grad(cost=None, wrt=self._actionParams,
                                                    known_grads={self._q_valsActA: self._action_grad_shared}),
        print ("Action grads: ", self._action_mean_grads[0])
        ## When passing in gradients it needs to be a proper list of gradient expressions
        self._action_mean_grads = list(self._action_mean_grads[0])
        # print ("isinstance(self._action_mean_grads, list): ", isinstance(self._action_mean_grads, list))
        # print ("Action grads: ", self._action_mean_grads)
        self._actionGRADUpdates = lasagne.updates.adam(self._action_mean_grads, self._actionParams, 
                    self._learning_rate,  beta1=0.9, beta2=0.9, epsilon=self._rms_epsilon)
        
        self._actGradGivens = {
            self._model.getStateSymbolicVariable(): self._model.getStates(),
            # self._model.getResultStateSymbolicVariable(): self._model.getResultStates(),
            # self._model.getRewardSymbolicVariable(): self._model.getRewards(),
            # self._model.getActionSymbolicVariable(): self._model.getActions(),
            # self._Fallen: self._fallen_shared,
            # self._advantage: self._advantage_shared,
            # self._KL_Weight: self._kl_weight_shared
        }
        
        # theano.gradient.grad_clip(x, lower_bound, upper_bound) # // TODO
        # self._actionUpdates = lasagne.updates.adam(-T.mean(self._q_vals_train_policy) + 
        #   (self._decay_weight * lasagne.regularization.regularize_network_params(
        #       self._model.getActorNetwork(), lasagne.regularization.l2)), self._actionParams, 
        #           self._learning_rate, beta1=0.9, beta2=0.9, epsilon=self._rms_epsilon)
        
        
        if ('train_extra_value_function' in self.getSettings() and (self.getSettings()['train_extra_value_function'] == True)):
            self._valsA = lasagne.layers.get_output(self._model._value_function, self._model.getStateSymbolicVariable(), deterministic=True)
            self._valsA_drop = lasagne.layers.get_output(self._model._value_function, self._model.getStateSymbolicVariable(), deterministic=False)
            self._valsNextState = lasagne.layers.get_output(self._model._value_function, self._model.getResultStateSymbolicVariable(), deterministic=True)
            self._valsTargetNextState = lasagne.layers.get_output(self._modelTarget._value_function, self._model.getResultStateSymbolicVariable(), deterministic=True)
            self._valsTarget = lasagne.layers.get_output(self._modelTarget._value_function, self._model.getStateSymbolicVariable(), deterministic=True)
            
            # self._target = T.mul(T.add(self._model.getRewardSymbolicVariable(), T.mul(self._discount_factor, self._q_valsB )), self._Fallen)
            # self._target = self._model.getRewardSymbolicVariable() + ((self._discount_factor * self._q_valsTargetNextState ) * self._NotFallen) + (self._NotFallen - 1)
            self._v_target = self._model.getRewardSymbolicVariable() + (self._discount_factor * self._valsTargetNextState ) 
            self._v_diff = self._v_target - self._valsA
            # loss = 0.5 * self._diff ** 2 
            loss_v = T.pow(self._v_diff, 2)
            self._v_loss = T.mean(loss_v)
            
            self._params_value = lasagne.layers.helper.get_all_params(self._model._value_function)
            self._givens_value = {
                self._model.getStateSymbolicVariable(): self._model.getStates(),
                self._model.getResultStateSymbolicVariable(): self._model.getResultStates(),
                self._model.getRewardSymbolicVariable(): self._model.getRewards(),
                # self._NotFallen: self._NotFallen_shared
                # self._model.getActionSymbolicVariable(): self._actions_shared,
            }
            self._value_regularization = (self._critic_regularization_weight * 
                                          lasagne.regularization.regularize_network_params(
                                        self._model._value_function, lasagne.regularization.l2))
            
            self._value_grad = T.grad(self._v_loss + self._value_regularization
                                                     , self._params_value)
            print ("Optimizing Value Function with ", self.getSettings()['optimizer'], " method")
            self._updates_value = lasagne.updates.adam(self._value_grad
                        , self._params_value, self._critic_learning_rate , beta1=0.9, beta2=0.9, epsilon=self._rms_epsilon)
            ## TD update
        DPG.compile(self)
        
    def compile(self):
        
        self._train = theano.function([], [self._loss, self._q_func], updates=self._updates_, givens=self._givens_)
        if ('train_extra_value_function' in self.getSettings() and (self.getSettings()['train_extra_value_function'])):
            self._train_value = theano.function([], [self._v_loss, self._valsA], updates=self._updates_value, givens=self._givens_value)
        # self._trainActor = theano.function([], [actLoss, self._q_valsActA], updates=actionUpdates, givens=actGivens)
        # self._trainActor = theano.function([], [self._q_func], updates=self._actionUpdates, givens=self._actGivens)
        self._trainActionGRAD  = theano.function([], [], updates=self._actionGRADUpdates, givens=self._actGradGivens)
        self._q_val = theano.function([], self._q_valsA,
                                       givens={self._model.getStateSymbolicVariable(): self._model.getStates(),
                                               self._model.getActionSymbolicVariable(): self._model.getActions()
                                               })
        self._q_val_Target = theano.function([#self._model.getStateSymbolicVariable(), 
                                              #self._model.getActionSymbolicVariable()
                                              ], 
                                             self._q_valsB_,
                                       givens={self._modelTarget.getStateSymbolicVariable(): self._model.getResultStates(),
                                               self._modelTarget.getActionSymbolicVariable(): self._model.getActions()
                                               }
                                             )
        #self._q_val_Target = theano.function([], self._q_valsB_, givens=self._givens_grad)
        self._q_action = theano.function([], self._q_valsActA,
                                       givens={self._model.getStateSymbolicVariable(): self._model.getStates()})
        self._action_Target = theano.function([], self._q_valsActTarget,
                                       givens={self._model.getResultStateSymbolicVariable(): self._model.getResultStates()})
        """
        inputs_ = [
                   self._model.getStateSymbolicVariable(), 
                   self._model.getRewardSymbolicVariable(), 
                   # ResultState
                   ]
        self._bellman_error = theano.function(inputs=inputs_, outputs=self._diff, allow_input_downcast=True)
        """
        # self._diffs = theano.function(input=[State])
        self._bellman_error2 = theano.function(inputs=[], outputs=self._diff, allow_input_downcast=True, givens=self._givens_)
        
        self._get_action_grad = theano.function([], outputs=lasagne.updates.get_or_compute_grads(T.mean(self._q_func), [self._model._actionInputVar] + self._params), allow_input_downcast=True, givens=self._givens_grad)
        
        if ('train_extra_value_function' in self.getSettings() and (self.getSettings()['train_extra_value_function'])):
            self._givens_grad = {
                self._model.getStateSymbolicVariable(): self._model.getStates(),
                # self._model.getResultStateSymbolicVariable(): self._model.getResultStates(),
                # self._model.getRewardSymbolicVariable(): self._model.getRewards(),
                # self._model.getActionSymbolicVariable(): self._model.getActions(),
            }
            self._get_state_grad = theano.function([], outputs=lasagne.updates.get_or_compute_grads(T.mean(self._valsA), [self._model._stateInputVar] + self._params_value), allow_input_downcast=True, givens=self._givens_grad)
        else:
            self._get_state_grad = theano.function([], outputs=lasagne.updates.get_or_compute_grads(T.mean(self._q_func), [self._model._stateInputVar] + self._params), allow_input_downcast=True, givens=self._givens_grad)
        
        
    def getGrads(self, states, actions=None, alreadyNormed=False):
        """
            The states should be normalized
        """
        # self.setData(states, actions, rewards, result_states)
        if ( alreadyNormed == False):
            states = norm_state(states, self._state_bounds)
        states = np.array(states, dtype=theano.config.floatX)
        self._model.setStates(states)
        if ( actions is None ):
            actions = self.predict_batch(states)
        self._model.setActions(actions)
        return self._get_state_grad()
    
    def getActionGrads(self, states, actions=None, alreadyNormed=False):
        """
            The states should be normalized
        """
        # self.setData(states, actions, rewards, result_states)
        if ( alreadyNormed == False):
            states = norm_state(states, self._state_bounds)
        states = np.array(states, dtype=theano.config.floatX)
        self._model.setStates(states)
        if ( actions is None ):
            actions = self.predict_batch(states)
        self._model.setActions(actions)
        return self._get_action_grad()
    
    def updateTargetModel(self):
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
            print ("Updating target Model")
        """
            Target model updates
        """
        # return
        ## I guess it is okay to lerp the entire network even though we only really want to 
        ## lerp the value function part of the networks, the target policy is not used for anythings
        all_paramsA = lasagne.layers.helper.get_all_param_values(self._model.getCriticNetwork())
        all_paramsB = lasagne.layers.helper.get_all_param_values(self._modelTarget.getCriticNetwork())
        all_paramsActA = lasagne.layers.helper.get_all_param_values(self._model.getActorNetwork())
        all_paramsActB = lasagne.layers.helper.get_all_param_values(self._modelTarget.getActorNetwork())
        if ('target_net_interp_weight' in self.getSettings()):
            lerp_weight = self.getSettings()['target_net_interp_weight']
        else:
            lerp_weight = 0.001
        # vals = lasagne.layers.helper.get_all_param_values(self._l_outActA)
        
        all_params = []
        for paramsA, paramsB in zip(all_paramsA, all_paramsB):
            params = (lerp_weight * paramsA) + ((1.0 - lerp_weight) * paramsB)
            all_params.append(params)
        lasagne.layers.helper.set_all_param_values(self._modelTarget.getCriticNetwork(), all_params)
        
        
        all_paramsAct = []
        for paramsA, paramsB in zip(all_paramsActA, all_paramsActB):
            params = (lerp_weight * paramsA) + ((1.0 - lerp_weight) * paramsB)
            all_paramsAct.append(params)
        lasagne.layers.helper.set_all_param_values(self._modelTarget.getActorNetwork(), all_paramsAct)
        
        
    def updateTargetModelValue(self):
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
            print ("Updating MBAE target Model")
        """
            Target model updates
        """
        all_paramsA = lasagne.layers.helper.get_all_param_values(self._model._value_function)
        # all_paramsActA = lasagne.layers.helper.get_all_param_values(self._model.getActorNetwork())
        lasagne.layers.helper.set_all_param_values(self._modelTarget._value_function, all_paramsA)
        # lasagne.layers.helper.set_all_param_values(self._modelTarget.getActorNetwork(), all_paramsActA)
            
    def getNetworkParameters(self):
        params = []
        params.append(lasagne.layers.helper.get_all_param_values(self._model.getCriticNetwork()))
        params.append(lasagne.layers.helper.get_all_param_values(self._model.getActorNetwork()))
        params.append(lasagne.layers.helper.get_all_param_values(self._modelTarget.getCriticNetwork()))
        params.append(lasagne.layers.helper.get_all_param_values(self._modelTarget.getActorNetwork()))
        if ('train_extra_value_function' in self.getSettings() and (self.getSettings()['train_extra_value_function'] == True)):
            params.append(lasagne.layers.helper.get_all_param_values(self._model._value_function))
            params.append(lasagne.layers.helper.get_all_param_values(self._modelTarget._value_function))
        return params
        
    def setNetworkParameters(self, params):
        lasagne.layers.helper.set_all_param_values(self._model.getCriticNetwork(), params[0])
        lasagne.layers.helper.set_all_param_values(self._model.getActorNetwork(), params[1])
        lasagne.layers.helper.set_all_param_values(self._modelTarget.getCriticNetwork(), params[2])
        lasagne.layers.helper.set_all_param_values(self._modelTarget.getActorNetwork(), params[3])
        if ('train_extra_value_function' in self.getSettings() and (self.getSettings()['train_extra_value_function'] == True)):
            lasagne.layers.helper.set_all_param_values(self._model._value_function, params[4])
            lasagne.layers.helper.set_all_param_values(self._modelTarget._value_function, params[5])
        
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
        # diff = self._bellman_error2()
        # self._tmp_diff_shared.set_value(diff)
        
        # _targets = rewards + (self._discount_factor * self._q_valsTargetNextState )
        
    def trainCritic(self, states, actions, rewards, result_states, falls):
        
        self.setData(states, actions, rewards, result_states, falls)
        if ('train_extra_value_function' in self.getSettings() and (self.getSettings()['train_extra_value_function'] == True)):
            loss_v, _ = self._train_value()
            if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['debug']):
                print ("MBAE Value function loss: ", loss_v)
            
            if (( self._updates % 500) == 0):
                self.updateTargetModelValue()
            
        self._updates += 1
        ## Compute actions for TargetNet
        target_actions = self._action_Target()
        self.setData(states, target_actions, rewards, result_states, falls)
        ## Get next q value using target network
        q_vals_b = self._q_val_Target()
        ## Compute target values
        target_tmp_ = rewards + ((self._discount_factor * q_vals_b ))
        self.setData(states, actions, rewards, result_states, falls)
        ### Set learning target (y)
        self._tmp_target_shared.set_value(target_tmp_)
        
        loss, _ = self._train()
        self.updateTargetModel()
        return loss
        
    def trainActor(self, states, actions, rewards, result_states, falls, advantage, exp_actions, forwardDynamicsModel=None):
        self.setData(states, actions, rewards, result_states, falls)
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['debug']):
            print("values: ", np.mean(self._q_val()* (1.0 / (1.0- self.getSettings()['discount_factor']))), " std: ", np.std(self._q_val()* (1.0 / (1.0- self.getSettings()['discount_factor']))) )
            print("Rewards: ", np.mean(rewards), " std: ", np.std(rewards), " shape: ", np.array(rewards).shape)

        loss = 0
        policy_mean = self.predict_batch(states)
        action_grads = self.getActionGrads(states, policy_mean, alreadyNormed=True)[0]
        
        """
            From DEEP REINFORCEMENT LEARNING IN PARAMETERIZED ACTION SPACE
            Hausknecht, Matthew and Stone, Peter
            
            actions.shape == action_grads.shape
        """
        use_parameter_grad_inversion=False
        if ( use_parameter_grad_inversion ):
            for i in range(action_grads.shape[0]):
                for j in range(action_grads.shape[1]):
                    if (action_grads[i,j] > 0):
                        inversion = (1.0 - actions[i,j]) / 2.0
                    else:
                        inversion = ( actions[i,j] - (-1.0)) / 2.0
                    action_grads[i,j] = action_grads[i,j] * inversion
        else:
            # Normalize
            norm = np.reshape(np.linalg.norm(action_grads, axis=1), (action_grads.shape[0], 1))
            action_grads = action_grads / norm
            pass
                
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['debug']):
            print("Policy mean: ", np.mean(self._q_action(), axis=0))
            print("Mean action grad: ", np.mean(action_grads, axis=0), " std ", np.std(action_grads, axis=0))
        
        ## Set data for gradient
        self._model.setStates(states)
        self._modelTarget.setStates(states)
        ## Why the -1.0??
        ## Because the SGD method is always performing MINIMIZATION!!
        self._action_grad_shared.set_value(-1.0*action_grads)
        self._trainActionGRAD()
        
        return loss
        
    def train(self, states, actions, rewards, result_states):
        loss = self.trainCritic(states, actions, rewards, result_states)
        lossActor = self.trainActor(states, actions, rewards, result_states)
        return loss
    
    def q_value(self, state):
        """
            For returning a vector of q values, state should NOT be normalized
        """
        # states = np.zeros((self._batch_size, self._state_length), dtype=theano.config.floatX)
        # states[0, ...] = state
        state = norm_state(state, self._state_bounds)
        state = np.array(state, dtype=theano.config.floatX)
        self._model.setStates(state)
        self._modelTarget.setStates(state)
        action = self._q_action()
        self._model.setActions(action)
        self._modelTarget.setActions(action)
        if ( ('disable_parameter_scaling' in self._settings) and (self._settings['disable_parameter_scaling'])):
            return scale_reward(self._q_val(), self.getRewardBounds()) * (1.0 / (1.0- self.getSettings()['discount_factor']))
            # return (self._q_val())[0]
        else:
            return scale_reward(self._q_val(), self.getRewardBounds()) * (1.0 / (1.0- self.getSettings()['discount_factor']))
        # return self._q_valTarget()[0]
        # return self._q_val()[0]
    
    def q_values(self, state):
        """
            For returning a vector of q values
        """
        state = norm_state(state, self._state_bounds)
        state = np.array(state, dtype=theano.config.floatX)
        self._model.setStates(state)
        self._modelTarget.setStates(state)
        action = self._q_action()
        self._model.setActions(action)
        self._modelTarget.setActions(action)
        if ( ('disable_parameter_scaling' in self._settings) and (self._settings['disable_parameter_scaling'])):
            return scale_reward(self._q_val(), self.getRewardBounds()) * (1.0 / (1.0- self.getSettings()['discount_factor']))
            # return (self._q_val())[0] 
        else:
            return scale_reward(self._q_val(), self.getRewardBounds()) * (1.0 / (1.0- self.getSettings()['discount_factor']))
        # return self._q_valTarget()
        # return self._q_val()
    
    def _q_action_std(self):
        ones = np.ones((self._model.getStateValues().shape[0], len(self.getActionBounds()[0])))
        return np.array(self.getSettings()["exploration_rate"] * ones)