import theano
from theano import tensor as T
from lasagne.layers import get_all_params
import numpy as np
import lasagne
import sys
import copy
sys.path.append('../')
from model.ModelUtil import *
from algorithm.AlgorithmInterface import AlgorithmInterface
from model.LearningUtil import loglikelihood, kl, entropy, change_penalty

# For debugging
# theano.config.mode='FAST_COMPILE'
# from DeepCACLA import DeepCACLA

class A_CACLA(AlgorithmInterface):
    
    def __init__(self, model, n_in, n_out, state_bounds, action_bounds, reward_bound, settings_):

        super(A_CACLA,self).__init__(model, n_in, n_out, state_bounds, action_bounds, reward_bound, settings_)
        
        # create a small convolutional neural network
        
        self._actor_buffer_states=[]
        self._actor_buffer_result_states=[]
        self._actor_buffer_actions=[]
        self._actor_buffer_rewards=[]
        self._actor_buffer_falls=[]
        self._actor_buffer_diff=[]
        
        self._NotFallen = T.bcol("Not_Fallen")
        ## because float64 <= float32 * int32, need to use int16 or int8
        self._NotFallen.tag.test_value = np.zeros((self._batch_size,1),dtype=np.dtype('int8'))
        
        self._NotFallen_shared = theano.shared(
            np.zeros((self._batch_size, 1), dtype='int8'),
            broadcastable=(False, True))
        
        self._tmp_diff = T.col("Tmp_Diff")
        self._tmp_diff.tag.test_value = np.zeros((self._batch_size,1),dtype=np.dtype(self.getSettings()['float_type']))
        
        self._tmp_diff_shared = theano.shared(
            np.zeros((self._batch_size, 1), dtype=self.getSettings()['float_type']),
            broadcastable=(False, True))
        
        self._dyna_target = T.col("DYNA_Target")
        self._dyna_target.tag.test_value = np.zeros((self._batch_size,1),dtype=np.dtype(self.getSettings()['float_type']))
        
        self._dyna_target_shared = theano.shared(
            np.zeros((self._batch_size, 1), dtype=self.getSettings()['float_type']),
            broadcastable=(False, True))
        
        self._KL_Weight = T.scalar("KL_Weight")
        self._KL_Weight.tag.test_value = np.zeros((1),dtype=np.dtype(self.getSettings()['float_type']))[0]
        
        self._kl_weight_shared = theano.shared(
            np.ones((1), dtype=self.getSettings()['float_type'])[0])
        self._kl_weight_shared.set_value(1.0)
        
        """
        self._target_shared = theano.shared(
            np.zeros((self._batch_size, 1), dtype='float64'),
            broadcastable=(False, True))
        """
        self._critic_regularization_weight = self.getSettings()["critic_regularization_weight"]
        self._critic_learning_rate = self.getSettings()["critic_learning_rate"]
        ## Target network
        self._modelTarget = copy.deepcopy(model)
        
        self._q_valsA = lasagne.layers.get_output(self._model.getCriticNetwork(), self._model.getStateSymbolicVariable(), deterministic=True)
        self._q_valsA_drop = lasagne.layers.get_output(self._model.getCriticNetwork(), self._model.getStateSymbolicVariable(), deterministic=False)
        self._q_valsNextState = lasagne.layers.get_output(self._model.getCriticNetwork(), self._model.getResultStateSymbolicVariable(), deterministic=True)
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
        
        # self._target = (self._model.getRewardSymbolicVariable() + (np.array([self._discount_factor] ,dtype=np.dtype(self.getSettings()['float_type']))[0] * self._q_valsTargetNextState )) * self._NotFallen
        # self._target = self._model.getRewardSymbolicVariable() + ((self._discount_factor * self._q_valsTargetNextState ) * self._NotFallen) + (self._NotFallen - 1)
        self._target = self._model.getRewardSymbolicVariable() + (self._discount_factor * self._q_valsTargetNextState ) 
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
            # self._NotFallen: self._NotFallen_shared
            # self._model.getActionSymbolicVariable(): self._actions_shared,
        }
        self._actGivens = {
            self._model.getStateSymbolicVariable(): self._model.getStates(),
            # self._model.getResultStateSymbolicVariable(): self._model.getResultStates(),
            # self._model.getRewardSymbolicVariable(): self._model.getRewards(),
            self._model.getActionSymbolicVariable(): self._model.getActions(),
            # self._NotFallen: self._NotFallen_shared
            self._tmp_diff: self._tmp_diff_shared
        }
        
        self._critic_regularization = (self._critic_regularization_weight * lasagne.regularization.regularize_network_params(
        self._model.getCriticNetwork(), lasagne.regularization.l2))
        self._actor_regularization = ( (self._regularization_weight * lasagne.regularization.regularize_network_params(
                self._model.getActorNetwork(), lasagne.regularization.l2)) )
        if (self.getSettings()['use_previous_value_regularization']):
            self._actor_regularization = self._actor_regularization + (( self.getSettings()['previous_value_regularization_weight']) * 
                       change_penalty(self._model.getActorNetwork(), self._modelTarget.getActorNetwork()) 
                      )
        elif ('regularization_type' in self.getSettings() and ( self.getSettings()['regularization_type'] == 'KL_Divergence')):
            self._kl_firstfixed = T.mean(kl(self._q_valsActTarget, T.ones_like(self._q_valsActTarget) * self.getSettings()['exploration_rate'], self._q_valsActA, T.ones_like(self._q_valsActA) * self.getSettings()['exploration_rate'], self._action_length))
            #self._actor_regularization = (( self._KL_Weight ) * self._kl_firstfixed ) + (10*(self._kl_firstfixed>self.getSettings()['kl_divergence_threshold'])*
            #                                                                         T.square(self._kl_firstfixed-self.getSettings()['kl_divergence_threshold']))
            self._actor_regularization = (self._kl_firstfixed ) *(self.getSettings()['kl_divergence_threshold'])
            
            print("Using regularization type : ", self.getSettings()['regularization_type']) 
        # SGD update
        # self._updates_ = lasagne.updates.rmsprop(self._loss, self._params, self._learning_rate, self._rho,
        #                                    self._rms_epsilon)
        self._value_grad = T.grad(self._loss + self._critic_regularization
                                                     , self._params)
        ## Clipping the max gradient
        """
        for x in range(len(self._value_grad)): 
            self._value_grad[x] = T.clip(self._value_grad[x] ,  -0.1, 0.1)
        """
        if (self.getSettings()['optimizer'] == 'rmsprop'):
            print ("Optimizing Value Function with ", self.getSettings()['optimizer'], " method")
            self._updates_ = lasagne.updates.rmsprop(self._value_grad
                                                     , self._params, self._learning_rate, self._rho,
                                           self._rms_epsilon)
        elif (self.getSettings()['optimizer'] == 'momentum'):
            print ("Optimizing Value Function with ", self.getSettings()['optimizer'], " method")
            self._updates_ = lasagne.updates.momentum(self._value_grad
                                                      , self._params, self._critic_learning_rate , momentum=self._rho)
        elif ( self.getSettings()['optimizer'] == 'adam'):
            print ("Optimizing Value Function with ", self.getSettings()['optimizer'], " method")
            self._updates_ = lasagne.updates.adam(self._value_grad
                        , self._params, self._critic_learning_rate , beta1=0.9, beta2=0.9, epsilon=self._rms_epsilon)
        elif ( self.getSettings()['optimizer'] == 'adagrad'):
            print ("Optimizing Value Function with ", self.getSettings()['optimizer'], " method")
            self._updates_ = lasagne.updates.adagrad(self._value_grad
                        , self._params, self._critic_learning_rate, epsilon=self._rms_epsilon)
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
        # self._actDiff = theano.tensor.elemwise.Elemwise(theano.scalar.mul)((self._model.getActionSymbolicVariable() - self._q_valsActA), theano.tensor.tile((self._diff * (1.0/(1.0-self._discount_factor))), self._action_length)) # Target network does not work well here?
        self._actDiff = (self._model.getActionSymbolicVariable() - self._q_valsActA_drop)
        # self._actDiff = ((self._model.getActionSymbolicVariable() - self._q_valsActA)) # Target network does not work well here?
        # self._actDiff_drop = ((self._model.getActionSymbolicVariable() - self._q_valsActA_drop)) # Target network does not work well here?
        ## This should be a single column vector
        # self._actLoss_ = theano.tensor.elemwise.Elemwise(theano.scalar.mul)(( T.transpose(T.sum(T.pow(self._actDiff, 2),axis=1) )), (self._diff * (1.0/(1.0-self._discount_factor))))
        # self._actLoss_ = theano.tensor.elemwise.Elemwise(theano.scalar.mul)(( T.reshape(T.sum(T.pow(self._actDiff, 2),axis=1), (self._batch_size, 1) )), 
        #                                                                        (self._tmp_diff * (1.0/(1.0-self._discount_factor)))
        # self._actLoss_ = (T.mean(T.pow(self._actDiff, 2),axis=1))
                                                                                
        self._actLoss_ = theano.tensor.elemwise.Elemwise(theano.scalar.mul)( (T.mean(T.pow(self._actDiff, 2),axis=1)), 
                                                                                (self._tmp_diff * (1.0/(1.0-self._discount_factor)))
                                                                            )
        # self._actLoss = T.sum(self._actLoss)/float(self._batch_size) 
        self._actLoss = T.mean(self._actLoss_) 
        # self._actLoss_drop = (T.sum(0.5 * self._actDiff_drop ** 2)/float(self._batch_size)) # because the number of rows can shrink
        # self._actLoss_drop = (T.mean(0.5 * self._actDiff_drop ** 2))
        self._policy_grad = T.grad(self._actLoss + self._actor_regularization ,  self._actionParams)
        ## Clipping the max gradient
        """
        for x in range(len(self._policy_grad)): 
            self._policy_grad[x] = T.clip(self._policy_grad[x] ,  -0.5, 0.5)
        """
        if (self.getSettings()['optimizer'] == 'rmsprop'):
            self._actionUpdates = lasagne.updates.rmsprop(self._policy_grad, self._actionParams, 
                    self._learning_rate , self._rho, self._rms_epsilon)
        elif (self.getSettings()['optimizer'] == 'momentum'):
            self._actionUpdates = lasagne.updates.momentum(self._policy_grad, self._actionParams, 
                    self._learning_rate , momentum=self._rho)
        elif ( self.getSettings()['optimizer'] == 'adam'):
            self._actionUpdates = lasagne.updates.adam(self._policy_grad, self._actionParams, 
                    self._learning_rate , beta1=0.9, beta2=0.999, epsilon=self._rms_epsilon)
        elif ( self.getSettings()['optimizer'] == 'adagrad'):
            self._actionUpdates = lasagne.updates.adagrad(self._policy_grad, self._actionParams, 
                    self._learning_rate, epsilon=self._rms_epsilon)
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
            # self._model.getActionSymbolicVariable(): self._model.getActions(),
        }
        
        ### Noisey state updates
        # self._target = (self._model.getRewardSymbolicVariable() + (np.array([self._discount_factor] ,dtype=np.dtype(self.getSettings()['float_type']))[0] * self._q_valsTargetNextState )) * self._NotFallen
        # self._target_dyna = theano.gradient.disconnected_grad(self._q_func)
        
        ### _q_valsA because the predicted state is stored in self._model.getStateSymbolicVariable()
        self._diff_dyna = self._dyna_target - self._q_valsNextState
        # loss = 0.5 * self._diff ** 2 
        loss = T.pow(self._diff_dyna, 2)
        self._loss_dyna = T.mean(loss)
        
        self._dyna_grad = T.grad(self._loss_dyna + self._critic_regularization ,  self._params)
        
        self._givens_dyna = {
            # self._model.getStateSymbolicVariable(): self._model.getStates(),
            self._model.getResultStateSymbolicVariable(): self._model.getResultStates(),
            # self._model.getRewardSymbolicVariable(): self._model.getRewards(),
            # self._NotFallen: self._NotFallen_shared
            # self._model.getActionSymbolicVariable(): self._actions_shared,
            self._dyna_target: self._dyna_target_shared
        }
        if (self.getSettings()['optimizer'] == 'rmsprop'):
            self._DYNAUpdates = lasagne.updates.rmsprop(self._dyna_grad, self._params, 
                    self._learning_rate , self._rho, self._rms_epsilon)
        elif (self.getSettings()['optimizer'] == 'momentum'):
            self._DYNAUpdates = lasagne.updates.momentum(self._dyna_grad, self._params, 
                    self._learning_rate , momentum=self._rho)
        elif ( self.getSettings()['optimizer'] == 'adam'):
            self._DYNAUpdates = lasagne.updates.adam(self._dyna_grad, self._params, 
                    self._learning_rate , beta1=0.9, beta2=0.999, epsilon=self._rms_epsilon)
        elif ( self.getSettings()['optimizer'] == 'adagrad'):
            self._DYNAUpdates = lasagne.updates.adagrad(self._dyna_grad, self._params, 
                    self._learning_rate, epsilon=self._rms_epsilon)
        else:
            print ("Unknown optimization method: ", self.getSettings()['optimizer'])
        
        ## Bellman error
        self._bellman = self._target - self._q_funcTarget
        
        # self._target = self._model.getRewardSymbolicVariable() +  (self._discount_factor * self._q_valsTargetNextState )
        ### Give v(s') the next state and v(s) (target) the current state  
        self._diff_adv = (self._discount_factor * self._q_func) - (self._q_valsTargetNextState )
        self._diff_adv_givens = {
            self._model.getStateSymbolicVariable(): self._model.getResultStates(),
            self._model.getResultStateSymbolicVariable(): self._model.getStates(),
        }
        
        A_CACLA.compile(self)
        
    def compile(self):
        
        #### Stuff for Debugging #####
        self._get_diff = theano.function([], [self._diff], givens=self._givens_)
        self._get_target = theano.function([], [self._target], givens={
            # self._model.getStateSymbolicVariable(): self._model.getStates(),
            self._model.getResultStateSymbolicVariable(): self._model.getResultStates(),
            self._model.getRewardSymbolicVariable(): self._model.getRewards(),
            # self._NotFallen: self._NotFallen_shared
            # self._model.getActionSymbolicVariable(): self._actions_shared,
        })
        ## Always want this one
        self._get_critic_loss = theano.function([], [self._loss], givens=self._givens_)
        if (self.getSettings()['debug_critic']):
            self._get_critic_regularization = theano.function([], [self._critic_regularization])
            # self._get_critic_loss = theano.function([], [self._loss], givens=self._givens_)
        
        if (self.getSettings()['debug_actor']):
            if (self.getSettings()['regularization_type'] == 'KL_Divergence'):
                self._get_actor_regularization = theano.function([], [self._actor_regularization], 
                                                                 givens={self._model.getStateSymbolicVariable(): self._model.getStates()})
            else:
                self._get_actor_regularization = theano.function([], [self._actor_regularization])
            self._get_actor_loss = theano.function([], [self._actLoss], givens=self._actGivens)
            self._get_actor_diff_ = theano.function([], [self._actDiff], givens={
                self._model.getStateSymbolicVariable(): self._model.getStates(),
                # self._model.getResultStateSymbolicVariable(): self._model.getResultStates(),
                # self._model.getRewardSymbolicVariable(): self._model.getRewards(),
                self._model.getActionSymbolicVariable(): self._model.getActions()
                # self._NotFallen: self._NotFallen_shared
            }) 
        
        # self._get_action_diff = theano.function([], [self._actLoss_], givens=self._actGivens)
        
        
        self._train = theano.function([], [self._loss, self._q_func], updates=self._updates_, givens=self._givens_)
        self._trainActor = theano.function([], [self._actLoss, self._q_func_drop], updates=self._actionUpdates, givens=self._actGivens)
        self._trainDyna = theano.function([], [self._loss_dyna], updates=self._DYNAUpdates, givens=self._givens_dyna)
        self._q_val = theano.function([], self._q_func,
                                       givens={self._model.getStateSymbolicVariable(): self._model.getStates()})
        self._val_TargetState = theano.function([], self._q_funcTarget,
                                       givens={self._model.getStateSymbolicVariable(): self._modelTarget.getStates()})
        self.get_q_valsTargetNextState = theano.function([], self._q_valsTargetNextState,
                                       givens={self._model.getResultStateSymbolicVariable(): self._model.getResultStates()})
        
        # self._q_val_drop = theano.function([], self._q_func_drop,
        #                                givens={self._model.getStateSymbolicVariable(): self._model.getStates()})
        # self._q_action_drop = theano.function([], self._q_valsActA_drop,
        #                                givens={self._model.getStateSymbolicVariable(): self._model.getStates()})
        self._q_action = theano.function([], self._q_valsActA,
                                       givens={self._model.getStateSymbolicVariable(): self._model.getStates()})
        # self._q_action_target = theano.function([], self._q_valsActTarget,
        #                               givens={self._model.getStateSymbolicVariable(): self._model.getStates()})
        # self._bellman_error_drop = theano.function(inputs=[self._model.getStateSymbolicVariable(), self._model.getRewardSymbolicVariable(), self._model.getResultStateSymbolicVariable()], outputs=self._diff_drop, allow_input_downcast=True)
        # self._bellman_error_drop2 = theano.function(inputs=[], outputs=self._diff_drop, allow_input_downcast=True, givens=self._givens_)
        
        # self._bellman_error = theano.function(inputs=[self._model.getStateSymbolicVariable(), self._model.getResultStateSymbolicVariable(), self._model.getRewardSymbolicVariable()], outputs=self._diff, allow_input_downcast=True)
        self._bellman_error2 = theano.function(inputs=[], outputs=self._diff, allow_input_downcast=True, givens=self._givens_)
        # self._bellman_errorTarget = theano.function(inputs=[], outputs=self._bellman, allow_input_downcast=True, givens=self._givens_)
        # self._diffs = theano.function(input=[self._model.getStateSymbolicVariable()])
        if ( 'mbae_value_use_target_net' in self.getSettings()
             and (self.getSettings()['mbae_value_use_target_net'] == True)
             ):
            self._params_Target = lasagne.layers.helper.get_all_params(self._modelTarget.getCriticNetwork())
            self._get_grad = theano.function([], outputs=lasagne.updates.get_or_compute_grads(T.mean(self._q_funcTarget), [self._model._stateInputVar] + self._params_Target), allow_input_downcast=True, givens=self._givens_grad)
        else:
            self._get_grad = theano.function([], outputs=lasagne.updates.get_or_compute_grads(T.mean(self._q_func), [self._model._stateInputVar] + self._params), allow_input_downcast=True, givens=self._givens_grad)
            
        self._get_grad_policy = theano.function([], outputs=lasagne.updates.get_or_compute_grads(self._actLoss, [self._model._stateInputVar] + self._actionParams), allow_input_downcast=True, givens=self._actGivens)
        # self._get_grad = theano.function([], outputs=lasagne.updates.rmsprop(T.mean(self._q_func), [lasagne.layers.get_all_layers(self._model.getCriticNetwork())[0].input_var] + self._params, self._learning_rate , self._rho, self._rms_epsilon), allow_input_downcast=True, givens=self._givens_grad)
        # self._get_grad2 = theano.gof.graph.inputs(lasagne.updates.rmsprop(loss, params, self._learning_rate, self._rho, self._rms_epsilon))
        
    def updateTargetModel(self):
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
            print ("Updating target Model")
        """
            Target model updates
        """
        if (( 'lerp_target_network' in self.getSettings()) and 
            self.getSettings()['lerp_target_network'] ) :
            all_paramsA = lasagne.layers.helper.get_all_param_values(self._model.getCriticNetwork())
            all_paramsB = lasagne.layers.helper.get_all_param_values(self._modelTarget.getCriticNetwork())
            lerp_weight = 0.01
            # vals = lasagne.layers.helper.get_all_param_values(self._l_outActA)
            
            # print ("l_out length: " + str(len(all_paramsA)))
            # print ("l_out length: " + str(all_paramsA[-6:]))
            # print ("l_out[0] length: " + str(all_paramsA[0]))
            # print ("l_out[4] length: " + str(all_paramsA[4]))
            # print ("l_out[5] length: " + str(all_paramsA[5]))
            # print ("l_out[6] length: " + str(all_paramsA[6]))
            # print ("l_out[7] length: " + str(all_paramsA[7]))
            # print ("l_out[11] length: " + str(all_paramsA[11]))
            # print ("param Values")
            all_params = []
            for paramsA, paramsB in zip(all_paramsA, all_paramsB):
                # print ("paramsA: " + str(paramsA))
                # print ("paramsB: " + str(paramsB))
                params = (lerp_weight * paramsA) + ((1.0 - lerp_weight) * paramsB)
                all_params.append(params)
                
            lasagne.layers.helper.set_all_param_values(self._modelTarget.getCriticNetwork(), all_params)
        else:
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
        self._NotFallen_shared.set_value(fallen)
        # diff_ = self.bellman_error(states, actions, rewards, result_states, falls)
        ## Easy fix for computing actor loss
        diff = self._bellman_error2()
        self._tmp_diff_shared.set_value(diff)
        
        # _targets = rewards + (self._discount_factor * self._q_valsTargetNextState )
        
    def trainCritic(self, states, actions, rewards, result_states, falls):
        self.setData(states, actions, rewards, result_states, falls)
        # print ("Performing Critic trainning update")
        # print("Value function rewards: ", rewards)
        if (( self._updates % self._weight_update_steps) == 0):
            self.updateTargetModel()
        self._updates += 1
        """
        critic_grads = self._get_grad()
        for grad in critic_grads:
            print( "Critic grads, min: ", np.min(grad), " max: ", np.max(grad), " avg: ", np.mean(grad))
        """
        # print ("Falls:", falls)
        # print ("Rewards: ", rewards)
        # print ("Target Values: ", self._get_target())
        # print ("V Values: ", np.mean(self._q_val()))
        # print ("diff Values: ", np.mean(self._get_diff()))
        # data = np.append(falls, self._get_target()[0], axis=1)
        # print ("Rewards, Falls, Targets:", np.append(rewards, data, axis=1))
        # print ("Rewards, Falls, Targets:", [rewards, falls, self._get_target()])
        # print ("Actions: ", actions)
        loss = 0
        pre_loss = self._get_critic_loss()[0]
        # print("Critic loss before: ", pre_loss)
        if ( pre_loss < 10.0): ## To protect the critic from odd losses
            loss, _ = self._train()
        else:
            print("Critic loss too large: ", pre_loss)
        
        return loss
    
    def trainActor(self, states, actions, rewards, result_states, falls, advantage, exp_actions=None, forwardDynamicsModel=None):
        self.setData(states, actions, rewards, result_states, falls)
        # print ("Performing Critic trainning update")
        # if (( self._updates % self._weight_update_steps) == 0):
        #     self.updateTargetModel()
        # self._updates += 1
        # loss, _ = self._train()
        # print( "Actor loss: ", self._get_action_diff())
        lossActor = 0
        if ( 'CACLA_use_advantage' in self.getSettings() 
             and (self.getSettings()['CACLA_use_advantage'] == True)):
            # print ("Using advantage for CACLA")
            diff_ = advantage
        else:
            diff_ = self.bellman_error(states, actions, rewards, result_states, falls)
        # print ("Rewards, Values, NextValues, Diff, new Diff")
        # print (np.concatenate((rewards, self._q_val(), self.get_q_valsTargetNextState(),  diff_, self._q_val() - (rewards + (self._discount_factor * self.get_q_valsTargetNextState()))), axis=1))
        """
        tmp_states=[]
        tmp_result_states=[]
        tmp_actions=[]
        tmp_rewards=[]
        tmp_falls=[]
        tmp_diff=[]
        """
        """
            ((not ('only_use_exp_actions_for_poli_updates' in self.getSettings())) or
             (('only_use_exp_actions_for_poli_updates' in self.getSettings())) and
             (not self.getSettings()['only_use_exp_actions_for_poli_updates'])
             )
          )
          or 
            (
             (diff_[i] > 0.0) and 
             (('only_use_exp_actions_for_poli_updates' in self.getSettings())) and
             (self.getSettings()['only_use_exp_actions_for_poli_updates']) and 
               (exp_actions[i] == 1)
            )
        """
        for i in range(len(diff_)):
            if ( (((diff_[i] > 0.0) or ( 'use_advantage_instaed_of_ptd' in self.getSettings() and self.getSettings()['use_advantage_instaed_of_ptd'])) and 
                   (exp_actions[i] != 0))
                  ):
                if (('dont_use_advantage' in self.getSettings()) and self.getSettings()['dont_use_advantage']):
                    # self._actor_buffer_diff.append([1.0 * (1.0-self._discount_factor)])
                    self._actor_buffer_diff.append([1.0])
                    #  print("Not using advantage")
                else:
                    self._actor_buffer_diff.append(diff_[i])
                self._actor_buffer_states.append(states[i])
                self._actor_buffer_actions.append(actions[i])
                self._actor_buffer_rewards.append(rewards[i])
                self._actor_buffer_result_states.append(result_states[i])
                self._actor_buffer_falls.append(falls[i])
        while ( len(self._actor_buffer_diff) > self.getSettings()['batch_size'] ):
            ### Get batch from buffer
            tmp_states = self._actor_buffer_states[:self.getSettings()['batch_size']]
            tmp_actions = self._actor_buffer_actions[:self.getSettings()['batch_size']]
            tmp_rewards = self._actor_buffer_rewards[:self.getSettings()['batch_size']]
            tmp_result_states = self._actor_buffer_result_states[:self.getSettings()['batch_size']]
            tmp_falls = self._actor_buffer_falls[:self.getSettings()['batch_size']]
            tmp_diff = np.array(self._actor_buffer_diff[:self.getSettings()['batch_size']], dtype=self.getSettings()['float_type'])
            self.setData(tmp_states, tmp_actions, tmp_rewards, tmp_result_states, tmp_falls)
            self._tmp_diff_shared.set_value(tmp_diff)
            # print ("Actor diff: ", np.mean(np.array(self._get_diff()) / (1.0/(1.0-self._discount_factor))))
            lossActor, _ = self._trainActor()
            if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
                print( "Length of positive actions: " , str(len(tmp_actions)), " Actor loss: ", lossActor, " actor buffer size: ", len(self._actor_buffer_actions))
                # print ("error:", self._get_actor_loss())
            if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['debug']):
                actions_ = self._q_action()
                print("Mean actions: ", np.mean(actions_, axis=0))
                print("Mean tmp_actions: ", np.mean(tmp_actions, axis=0))
                print("actions std ", np.std(actions_, axis=0))
            ### Remove batch from buffer
            self._actor_buffer_states=self._actor_buffer_states[self.getSettings()['batch_size']:]
            self._actor_buffer_actions = self._actor_buffer_actions[self.getSettings()['batch_size']:]
            self._actor_buffer_rewards = self._actor_buffer_rewards[self.getSettings()['batch_size']:]
            self._actor_buffer_result_states = self._actor_buffer_result_states[self.getSettings()['batch_size']:]
            self._actor_buffer_falls =self._actor_buffer_falls[self.getSettings()['batch_size']:]
            self._actor_buffer_diff = self._actor_buffer_diff[self.getSettings()['batch_size']:]
            # print( " Actor loss: ", lossActor)
            # print("Diff for actor: ", self._get_diff())
            # print ("Tmp_diff: ", tmp_diff)
            # print ( "Action before diff: ", self._get_actor_diff_())
            # print( "Action diff: ", self._get_action_diff())
            # return np.sqrt(lossActor);
            """
            kl_after = self.kl_divergence()
            kl_coeff = self._kl_weight_shared.get_value()
            if kl_after > 1.3*self.getSettings()['kl_divergence_threshold']: 
                kl_coeff *= 1.5
                # if (kl_coeff > 1e-8):
                self._kl_weight_shared.set_value(kl_coeff)
                print "Got KL=%.3f (target %.3f). Increasing penalty coeff => %.3f."%(kl_after, self.getSettings()['kl_divergence_threshold'], kl_coeff)
            elif kl_after < 0.7*self.getSettings()['kl_divergence_threshold']: 
                kl_coeff /= 1.5
                # if (kl_coeff > 1e-8):
                self._kl_weight_shared.set_value(kl_coeff)
                print "Got KL=%.3f (target %.3f). Decreasing penalty coeff => %.3f."%(kl_after, self.getSettings()['kl_divergence_threshold'], kl_coeff)
            else:
                print ("KL=%.3f is close enough to target %.3f."%(kl_after, self.getSettings()['kl_divergence_threshold']))
            print ("KL_divergence: ", self.kl_divergence(), " kl_weight: ", self._kl_weight_shared.get_value())
            """
        return lossActor
    
    def trainDyna(self, predicted_states, actions, rewards, result_states, falls):
        """
            Performs a DYNA type update
            Because I am using target networks a direct DYNA update does nothing. 
            The gradients are not calculated for the target network.
            L(\theta) = (r + V(s'|\theta')) - V(s|\theta))
            Instead what is done is this
            L(\theta) = V(s_1|\theta')) - V(s_2|\theta))
            Where s1 comes from the simulation and s2 is a predicted and noisey value from an fd model
            Parameters
            ----------
            predicted_states : predicted states, s_1
            
            actions : list of actions
                
            rewards : rewards for taking action a_i
            
            result_states : simulated states, s_2
            
            falls: list of flags for whether or not the character fell
            Returns
            -------
            
            loss: the loss for the DYNA type update

        """
        self.setData( result_states, actions, rewards, predicted_states, falls)
        values = self._val_TargetState()
        # print ("Dyna values: ", values)
        self._dyna_target_shared.set_value(values)
        dyna_loss = self._trainDyna()
        return dyna_loss[0]
    
    def train(self, states, actions, rewards, result_states, falls):
        loss = self.trainCritic(states, actions, rewards, result_states, falls)
        lossActor = self.trainActor(states, actions, rewards, result_states, falls)
        return loss

    def _q_action_std(self):
        ones = np.ones((self._model.getStateValues().shape[0], len(self.getActionBounds()[0])))
        return np.array(self.getSettings()["exploration_rate"] * ones)
