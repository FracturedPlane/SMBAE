import theano
from theano import tensor as T

import numpy as np
# import lasagne
import sys
import copy
sys.path.append('../')
from model.ModelUtil import norm_state, scale_state, norm_action, scale_action, action_bound_std, scale_reward
from algorithm.AlgorithmInterface import AlgorithmInterface
from model.LearningUtil import loglikelihood, likelihood, likelihoodMEAN, kl, kl_D, entropy, flatgrad, zipsame, get_params_flat, setFromFlat
from keras.optimizers import SGD
# from keras.utils.np_utils import to_categoricalnetwork
import keras.backend as K
import keras

# For debugging
# theano.config.mode='FAST_COMPILE'
# from DeepCACLA import DeepCACLA

class QPropKeras(AlgorithmInterface):
    
    def __init__(self, model, n_in, n_out, state_bounds, action_bounds, reward_bound, settings_):

        super(QPropKeras,self).__init__(model, n_in, n_out, state_bounds, action_bounds, reward_bound, settings_)
        
        ## primary network
        self._model = model

        ## Target network for DPG
        self._modelTarget = copy.deepcopy(model)
        ## Target network for PPO
        self._modelTarget2 = copy.deepcopy(model)
        # self._modelTarget = model
        self._learning_rate = self.getSettings()['learning_rate']
        self._discount_factor= self.getSettings()['discount_factor']
        self._rho = self.getSettings()['rho']
        self._rms_epsilon = self.getSettings()['rms_epsilon']
        
        self._q_valsActA = self._model.getActorNetwork()(self._model._stateInput)[:,:self._action_length]
        self._q_valsActASTD = self._model.getActorNetwork()(self._model._stateInput)[:,self._action_length:]
        self._q_valsActTarget_State = self._modelTarget2.getActorNetwork()(self._model._stateInput)[:,:self._action_length]
        
        # self._q_valsActTarget_State = self._modelTarget.getActorNetwork()(self._model._stateInput)[:,:self._action_length]
        # self._q_valsActTargetSTD = self._modelTarget.getActorNetwork()(self._model._stateInput)[:,self._action_length:]
        
        self._q_valsActASTD = ( T.ones_like(self._q_valsActA)) * self.getSettings()['exploration_rate']
        self._q_valsActTargetSTD = (T.ones_like(self._q_valsActTarget_State)) * self.getSettings()['exploration_rate']
        
        self._Advantage = T.col("Advantage")
        self._Advantage.tag.test_value = np.zeros((self._batch_size,1),dtype=np.dtype(self.getSettings()['float_type']))
        self._advantage_shared = theano.shared(
            np.zeros((self._batch_size, 1), dtype=self.getSettings()['float_type']),
            broadcastable=(False, True))
        
        self._LEARNING_PHASE = T.scalar(dtype='uint8', name='keras_learning_phase')  # 0 = test, 1 = train
        
        self._QProp_N = T.col("QProp_N")
        self._QProp_N.tag.test_value = np.zeros((self._batch_size,1),dtype=np.dtype(self.getSettings()['float_type']))
        self._QProp_N_shared = theano.shared(
            np.zeros((self._batch_size, 1), dtype=self.getSettings()['float_type']),
            broadcastable=(False, True)) 
        
        self._q_function = self._model.getCriticNetwork()([self._model._stateInput, self._q_valsActA])
        self._q_function_Target = self._model.getCriticNetwork()([self._model._stateInput, self._model._actionInput])
        # self._value = self._model.getCriticNetwork()([self._model._stateInput, K.learning_phase()])
        self._value_Target = self._modelTarget2.getValueFunction()([self._model._stateInput])
        self._value = self._model.getValueFunction()([self._model._stateInput])
        # self._value = self._model.getCriticNetwork()([self._model._stateInput])
        
        self._actor_entropy = 0.5 * T.mean((2 * np.pi * self._q_valsActASTD ) )
        
        ## Compute on-policy policy gradient
        self._prob = likelihood(self._model._actionInput, self._q_valsActA, self._q_valsActASTD, self._action_length)
        ### How should this work if the target network is very odd, as in not a slightly outdated copy.
        self._prob_target = likelihood(self._model._actionInput, self._q_valsActTarget_State, self._q_valsActTargetSTD, self._action_length)
        ## This does the sum already
        self._r = (self._prob / self._prob_target)
        self._actLoss_ = theano.tensor.elemwise.Elemwise(theano.scalar.mul)((self._r), self._Advantage)
        ppo_epsilon = self.getSettings()['kl_divergence_threshold']
        self._actLoss_2 = theano.tensor.elemwise.Elemwise(theano.scalar.mul)((theano.tensor.clip(self._r, 1.0 - ppo_epsilon, 1+ppo_epsilon), self._Advantage))
        self._actLoss_ = theano.tensor.minimum((self._actLoss_), (self._actLoss_2))
        # self._actLoss = ((T.mean(self._actLoss_) )) + -self._actor_regularization
        # self._actLoss = (-1.0 * (T.mean(self._actLoss_) + (self.getSettings()['std_entropy_weight'] * self._actor_entropy )))
        self._actLoss = -1.0 * (T.mean(self._actLoss_) + T.mean(self._QProp_N * self._q_function) )
        self._actLoss_PPO = -1.0 * (T.mean(self._actLoss_))  
        
        # self._policy_grad = T.grad(self._actLoss ,  self._actionParams)
        
        QPropKeras.compile(self)
        
    def compile(self):
        # sgd = SGD(lr=0.001, momentum=0.9)
        
        sgd = keras.optimizers.Adam(lr=self.getSettings()['critic_learning_rate'], beta_1=0.9, beta_2=0.999, epsilon=self._rms_epsilon, decay=0.0)
        print ("Clipping: ", sgd.decay)
        self._model.getCriticNetwork().compile(loss='mse', optimizer=sgd)
        
        sgd = keras.optimizers.Adam(lr=np.float32(self.getSettings()['critic_learning_rate']), beta_1=np.float32(0.9), beta_2=np.float32(0.999), epsilon=np.float32(self._rms_epsilon), decay=np.float32(0.0))
        print ("Clipping: ", sgd.decay)
        print("sgd, critic: ", sgd)
        self._model.getValueFunction().compile(loss='mse', optimizer=sgd)
        # sgd = SGD(lr=0.0005, momentum=0.9)
        
        def neg_y(true_y, pred_y):
            return -pred_y
        sgd = keras.optimizers.Adam(lr=np.float32(self.getSettings()['learning_rate']), beta_1=np.float32(0.9), beta_2=np.float32(0.999), epsilon=np.float32(self._rms_epsilon), decay=np.float32(0.0))
        print("sgd, actor: ", sgd)
        print ("Clipping: ", sgd.decay)
        self._model.getActorNetwork().compile(loss=neg_y, optimizer=sgd)
        
        self._trainPolicy = theano.function([self._model._stateInput,
                                             self._model._actionInput,
                                             self._Advantage,
                                             self._QProp_N], 
                                            [self._actLoss, self._r, self._q_function], 
                        updates= adam_updates(self._actLoss, self._model.getActorNetwork().trainable_weights, learning_rate=self._learning_rate).items())
        
        self._trainPolicy_PPO = theano.function([self._model._stateInput,
                                             self._model._actionInput,
                                             self._Advantage], 
                                            [self._actLoss_PPO, self._r], 
                        updates= adam_updates(self._actLoss_PPO, self._model.getActorNetwork().trainable_weights, learning_rate=self._learning_rate).items())
        ### DPG like updates
        updates= adam_updates(-T.mean(self._q_function), self._model.getActorNetwork().trainable_weights, learning_rate=self._learning_rate).items()
        self._trainPolicy_DPG = theano.function([self._model._stateInput], 
                                           [self._q_function], 
                                           updates= updates)
        
        self._r = theano.function([self._model._stateInput,
                                             self._model._actionInput], 
                                  [self._r])
        
        ### Compute gradient of state wrt value function
        weights = [self._model._actionInput]
        if ("use_extra_value_for_MBAE_state_grads" in self.getSettings() 
            and (self.getSettings()["use_extra_value_for_MBAE_state_grads"] == True)):
            gradients = K.gradients(T.mean(self._value), [self._model._stateInput]) # gradient tensors
            print( "********Using use_extra_value_for_MBAE_state_grads*******")
        else:
            gradients = K.gradients(T.mean(self._q_function), [self._model._stateInput]) # gradient tensors
            print( "********NOT Using use_extra_value_for_MBAE_state_grads*******")
        ### DPG related functions
        self._get_gradients = K.function(inputs=[self._model._stateInput,  K.learning_phase()], outputs=gradients)
        self._q_func = K.function([self._model._stateInput], [self._q_function])
        self._q_func_Target = K.function([self._model._stateInput, self._model._actionInput], [self._q_function_Target])
        self._value_Target = K.function([self._model._stateInput, K.learning_phase()], [self._value_Target])
        self._value = K.function([self._model._stateInput, K.learning_phase()], [self._value])
        ### PPO related functions
        self._q_action_std = K.function([self._model._stateInput], [self._q_valsActASTD])
        
    def getGrads(self, states, alreadyNormed=False):
        """
            The states should be normalized
        """
        # self.setData(states, actions, rewards, result_states)
        if ( alreadyNormed == False):
            states = norm_state(states, self._state_bounds)
        states = np.array(states, dtype=self._settings['float_type'])
        # grads = np.reshape(np.array(self._get_gradients([states])[0], dtype=self._settings['float_type']), (states.shape[0],states.shape[1]))
        grads = np.array(self._get_gradients([states, 0]), dtype=self._settings['float_type'])
        # print ("State grads: ", grads.shape)
        # print ("State grads: ", repr(grads))
        return grads
        
    def updateTargetModelValue(self):
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
            print ("Updating target Model")
        """
            Target model updates
        """
        self._modelTarget2.getValueFunction().set_weights( copy.deepcopy(self._model.getValueFunction().get_weights()))
        self._modelTarget2.getActorNetwork().set_weights( copy.deepcopy(self._model.getActorNetwork().get_weights()))
        
    def updateTargetModel(self):
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
            print ("Updating target Model")
        """
            Target model updates
        """
        # return
        ## I guess it is okay to lerp the entire network even though we only really want to 
        ## lerp the value function part of the networks, the target policy is not used for anythings
        all_paramsA = self._model.getCriticNetwork().get_weights()
        all_paramsB = self._modelTarget.getCriticNetwork().get_weights()
        if ('target_net_interp_weight' in self.getSettings()):
            lerp_weight = self.getSettings()['target_net_interp_weight']
        else:
            lerp_weight = 0.001
        # vals = lasagne.layers.helper.get_all_param_values(self._l_outActA)
        
        all_params = []
        for paramsA, paramsB in zip(all_paramsA, all_paramsB):
            # print ("paramsA: " + str(paramsA))
            # print ("paramsB: " + str(paramsB))
            params = (lerp_weight * paramsA) + ((1.0 - lerp_weight) * paramsB)
            all_params.append(params)
        self._modelTarget.getCriticNetwork().set_weights(all_params)
        
        all_paramsA_Act = self._model.getActorNetwork().get_weights()
        all_paramsB_Act = self._modelTarget.getActorNetwork().get_weights()
        
        all_params = []
        for paramsA, paramsB in zip(all_paramsA_Act, all_paramsB_Act):
            # print ("paramsA: " + str(paramsA))
            # print ("paramsB: " + str(paramsB))
            params = (lerp_weight * paramsA) + ((1.0 - lerp_weight) * paramsB)
            all_params.append(params)
        self._modelTarget.getActorNetwork().set_weights(all_params)

    
    def getNetworkParameters(self):
        params = []
        params.append(copy.deepcopy(self._model.getCriticNetwork().get_weights()))
        params.append(copy.deepcopy(self._model.getActorNetwork().get_weights()))
        params.append(copy.deepcopy(self._modelTarget.getCriticNetwork().get_weights()))
        params.append(copy.deepcopy(self._modelTarget.getActorNetwork().get_weights()))
        params.append(copy.deepcopy(self._model.getValueFunction().get_weights()))
        params.append(copy.deepcopy(self._modelTarget2.getValueFunction().get_weights()))
        params.append(copy.deepcopy(self._modelTarget2.getActorNetwork().get_weights()))
        return params
    
    def setNetworkParameters(self, params):
        """
        for i in range(len(params[0])):
            params[0][i] = np.array(params[0][i], dtype=self._settings['float_type'])
            """
        self._model.getCriticNetwork().set_weights(params[0])
        self._model.getActorNetwork().set_weights( params[1] )
        self._modelTarget.getCriticNetwork().set_weights( params[2])
        self._modelTarget.getActorNetwork().set_weights( params[3])
        self._model.getValueFunction().set_weights( params[4])
        self._modelTarget2.getValueFunction().set_weights( params[5])
        self._modelTarget2.getActorNetwork().set_weights( params[6])
    
    def setData(self, states, actions, rewards, result_states, fallen):
        pass
        # _targets = rewards + (self._discount_factor * self._q_valsTargetNextState )
        
    def trainOnPolicyCritic(self, states, actions, rewards, result_states, falls):
        # print ("Performing Critic trainning update")
        if (( self._updates % self._weight_update_steps) == 0):
            self.updateTargetModelValue()
        self._updates += 1
        ### Set learning phase to test time to skip dropout affects
        # K.set_learning_phase(0)
        # test_phase = np.zeros_like(rewards)
        y_ = self._value_Target([result_states,0])[0]
        # print ("y_: ", y_)
        # y_ = self._modelTarget2.getValueFunction().predict(result_states)
        # v = self._model.getValueFunction().predict(states, batch_size=states.shape[0])
        target_ = rewards + ((self._discount_factor * y_))
        target_ = np.array(target_, dtype=self._settings['float_type'])
        score = self._model.getValueFunction().fit(states, target_,
              nb_epoch=1, batch_size=states.shape[0],
              verbose=0
              # callbacks=[early_stopping],
              )
        loss = score.history['loss'][0]
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['debug']):
            print(" Value Function loss: ", loss)
        
        return loss
    
    def trainCritic(self, states, actions, rewards, result_states, falls):
        
        # self.setData(states, actions, rewards, result_states, falls)
        ## get actions for target policy
        target_actions = self._modelTarget.getActorNetwork().predict(states, batch_size=states.shape[0])
        ## Get next q value
        q_vals_b = self._modelTarget.getCriticNetwork().predict([states, target_actions], batch_size=states.shape[0])
        # q_vals_b = self._q_val()
        ## Compute target values
        # target_tmp_ = rewards + ((self._discount_factor* q_vals_b )* falls)
        target_tmp_ = rewards + ((self._discount_factor * q_vals_b ))
        # self.setData(states, actions, rewards, result_states, falls)
        # self._tmp_target_shared.set_value(target_tmp_)
        
        # self._target = T.mul(T.add(self._model.getRewardSymbolicVariable(), T.mul(self._discount_factor, self._q_valsB )), self._Fallen)
        
        loss = self._model.getCriticNetwork().fit([states, actions], target_tmp_,
                        batch_size=states.shape[0],
                        nb_epoch=1,
                        verbose=False,
                        shuffle=False)
        
        self.updateTargetModel()
        loss = loss.history['loss'][0]
        return loss
    
    def trainActor(self, states, actions, rewards, result_states, falls, advantage, exp_actions=None, forwardDynamicsModel=None):
        lossActor = 0
        
        if (( self._updates % self._weight_update_steps) == 0):
            self.updateTargetModelValue()
        self._updates += 1
        """
        score = self._model.getActorNetwork().fit([states, actions, advantage], np.zeros_like(rewards),
              nb_epoch=1, batch_size=32,
              verbose=0
              # callbacks=[early_stopping],
              )
        """
        train_DPG = False
        
        if ( (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train'] ) 
             and True 
             ):
            mbae_actions=[]
            mbae_advantage=[]
            other_actions=[]
            other_advantage=[]
            policy_mean = self._model.getActorNetwork().predict(states, batch_size=states.shape[0])[:,:self._action_length]
            # print ("exp_actions: ", exp_actions)
            for k in range(actions.shape[0]):
                if (exp_actions[k] == 2):
                    mbae_actions.append(actions[k]-policy_mean[k])
                    mbae_advantage.append(advantage[k])
                else:
                    other_actions.append(actions[k]-policy_mean[k])
                    other_advantage.append(advantage[k])
            
            
            policy_mean = self._model.getActorNetwork().predict(states, batch_size=states.shape[0])[:,:self._action_length]
            print ("MBAE Actions: ", len(mbae_actions), ", ", len(mbae_actions)/actions.shape[0], "%")
            print ("MBAE Actions std: ", np.std(mbae_actions, axis=0), " mean ", np.mean(np.std(mbae_actions, axis=0)))
            print ("MBAE Actions advantage: ", np.mean(mbae_advantage, axis=0))
            print ("Normal Actions std: ", np.std(other_actions, axis=0), " mean ", np.mean(np.std(other_actions, axis=0)))
            print ("Normal Actions advantage: ", np.mean(other_advantage, axis=0))        
            
        if ( train_DPG ) :
            q_ = np.mean(self._trainPolicy_DPG(states))
            if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['debug']):
                print ("Policy loss: ", q_)
            return
                
        r_ = np.mean(self._r(states, actions))
        
        
        ### From Q-prop paper, compute adaptive control variate.
        sampled_q = self._model.getCriticNetwork().predict([states, actions], batch_size=states.shape[0])
        # sampled_q = self._q_func_Target([states, actions])[0]
        sampled_q = scale_reward(sampled_q, self.getRewardBounds()) * (1.0 / (1.0- self.getSettings()['discount_factor']))
        true_q = self._q_func([states])[0]
        ## Scale q func to be in same space as advantage
        true_q = scale_reward(true_q, self.getRewardBounds()) * (1.0 / (1.0- self.getSettings()['discount_factor']))
        cov = advantage * (sampled_q - true_q)
        # var = true_q * true_q
        # n = cov / var
        ### practical implementation n = 1 when cov > 0, otherwise 0
        n = (np.sign(cov) + 1.0 ) / 2.0
        # n = np.zeros_like(n)
        advantage = (advantage - (n * (sampled_q - true_q)))
        
        std = np.std(advantage)
        mean = np.mean(advantage)
        if ( 'advantage_scaling' in self.getSettings() and ( self.getSettings()['advantage_scaling'] != False) ):
            std = std / self.getSettings()['advantage_scaling']
            mean = 0.0
        advantage = (advantage - mean) / std
        
        if (r_ < 2.0) and ( r_ > 0.5):  ### update not to large
            (lossActor, r_, q_) = self._trainPolicy(states, actions, advantage, n)
            # lossActor = score.history['loss'][0]
            if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['debug']):
                print ("Policy loss: ", lossActor, " r: ", np.mean(r_),  " q: ", np.mean(q_), )
                print ("Policy mean: ", np.mean(self._model.getActorNetwork().predict(states, batch_size=states.shape[0])[:,:self._action_length], axis=0))
                print ("Policy std: ", np.mean(self._q_action_std([states])[0], axis=0))
                print ("Gradient Info: n, mean:", np.mean(n), " std: ", np.std(n))
                print ("Gradient Info: cov, mean:", np.mean(cov), " std: ", np.std(cov))
        else:
            if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
                print ("Policy Gradient too large: ", np.mean(r_))
            
        self.updateTargetModel()
        
        return lossActor
    
    def train(self, states, actions, rewards, result_states, falls):
        loss = self.trainCritic(states, actions, rewards, result_states, falls)
        lossActor = self.trainActor(states, actions, rewards, result_states, falls)
        return loss
    
    def predict(self, state, deterministic_=True, evaluation_=False, p=None, sim_index=None, bootstrapping=False):
        state = norm_state(state, self._state_bounds)
        state = np.array(state, dtype=self._settings['float_type'])
        action_ = scale_action(self._model.getActorNetwork().predict(state, batch_size=1)[:,:self._action_length], self._action_bounds)
        return action_
    
    def predict_std(self, state, deterministic_=True):
        state = norm_state(state, self._state_bounds)   
        state = np.array(state, dtype=self._settings['float_type'])
        
        # action_std = self._model.getActorNetwork().predict(state, batch_size=1)[:,self._action_length:] * (action_bound_std(self._action_bounds))
        action_std = self._q_action_std([state])[0] * action_bound_std(self._action_bounds)
        # print ("Policy std: ", repr(action_std))
        return action_std
    
    def predictWithDropout(self, state, deterministic_=True):
        # states = np.zeros((self._batch_size, self._state_length), dtype=self._settings['float_type'])
        # states[0, ...] = state
        state = np.array(state, dtype=self._settings['float_type'])
        state = norm_state(state, self._state_bounds)
        action_ = scale_action(self._model.getActorNetwork().predict(states, batch_size=1)[:,:self._action_length], self._action_bounds)
        # else:
        # action_ = scale_action(self._q_action()[0], self._action_bounds)
        # action_ = q_valsActA[0]
        return action_
    
    def q_value(self, state):
        state = norm_state(state, self._state_bounds)
        state = np.array(state, dtype=self._settings['float_type'])
        value = scale_reward(self._value([state,0])[0], self.getRewardBounds()) * (1.0 / (1.0- self.getSettings()['discount_factor']))
        return value
    
    def q_values(self, state):
        """
            For returning a vector of q values, state should already be normalized
        """
        state = np.array(state, dtype=self._settings['float_type'])
        return self._value([state,0])[0]
    
    def q_valueWithDropout(self, state):
        # states = np.zeros((self._batch_size, self._state_length), dtype=self._settings['float_type'])
        # states[0, ...] = state
        state = np.array(state, dtype=self._settings['float_type'])
        state = norm_state(state, self._state_bounds)
        self._model.setStates(state)
        return scale_reward(self._q_val_drop(), self.getRewardBounds())
    
    def bellman_error(self, states, actions, rewards, result_states, falls):
        """
            Computes the one step temporal difference.
        """
        y_ = self._value_Target([result_states,0])[0]
        # y_ = self._modelTarget2.getValueFunction().predict(result_states, batch_size=states.shape[0])
        target_ = rewards + ((self._discount_factor * y_))
        # values =  self._model.getValueFunction().predict(states, batch_size=states.shape[0])
        values = self._value([states,0])[0]
        bellman_error = target_ - values
        return bellman_error
        # return self._bellman_errorTarget()
        
    def trainDyna(self, predicted_states, actions, rewards, result_states, falls):
        """
            Performs a DYNA type update
            Because I am using target networks a direct DYNA update does nothing. 
            The gradients are not calculated for the target network.
            L(\theta) = (r + V(s'|\theta')) - V(s|\theta))
            Instead what is done is this
            L(\theta) = V(s'|\theta')) - V(\hat{s'}|\theta))
            Where s' comes from the simulation and \hat{s'} is a predicted ( \hat{s'} <- fd(a,s) ) and noisy value from an fd model
            Parameters
            ----------
            predicted_states : predicted states, s_1
            actions : list of actions
            
            result_states : simulated states, s_2
            
            falls: list of flags for whether or not the character fell
            Returns
            -------
            loss: the loss for the DYNA type update

        """
        # self.setData( result_states, actions, rewards, predicted_states, falls)
        # values = self._modelTarget2.getValueFunction().predict(result_states, batch_size=result_states.shape[0])
        values = self._value_Target([result_states,0])[0]
        # values = self._val_TargetState()
        # print ("Dyna values: ", values)
        score = self._model.getValueFunction().fit(predicted_states, values,
              nb_epoch=1, batch_size=result_states.shape[0],
              verbose=0
              # callbacks=[early_stopping],
              )
        loss = score.history['loss'][0]
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['debug']):
            print("**************Dyna Function loss: ", loss)
            
        return loss
        
from collections import OrderedDict
def adam_updates(loss, params, learning_rate=0.001, beta1=0.9,
         beta2=0.999, epsilon=1e-8):

    all_grads = T.grad(loss, params)
    t_prev = theano.shared(np.array(0,dtype="float64"))
    updates = OrderedDict()

    t = t_prev + 1
    a_t = learning_rate*T.sqrt(1-beta2**t)/(1-beta1**t)

    for param, g_t in zip(params, all_grads):
        value = param.get_value(borrow=True)
        m_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                               broadcastable=param.broadcastable)
        v_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                               broadcastable=param.broadcastable)

        m_t = beta1*m_prev + (1-beta1)*g_t
        v_t = beta2*v_prev + (1-beta2)*g_t**2
        step = a_t*m_t/(T.sqrt(v_t) + epsilon)

        updates[m_prev] = m_t
        updates[v_prev] = v_t
        updates[param] = param - step

    updates[t_prev] = t
    return updates