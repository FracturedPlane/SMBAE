import theano
from theano import tensor as T
from lasagne.layers import get_all_params
import numpy as np
# import lasagne
import sys
import copy
sys.path.append('../')
from model.ModelUtil import *
from algorithm.AlgorithmInterface import AlgorithmInterface
from model.LearningUtil import loglikelihood, kl, entropy, change_penalty
from keras.optimizers import SGD
# from keras.utils.np_utils import to_categoricalnetwork
import keras.backend as K
import keras


# For debugging
# theano.config.mode='FAST_COMPILE'
from collections import OrderedDict

class DPGKeras(AlgorithmInterface):
    
    def __init__(self, model, n_in, n_out, state_bounds, action_bounds, reward_bound, settings_):
        """
            In order to get this to work we need to be careful not to update the actor parameters
            when updating the critic. This can be an issue when the Concatenating networks together.
            The first first network becomes a part of the second. However you can still access the first
            network by itself but an updates on the second network will effect the first network.
            Care needs to be taken to make sure only the parameters of the second network are updated.
        """
        
        super(DPGKeras,self).__init__( model, n_in, n_out, state_bounds, action_bounds, reward_bound, settings_)

        self._modelTarget = copy.deepcopy(model)
        
        
        self._discount_factor= self.getSettings()['discount_factor']
        self._rho = self.getSettings()['rho']
        self._rms_epsilon = self.getSettings()['rms_epsilon']
        
        self._q_valsActA = self._model.getActorNetwork()(self._model._stateInput)[:,:self._action_length]
        # self._q_valsActASTD = self._model.getActorNetwork()(self._model.getStateSymbolicVariable())[:,self._action_length:]
        
        self._q_valsActTarget_State = self._modelTarget.getActorNetwork()(self._model._stateInput)[:,:self._action_length]
        # self._q_valsActTargetSTD = self._modelTarget.getActorNetwork()(self._model.getStateSymbolicVariable())[:,self._action_length:]
        
        # self._q_function = self._model.getCriticNetwork()(self._model.getStateSymbolicVariable(), self._q_valsActA)
        self._q_function = self._model.getCriticNetwork()([self._model._stateInput, self._q_valsActA])
        self._q_function_Target = self._model.getCriticNetwork()([self._model._stateInput, self._model._actionInput])
        
        
        # print ("Initial W " + str(self._w_o.get_value()) )
        
            ## TD update
        DPGKeras.compile(self)
        
    def compile(self):
        
        """
            self._act_target = self._modelTarget().getActorNetwork()
            self._q_val_target = self._modelTarget().getCriticNetwork()
            
            self._q_vals_b = self._q_val_target(self._act_target(self._states))
            
            self._q_val = self._modelTarget().getCriticNetwork()(self._model().getActorNetwork()(self._states))
            self._y_target = rewards + ((self._discount_factor * q_vals_b ))
        """
        
        sgd = keras.optimizers.Adam(lr=self.getSettings()['critic_learning_rate'], beta_1=0.9, beta_2=0.999, epsilon=self._rms_epsilon, decay=0.0)
        print ("Clipping: ", sgd.decay)
        self._model.getCriticNetwork().compile(loss='mse', optimizer=sgd)
        
        # self._actor_optimizer = keras.optimizers.Adam(lr=self.getSettings()['critic_learning_rate'], beta_1=0.9, beta_2=0.999, epsilon=self._rms_epsilon, decay=0.0)
        # updates = self._actor_optimizer.get_updates(self._model.getActorNetwork().trainable_weights, loss=-T.mean(self._q_function), constraints=[])
        updates= adam_updates(-T.mean(self._q_function), self._model.getActorNetwork().trainable_weights, learning_rate=self._learning_rate).items()
        self._trainPolicy = theano.function([self._model._stateInput], 
                                           [self._q_function], 
                                           updates= updates)
        
        
        weights = [self._model._actionInput]
        
        gradients = K.gradients(T.mean(self._q_function), [self._model._stateInput]) # gradient tensors

        self._get_gradients = K.function(inputs=[self._model._stateInput], outputs=gradients)
        
        self._q_func = K.function([self._model._stateInput], [self._q_function])
        
        # updates = self._actor_optimizer.get_updates(self._model.getActorNetwork().trainable_weights, loss=gradients, constraints=[])
        ### Train the Q network, just uses MSE
        # self._train_q = Model(inputs=[self._states, self._targets])
        # self._model.getCriticNetwork().compile(loss='mse',
        #                          optimizer=Adam(lr=2.0e-4, beta_1=0.5))
        # self._train_q.summary()
        
        
        # self._train_policy = K.function([self._states, ytrue],[loss, accuracy],updates=updates)
        ### Compute and return the target q value
        # self._q_target = K.function([self._states, self._rewards],[self._y_target])
        # self._q_action = K.function([self._states, self._rewards],[self._y_target])
        ### Train the policy network
        # self._model.getActorNetwork().compile(loss=neg_y, optimizer=self._actor_optimizer)
        
        
        
    def getGrads(self, states, actions=None, alreadyNormed=False):
        """
            The states should be normalized
        """
        # self.setData(states, actions, rewards, result_states)
        if ( alreadyNormed == False):
            states = norm_state(states, self._state_bounds)
        states = np.array(states, dtype=theano.config.floatX)
        grads = self._get_gradients([states])
        return grads
    
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
        params.append(copy.deepcopy(self._model.getCriticNetwork().get_weights()))
        params.append(copy.deepcopy(self._model.getActorNetwork().get_weights()))
        params.append(copy.deepcopy(self._modelTarget.getCriticNetwork().get_weights()))
        params.append(copy.deepcopy(self._modelTarget.getActorNetwork().get_weights()))
        return params
    
    def setNetworkParameters(self, params):
        """
        for i in range(len(params[0])):
            params[0][i] = np.array(params[0][i], dtype=theano.config.floatX)
            """
        self._model.getCriticNetwork().set_weights(params[0])
        self._model.getActorNetwork().set_weights( params[1] )
        self._modelTarget.getCriticNetwork().set_weights( params[2])
        self._modelTarget.getActorNetwork().set_weights( params[3])    

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
        # self._fallen_shared.set_value(fallen)
        # diff_ = self.bellman_error(states, actions, rewards, result_states, falls)
        ## Easy fix for computing actor loss
        # diff = self._bellman_error2()
        # self._tmp_diff_shared.set_value(diff)
        
        # _targets = rewards + (self._discount_factor * self._q_valsTargetNextState )
        
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
        
    def trainActor(self, states, actions, rewards, result_states, falls, advantage, exp_actions, forwardDynamicsModel=None):
        # self.setData(states, actions, rewards, result_states, falls)
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['debug']):
            print("values: ", np.mean(self._q_val()* (1.0 / (1.0- self.getSettings()['discount_factor']))), " std: ", np.std(self._q_val()* (1.0 / (1.0- self.getSettings()['discount_factor']))) )
            print("Rewards: ", np.mean(rewards), " std: ", np.std(rewards), " shape: ", np.array(rewards).shape)
        # print("Policy mean: ", np.mean(self._q_action(), axis=0))
        loss = 0
        # loss = self._trainActor()
        
        q_fun = np.mean(self._trainPolicy(states))
        
        if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['debug']):
            # print("Actions mean:     ", np.mean(actions, axis=0))
            poli_mean = self._model.getActorNetwork().predict(states, batch_size=states.shape[0])
            print("Policy mean: ", np.mean(poli_mean, axis=0))
            # print("Actions std:  ", np.mean(np.sqrt( (np.square(np.abs(actions - np.mean(actions, axis=0))))/1.0), axis=0) )
            # print("Actions std:  ", np.std((actions - self._q_action()), axis=0) )
            # print("Actions std:  ", np.std((actions), axis=0) )
            # print("Policy std: ", np.mean(self._q_action_std(), axis=0))
            # print("Mean Next State Grad grad: ", np.mean(next_state_grads, axis=0), " std ", np.std(next_state_grads, axis=0))
            # print("Mean ation grad: ", np.mean(action_grads, axis=0), " std ", np.std(action_grads, axis=0))
            print ("Actor Loss: ", q_fun)
            
        return q_fun
        
    def train(self, states, actions, rewards, result_states):
        loss = self.trainCritic(states, actions, rewards, result_states)
        lossActor = self.trainActor(states, actions, rewards, result_states)
        return loss
    
    def predict(self, state, deterministic_=True, evaluation_=False, p=None, sim_index=None, bootstrapping=False):
        # states = np.zeros((self._batch_size, self._state_length), dtype=self._settings['float_type'])
        # states[0, ...] = state
        # state = np.array(state, dtype=self._settings['float_type'])
        state = norm_state(state, self._state_bounds)
        state = np.array(state, dtype=self._settings['float_type'])
        self._model.setStates(state)
        # action_ = lasagne.layers.get_output(self._model.getActorNetwork(), state, deterministic=deterministic_).mean()
        # action_ = scale_action(self._q_action()[0], self._action_bounds)
        # if deterministic_:
        action_ = scale_action(self._model.getActorNetwork().predict(state, batch_size=1), self._action_bounds)
        # action_ = scale_action(self._q_action_target()[0], self._action_bounds)
        # else:
        # action_ = scale_action(self._q_action()[0], self._action_bounds)
        # action_ = q_valsActA[0]
        return action_
    
    def q_value(self, state):
        # states = np.zeros((self._batch_size, self._state_length), dtype=self._settings['float_type'])
        # states[0, ...] = state
        state = norm_state(state, self._state_bounds)
        state = np.array(state, dtype=self._settings['float_type'])
        # return scale_reward(self._q_valTarget(), self.getRewardBounds())[0]
        poli_mean = self._model.getActorNetwork().predict(state, batch_size=1)
        value = scale_reward(self._model.getCriticNetwork().predict([state, poli_mean] , batch_size=1), self.getRewardBounds()) * (1.0 / (1.0- self.getSettings()['discount_factor']))
        # value = scale_reward(self._q_func(state), self.getRewardBounds()) * (1.0 / (1.0- self.getSettings()['discount_factor']))
        return value
        # return self._q_val()[0]
    
    def q_values(self, state):
        """
            For returning a vector of q values, state should already be normalized
        """
        state = np.array(state, dtype=self._settings['float_type'])
        # print ("Getting q_values: ", state)
        poli_mean = self._model.getActorNetwork().predict(state, batch_size=state.shape[0])
        value = self._model.getCriticNetwork().predict([state, poli_mean] , batch_size=state.shape[0])
        # print ("q_values: ", value)
        return value

    def bellman_error(self, states, actions, rewards, result_states, falls):
        """
            Computes the one step temporal difference.
        """
        y_ = self._modelTarget.getCriticNetwork().predict([result_states, actions], batch_size=states.shape[0])
        target_ = rewards + ((self._discount_factor * y_))
        poli_mean = self._model.getActorNetwork().predict(states, batch_size=states.shape[0])
        values =  self._model.getCriticNetwork().predict([states, poli_mean], batch_size=states.shape[0])
        bellman_error = target_ - values
        return bellman_error
        # return self._bellman_errorTarget()        
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