import theano
from theano import tensor as T
import numpy as np
import lasagne
import sys
from itertools import chain
sys.path.append('../')
import copy
# from ModelUtil import *
from model.ModelUtil import norm_state, scale_state, norm_action, scale_action, action_bound_std, scale_reward
from model.LearningUtil import loglikelihood, likelihood, kl, entropy, flatgrad, zipsame, get_params_flat, setFromFlat

# For debugging
# theano.config.mode='FAST_COMPILE'
# from DeepCACLA import DeepCACLA

class AlgorithmInterface(object):
    
    def __init__(self, model, n_in, n_out, state_bounds, action_bounds, reward_bound, settings_):

        ## primary network, don't make a copy of this
        self._model = model
        
        self._batch_size=settings_['batch_size']
        self._state_length = n_in
        self._action_length = n_out
        self.setSettings(settings_)
        
        AlgorithmInterface.setActionBounds(self, action_bounds) 
        AlgorithmInterface.setStateBounds(self, state_bounds) 
        AlgorithmInterface.setRewardBounds(self, reward_bound) 
        
        # data types for model
        
        """
        State = T.dmatrix("State")
        State.tag.test_value = np.random.rand(batch_size,self._state_length)
        ResultState = T.dmatrix("ResultState")
        ResultState.tag.test_value = np.random.rand(batch_size,self._state_length)
        Reward = T.col("Reward")
        Reward.tag.test_value = np.random.rand(batch_size,1)
        Action = T.dmatrix("Action")
        Action.tag.test_value = np.random.rand(batch_size, self._action_length)
        """
        # create a small convolutional neural network
        
        self._learning_rate = self.getSettings()['learning_rate']
        self._discount_factor= self.getSettings()['discount_factor']
        self._rho = self.getSettings()['rho']
        self._rms_epsilon = self.getSettings()['rms_epsilon']
        
        self._weight_update_steps= self.getSettings()['steps_until_target_network_update']
        self._regularization_weight= self.getSettings()['regularization_weight']
        self._updates=0
        
    def setActor(self, actor):
        self._actor = actor
    def setEnvironment(self, sim):
        self._sim = sim # The real simulator that is used for predictions
        
    def compile(self):
        """
            Compiles the functions for this algorithm
        """
        pass
    
    def numUpdates(self):
        return self._updates
        
    def updateTargetModel(self):
        pass

    def getNetworkParameters(self):
        params = []
        params.append(lasagne.layers.helper.get_all_param_values(self._model.getCriticNetwork()))
        params.append(lasagne.layers.helper.get_all_param_values(self._model.getActorNetwork()))
        params.append(lasagne.layers.helper.get_all_param_values(self._modelTarget.getCriticNetwork()))
        params.append(lasagne.layers.helper.get_all_param_values(self._modelTarget.getActorNetwork()))
        return params
    
    def setNetworkParameters(self, params):
        """
        for i in range(len(params[0])):
            params[0][i] = np.array(params[0][i], dtype=self._settings['float_type'])
            """
        lasagne.layers.helper.set_all_param_values(self._model.getCriticNetwork(), params[0])
        lasagne.layers.helper.set_all_param_values(self._model.getActorNetwork(), params[1])
        lasagne.layers.helper.set_all_param_values(self._modelTarget.getCriticNetwork(), params[2])
        lasagne.layers.helper.set_all_param_values(self._modelTarget.getActorNetwork(), params[3])
        
    def setTaskNetworkParameters(self, otherModel):
        all_paramsA = lasagne.layers.helper.get_all_param_values(otherModel.getModel().getCriticNetworkTaskPart())
        all_paramsActA = lasagne.layers.helper.get_all_param_values(otherModel.getModel().getActorNetworkTaskPart())
        lasagne.layers.helper.set_all_param_values(self._model.getCriticNetworkTaskPart(), all_paramsA)
        lasagne.layers.helper.set_all_param_values(self._model.getActorNetworkTaskPart(), all_paramsActA)
        ### Targetnets
        all_paramsA = lasagne.layers.helper.get_all_param_values(otherModel.getModelTarget().getCriticNetworkTaskPart())
        all_paramsActA = lasagne.layers.helper.get_all_param_values(otherModel.getModelTarget().getActorNetworkTaskPart())
        lasagne.layers.helper.set_all_param_values(self.getModelTarget().getCriticNetworkTaskPart(), all_paramsA)
        lasagne.layers.helper.set_all_param_values(self.getModelTarget().getActorNetworkTaskPart(), all_paramsActA)

    def setAgentNetworkParamters(self, otherModel):
        all_paramsA = lasagne.layers.helper.get_all_param_values(otherModel.getModel().getCriticNetworkAgentPart())
        all_paramsActA = lasagne.layers.helper.get_all_param_values(otherModel.getModel().getActorNetworkAgentPart())
        lasagne.layers.helper.set_all_param_values(self._model.getCriticNetworkAgentPart(), all_paramsA)
        lasagne.layers.helper.set_all_param_values(self._model.getActorNetworkAgentPart(), all_paramsActA)
        ## targetNets
        all_paramsA = lasagne.layers.helper.get_all_param_values(otherModel.getModelTarget().getCriticNetworkAgentPart())
        all_paramsActA = lasagne.layers.helper.get_all_param_values(otherModel.getModelTarget().getActorNetworkAgentPart())
        lasagne.layers.helper.set_all_param_values(self.getModelTarget().getCriticNetworkAgentPart(), all_paramsA)
        lasagne.layers.helper.set_all_param_values(self.getModelTarget().getActorNetworkAgentPart(), all_paramsActA)
        
    def copyLayerParameters(self, layers, otherLayers):
        """
            Copies the values from one list of layers to another
        """
        params = chain.from_iterable(l.get_params() for l in layers)
        paramsOther = chain.from_iterable(l.get_params() for l in otherLayers)
        for p, v in zip(params, paramsOther):
            print ("params: ", p.get_value().shape, " vs ", 
                   v.get_value().shape )
            if (p.get_value().shape != v.get_value().shape):
                print ("Shape mismatch")
                raise ValueError("mismatch: parameter has shape %r but value to "
                             "set has shape %r" %
                            (p.get_value().shape, v.get_value().shape))
            p.set_value(v.get_value())
            
        
    def setCombinedNetworkParamters(self, otherModel):
        self.copyLayerParameters(self.getModel().getCriticNetworkCombinedPart(), 
            otherModel.getModel().getCriticNetworkCombinedPart())
        self.copyLayerParameters(self.getModel().getActorNetworkCombinedPart(), 
            otherModel.getModel().getActorNetworkCombinedPart())
        ### Target nets
        self.copyLayerParameters(self.getModelTarget().getCriticNetworkCombinedPart(), 
            otherModel.getModelTarget().getCriticNetworkCombinedPart())
        self.copyLayerParameters(self.getModelTarget().getActorNetworkCombinedPart(), 
            otherModel.getModelTarget().getActorNetworkCombinedPart())
                    
    def setMergeLayerNetworkParamters(self, otherModel, zeroInjectedMergeLayer=False):
        """
            This method merges the two layers of the networks together that may have
            different sizes. THere is also an option method to zero all of the
            parameters of the part of the network that inserts from the
            task part of the network
            
            Only works if there is just one merge layer
        """
        zero_dense=False
        other_Layers = otherModel.getModel().getCriticNetworkMergeLayers()
        self_Layers = self.getModel().getCriticNetworkMergeLayers()
        
        print (" merge params: ", len(other_Layers))
        for i_ in range(len(other_Layers)):
            print ("merge params ", i_, ": ", other_Layers[i_].W.get_value().shape)
        # print ("params: ", all_paramsA)
        print (self_Layers)
        if (self_Layers[0].W.get_value().shape == other_Layers[0].W.get_value().shape ):
            print("Matching merge layer shapes ")
            print (self_Layers[0].W.get_value().shape, " vs ", other_Layers[0].W.get_value().shape)
        else: ## Two network were of different same size
            print("Merge layer shapes do not match ")
            print (self_Layers[0].W.get_value().shape, " vs ", other_Layers[0].W.get_value().shape)
            if ( zeroInjectedMergeLayer ):
                print ("Zeroing injected merge part")
                # values = np.zeros(self_Layers[0].W.get_value().shape, dtype=self.getSettings()['float_type'])
                ### Similar to what is done in DDPG to get i
                values = self_Layers[0].W.get_value() * 0.01
                ### also zero the dense network from the task part
                """
                if (zero_dense):
                    l = self.getModel().getCriticNetworkTaskPart()
                    values_2 = np.zeros(l.W.get_value().shape, dtype=self.getSettings()['float_type'])
                    l.W.set_value(values_2)
                """
            else:
                values = self_Layers[0].W.get_value()
            ### copy over other values
            other_shape = other_Layers[0].W.get_value().shape
            self_shape = self_Layers[0].W.get_value().shape
            ### Needs to be put on the end of the matrix
            values[self_shape[0]-other_shape[0]:self_shape[0],
                   self_shape[1]-other_shape[1]:self_shape[1]] = other_Layers[0].W.get_value()
            ### Update current net params
            other_Layers[0].W.set_value(values)
        # lasagne.layers.helper.set_all_param_values(self_Layers, all_paramsM)
        self.copyLayerParameters(self_Layers, other_Layers)
        
        other_Layers = otherModel.getModelTarget().getCriticNetworkMergeLayers()
        self_Layers = self.getModelTarget().getCriticNetworkMergeLayers()
        
        print (" merge params: ", len(other_Layers))
        for i_ in range(len(other_Layers)):
            print ("merge params ", i_, ": ", other_Layers[i_].W.get_value().shape)
        # print ("params: ", all_paramsA)
        print (self_Layers)
        if (self_Layers[0].W.get_value().shape == other_Layers[0].W.get_value().shape ):
            print("Matching merge layer shapes ")
            print (self_Layers[0].W.get_value().shape, " vs ", other_Layers[0].W.get_value().shape)
        else:
            print("Merge layer shapes do not match ")
            print (self_Layers[0].W.get_value().shape, " vs ", other_Layers[0].W.get_value().shape)
            if ( zeroInjectedMergeLayer ):
                print ("Zeroing injected merge part")
                # values = np.zeros(self_Layers[0].W.get_value().shape, dtype=self.getSettings()['float_type'])
                ### Similar to what is done in DDPG to get initial policy to be close to the mean.
                values = self_Layers[0].W.get_value() * 0.01
                ### also zero the dense network from the task part
                """
                if (zero_dense):
                    l = self.getModel().getCriticNetworkTaskPart()
                    values_2 = np.zeros(l.W.get_value().shape, dtype=self.getSettings()['float_type'])
                    l.W.set_value(values_2)
                """
            else:
                values = self_Layers[0].W.get_value()
            ### copy over other values
            other_shape = other_Layers[0].W.get_value().shape
            self_shape = self_Layers[0].W.get_value().shape
            ### Needs to be put on the end of the matrix
            values[self_shape[0]-other_shape[0]:self_shape[0],
                   self_shape[1]-other_shape[1]:self_shape[1]] = other_Layers[0].W.get_value()
            ### Update current net params
            other_Layers[0].W.set_value(values)
        # lasagne.layers.helper.set_all_param_values(self_Layers, all_paramsM)
        self.copyLayerParameters(self_Layers, other_Layers)
        
        ### Now for the possibly different shaped actor
        other_Layers = otherModel.getModel().getActorNetworkMergeLayers()
        self_Layers = self.getModel().getActorNetworkMergeLayers()
        print ("Actor merge params: ", len(other_Layers))
        for i_ in range(len(other_Layers)):
            print ("Actor merge params ", i_, ": ", other_Layers[i_].W.get_value().shape)
        # print ("params: ", all_paramsA)
        print (self_Layers)
        if (self_Layers[0].W.get_value().shape == other_Layers[0].W.get_value().shape ):
            print("Matching merge layer shapes ")
            print (self_Layers[0].W.get_value().shape, " vs ", other_Layers[0].W.get_value().shape)
        else:
            print("Merge layer shapes do not match ")
            print (self_Layers[0].W.get_value().shape, " vs ", other_Layers[0].W.get_value().shape)
            if ( zeroInjectedMergeLayer ):
                print ("Zeroing injected merge part")
                # values = np.zeros(self_Layers[0].W.get_value().shape, dtype=self.getSettings()['float_type'])
                ### Similar to what is done in DDPG to get initial policy to be close to the mean.
                values = self_Layers[0].W.get_value() * 0.01
                ### also zero the dense network from the task part
                """
                if (zero_dense):
                    l = self.getModel().getActorNetworkTaskPart()
                    values_2 = np.zeros(l.W.get_value().shape, dtype=self.getSettings()['float_type'])
                    l.W.set_value(values_2)
                """
            else:
                values = self_Layers[0].W.get_value()
            ### copy over other values
            other_shape = other_Layers[0].W.get_value().shape
            self_shape = self_Layers[0].W.get_value().shape
            ### Needs to be put on the end of the matrix
            values[self_shape[0]-other_shape[0]:self_shape[0],
                   self_shape[1]-other_shape[1]:self_shape[1]] = other_Layers[0].W.get_value()
            ### Update current net params
            other_Layers[0].W.set_value(values)
                
        # lasagne.layers.helper.set_all_param_values(self_Layers, all_paramsM)
        self.copyLayerParameters(self_Layers, other_Layers)
        
        ### Now for the possibly different shaped actor
        other_Layers = otherModel.getModelTarget().getActorNetworkMergeLayers()
        self_Layers = self.getModelTarget().getActorNetworkMergeLayers()
        print ("Actor merge params: ", len(other_Layers))
        for i_ in range(len(other_Layers)):
            print ("Actor merge params ", i_, ": ", other_Layers[i_].W.get_value().shape)
        # print ("params: ", all_paramsA)
        print (self_Layers)
        if (self_Layers[0].W.get_value().shape == other_Layers[0].W.get_value().shape ):
            print("Matching merge layer shapes ")
            print (self_Layers[0].W.get_value().shape, " vs ", other_Layers[0].W.get_value().shape)
        else:
            print("Merge layer shapes do not match ")
            print (self_Layers[0].W.get_value().shape, " vs ", other_Layers[0].W.get_value().shape)
            if ( zeroInjectedMergeLayer ):
                print ("Zeroing injected merge part")
                # values = np.zeros(self_Layers[0].W.get_value().shape, dtype=self.getSettings()['float_type'])
                ### Similar to what is done in DDPG to get initial policy to be close to the mean.
                values = self_Layers[0].W.get_value() * 0.01
                ### also zero the dense network from the task part
                """
                if (zero_dense):
                    l = self.getModel().getActorNetworkTaskPart()
                    values_2 = np.zeros(l.W.get_value().shape, dtype=self.getSettings()['float_type'])
                    l.W.set_value(values_2)
                """
            else:
                values = self_Layers[0].W.get_value()
            ### copy over other values
            other_shape = other_Layers[0].W.get_value().shape
            self_shape = self_Layers[0].W.get_value().shape
            ### Needs to be put on the end of the matrix
            values[self_shape[0]-other_shape[0]:self_shape[0],
                   self_shape[1]-other_shape[1]:self_shape[1]] = other_Layers[0].W.get_value()
            ### Update current net params
            other_Layers[0].W.set_value(values)
                
        # lasagne.layers.helper.set_all_param_values(self_Layers, all_paramsM)
        self.copyLayerParameters(self_Layers, other_Layers)
            
    def getModel(self):
        return self._model

    def getModelTarget(self):
        return self._modelTarget
        
    def getGrads(self, states, alreadyNormed=False):
        """
            The states should be normalized
        """
        # self.setData(states, actions, rewards, result_states)
        if ( alreadyNormed == False):
            states = norm_state(states, self._state_bounds)
        states = np.array(states, dtype=self._settings['float_type'])
        self._model.setStates(states)
        return self._get_grad()
    
    def getAdvantageGrads(self, states, next_states, alreadyNormed=False):
        """
            The states should be normalized
        """
        # self.setData(states, actions, rewards, result_states)
        if ( alreadyNormed == False):
            states = norm_state(states, self._state_bounds)
            next_states = norm_state(next_states, self._state_bounds)
        states = np.array(states, dtype=self._settings['float_type'])
        self._model.setStates(states)
        self._model.setResultStates(next_states)
        return self._get_grad()
    
    
    def predict(self, state, deterministic_=True, evaluation_=False, p=None, sim_index=None, bootstrapping=False):
        # states = np.zeros((self._batch_size, self._state_length), dtype=self._settings['float_type'])
        # states[0, ...] = state
        """
        if ( ('disable_parameter_scaling' in self._settings) and (self._settings['disable_parameter_scaling'])):
            pass
        else:
        """
        # print ("Agent state bounds: ", self._state_bounds)
        state = norm_state(state, self._state_bounds)
        # print ("Agent normalized state: ", state)
        state = np.array(state, dtype=self._settings['float_type'])
        self._model.setStates(state)
        # action_ = lasagne.layers.get_output(self._model.getActorNetwork(), state, deterministic=deterministic_).mean()
        # action_ = scale_action(self._q_action()[0], self._action_bounds)
        # if deterministic_:
        if ( ('disable_parameter_scaling' in self._settings) and (self._settings['disable_parameter_scaling'])):
            action_ = self._q_action()
            # action_ = scale_action(self._q_action()[0], self._action_bounds)
        else:
            action_ = scale_action(self._q_action(), self._action_bounds)
        # print ("Agent Scaled action: ", action_)
        # action_ = scale_action(self._q_action_target()[0], self._action_bounds)
        # else:
        # action_ = scale_action(self._q_action()[0], self._action_bounds)
        # action_ = q_valsActA[0]
        return action_
    
    def predict_batch(self, states, deterministic_=True):
        """
            These input and output do not need to be normalized/scalled
        """
        # state = norm_state(state, self._state_bounds)
        states = np.array(states, dtype=self._settings['float_type'])
        self._model.setStates(states)
        actions_ = self._q_action()
        return actions_
    
    def predict_std(self, state, deterministic_=True):
        # states = np.zeros((self._batch_size, self._state_length), dtype=self._settings['float_type'])
        # states[0, ...] = state
        """
        if ( ('disable_parameter_scaling' in self._settings) and (self._settings['disable_parameter_scaling'])):
            pass
        else:
        """
        state = norm_state(state, self._state_bounds)   
        state = np.array(state, dtype=self._settings['float_type'])
        self._model.setStates(state)
        # action_ = lasagne.layers.get_output(self._model.getActorNetwork(), state, deterministic=deterministic_).mean()
        # action_ = scale_action(self._q_action()[0], self._action_bounds)
        # if deterministic_:
        # action_std = scale_action(self._q_action_std()[0], self._action_bounds)
        if ( ('disable_parameter_scaling' in self._settings) and (self._settings['disable_parameter_scaling'])):
            action_std = self._q_action_std()
            # action_std = self._q_action_std()[0] * (action_bound_std(self._action_bounds))
        else:
            action_std = self._q_action_std() * (action_bound_std(self._action_bounds))
        # else:
        # action_ = scale_action(self._q_action()[0], self._action_bounds)
        # action_ = q_valsActA[0]
        return action_std
    
    def predictWithDropout(self, state, deterministic_=True):
        # states = np.zeros((self._batch_size, self._state_length), dtype=self._settings['float_type'])
        # states[0, ...] = state
        """
        if ( ('disable_parameter_scaling' in self._settings) and (self._settings['disable_parameter_scaling'])):
            pass
        else:
        """
        state = np.array(state, dtype=self._settings['float_type'])
        state = norm_state(state, self._state_bounds)
        self._model.setStates(state)
        # action_ = lasagne.layers.get_output(self._model.getActorNetwork(), state, deterministic=deterministic_).mean()
        # action_ = scale_action(self._q_action()[0], self._action_bounds)
        # if deterministic_:
        action_ = scale_action(self._q_action_drop(), self._action_bounds)
        # else:
        # action_ = scale_action(self._q_action()[0], self._action_bounds)
        # action_ = q_valsActA[0]
        return action_
    
    def q_value(self, state):
        """
            For returning a vector of q values, state should NOT be normalized
        """
        # states = np.zeros((self._batch_size, self._state_length), dtype=self._settings['float_type'])
        # states[0, ...] = state
        """
        if ( ('disable_parameter_scaling' in self._settings) and (self._settings['disable_parameter_scaling'])):
            pass
        else:
        """
        # print ("Agent state bounds: ", self._state_bounds)
        state = norm_state(state, self._state_bounds)
        # print ("Agent normalized state: ", state)
        state = np.array(state, dtype=self._settings['float_type'])
        self._model.setStates(state)
        self._modelTarget.setStates(state)
        if ( ('disable_parameter_scaling' in self._settings) and (self._settings['disable_parameter_scaling'])):
            value = scale_reward(self._q_val(), self.getRewardBounds()) * (1.0 / (1.0- self.getSettings()['discount_factor']))
            # return (self._q_val())[0]
        else:
            value = scale_reward(self._q_val(), self.getRewardBounds()) * (1.0 / (1.0- self.getSettings()['discount_factor']))
#         print ("Agent scaled value: ", value)
        return value
        # return self._q_valTarget()[0]
        # return self._q_val()[0]
    
    def q_values(self, state, alreadyNormed=True):
        """
            For returning a vector of q values
        """
        """
        if ( ('disable_parameter_scaling' in self._settings) and (self._settings['disable_parameter_scaling'])):
            pass
        else:
        """
        state = norm_state(state, self._state_bounds)
        state = np.array(state, dtype=self._settings['float_type'])
        self._model.setStates(state)
        self._modelTarget.setStates(state)
        if ( ('disable_parameter_scaling' in self._settings) and (self._settings['disable_parameter_scaling'])):
            return scale_reward(self._q_val(), self.getRewardBounds()) * (1.0 / (1.0- self.getSettings()['discount_factor']))
            # return (self._q_val())[0] 
        else:
            return scale_reward(self._q_val(), self.getRewardBounds()) * (1.0 / (1.0- self.getSettings()['discount_factor']))
        # return self._q_valTarget()
        # return self._q_val()
    
    def q_valueWithDropout(self, state):
        # states = np.zeros((self._batch_size, self._state_length), dtype=self._settings['float_type'])
        # states[0, ...] = state
        if ( ('disable_parameter_scaling' in self._settings) and (self._settings['disable_parameter_scaling'])):
            pass
        else:
            state = norm_state(state, self._state_bounds)
            
        state = np.array(state, dtype=self._settings['float_type'])
        self._model.setStates(state)
        if ( ('disable_parameter_scaling' in self._settings) and (self._settings['disable_parameter_scaling'])):
            return scale_reward(self._q_val_drop(), self.getRewardBounds()) * (1.0 / (1.0- self.getSettings()['discount_factor']))
            # return (self._q_val_drop())[0]
        else:
            return scale_reward(self._q_val_drop(), self.getRewardBounds()) * (1.0 / (1.0- self.getSettings()['discount_factor']))
    
    def bellman_error(self, states, actions, rewards, result_states, falls):
        self.setData(states, actions, rewards, result_states, falls)
        return self._bellman_error2()
        # return self._bellman_errorTarget()

    
    def train(self, states, actions, rewards, result_states):
        loss = self.trainCritic(states, actions, rewards, result_states)
        lossActor = self.trainActor(states, actions, rewards, result_states)
        return loss
    
    def getSettings(self):
        return self._settings
    def setSettings(self, settings_):
        self._settings = copy.deepcopy(settings_)
    
    def setStateBounds(self, bounds):
        self._state_bounds = bounds
    def setActionBounds(self, bounds):
        self._action_bounds = bounds
    def setRewardBounds(self, bounds):
        self._reward_bounds = bounds
    def getStateBounds(self):
        return self._state_bounds
    def getActionBounds(self):
        return self._action_bounds
    def getRewardBounds(self):
        return self._reward_bounds
        
        ### Setting network input values ###    
    def setStates(self, states):
        self._states_shared.set_value(states)
    def setActions(self, actions):
        self._actions_shared.set_value(actions)
    def setResultStates(self, resultStates):
        self._next_states_shared.set_value(resultStates)
    def setRewards(self, rewards):
        self._rewards_shared.set_value(rewards)
    
    def getStateSize(self):
        return self._state_length
    def getActionSize(self): 
        return self._action_length

    def init(self, state_length, action_length, state_bounds, action_bounds, actor, exp, settings):
        pass
    
    def initEpoch(self):
        pass
    
    def trainDyna(self, predicted_states, actions, rewards, result_states, falls):
        """
            Performs a DYNA type update
            Because I am using target network a direct DYNA update does nothing. 
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
        pass
    
    def clearExperts(self):
        """
            Remore all expert policies used by this method
        """
        pass