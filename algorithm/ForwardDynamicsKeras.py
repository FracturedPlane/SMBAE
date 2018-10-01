import theano
from theano import tensor as T
import numpy as np
# import lasagne
import sys
sys.path.append('../')
from model.ModelUtil import *
from model.LearningUtil import loglikelihood, loglikelihoodMEAN, kl, entropy, flatgrad, zipsame, get_params_flat, setFromFlat, likelihood, loglikelihoodMEAN
from model.LearningUtil import loglikelihood, likelihood, likelihoodMEAN, kl, kl_D, entropy, flatgrad, zipsame, get_params_flat, setFromFlat
from keras.optimizers import SGD
# from keras.utils.np_utils import to_categoricalnetwork
import keras.backend as K
import keras

# For debugging
# theano.config.mode='FAST_COMPILE'
from algorithm.AlgorithmInterface import AlgorithmInterface

class ForwardDynamicsKeras(AlgorithmInterface):
    
    def __init__(self, model, state_length, action_length, state_bounds, action_bounds, settings_):

        super(ForwardDynamicsKeras,self).__init__(model, state_length, action_length, state_bounds, action_bounds, 0, settings_)
        self._model = model
        self._learning_rate = self.getSettings()["fd_learning_rate"]
        self._regularization_weight = 1e-6
        
        condition_reward_on_result_state = False
        self._train_combined_loss = False
        
        ### data types for model
        self._fd_grad_target = T.matrix("FD_Grad")
        self._fd_grad_target.tag.test_value = np.zeros((self._batch_size,self._state_length), dtype=np.dtype(self.getSettings()['float_type']))
        self._fd_grad_target_shared = theano.shared(
            np.zeros((self._batch_size, self._state_length),
                      dtype=self.getSettings()['float_type']))
        
        ##
        
        self._forward = self._model.getForwardDynamicsNetwork()([self._model._stateInput, self._model._actionInput])
        self._reward = self._model.getRewardNetwork()([self._model._stateInput, self._model._actionInput])
        
        ForwardDynamicsKeras.compile(self)
    
    def compile(self):
        # sgd = SGD(lr=0.001, momentum=0.9)
        sgd = keras.optimizers.Adam(lr=np.float32(self.getSettings()['fd_learning_rate']), beta_1=np.float32(0.95), beta_2=np.float32(0.999), epsilon=np.float32(self._rms_epsilon), decay=np.float32(0.0))
        print ("Clipping: ", sgd.decay)
        print("sgd, critic: ", sgd)
        self._model.getRewardNetwork().compile(loss='mse', optimizer=sgd)
        # sgd = SGD(lr=0.0005, momentum=0.9)
        sgd = keras.optimizers.Adam(lr=np.float32(self.getSettings()['fd_learning_rate']), beta_1=np.float32(0.95), beta_2=np.float32(0.999), epsilon=np.float32(self._rms_epsilon), decay=np.float32(0.0))
        print("sgd, actor: ", sgd)
        print ("Clipping: ", sgd.decay)
        self._model.getForwardDynamicsNetwork().compile(loss='mse', optimizer=sgd)

        self._params = self._model.getForwardDynamicsNetwork().trainable_weights        
        """
        weights = [self._model._actionInput]
        gradients = K.gradients(T.mean(self._q_function), [self._model._stateInput]) # gradient tensors
        ### DPG related functions
        self._get_gradients = K.function(inputs=[self._model._stateInput], outputs=gradients)
        """
        ### Get reward input grad
        weights = [self._model._actionInput]
        reward_gradients = K.gradients(T.mean(self._reward), [self._model._actionInput]) # gradient tensors
        ### DPG related functions
        self._get_grad_reward = K.function(inputs=[self._model._stateInput, self._model._actionInput, K.learning_phase()], outputs=reward_gradients)
        
        
        self._get_grad = theano.function([self._model._stateInput, self._model._actionInput, K.learning_phase()], outputs=T.grad(cost=None, wrt=[self._model._actionInput] + self._params,
                                                            known_grads={self._forward: self._fd_grad_target_shared}), 
                                         allow_input_downcast=True)
        
        # self._get_grad_reward = theano.function([], outputs=lasagne.updates.get_or_compute_grads((self._reward_loss_NoDrop), [lasagne.layers.get_all_layers(self._model.getRewardNetwork())[0].input_var] + self._reward_params), allow_input_downcast=True,
        # self._get_grad_reward = theano.function([], outputs=lasagne.updates.get_or_compute_grads(T.mean(self._reward), [self._model._actionInputVar] + self._reward_params), allow_input_downcast=True, 
        #                                         givens=self._inputs_reward_)
        
        self.fd = K.function([self._model._stateInput, self._model._actionInput, K.learning_phase()], [self._forward])
        self.reward = K.function([self._model._stateInput, self._model._actionInput, K.learning_phase()], [self._reward])
        
    def getNetworkParameters(self):
        params = []
        params.append(copy.deepcopy(self._model.getForwardDynamicsNetwork().get_weights()))
        params.append(copy.deepcopy(self._model.getRewardNetwork().get_weights()))
        return params
    
    def setNetworkParameters(self, params):
        self._model.getForwardDynamicsNetwork().set_weights(params[0])
        self._model.getRewardNetwork().set_weights( params[1] )
        
    def setData(self, states, actions, result_states=None, rewards=None):
        pass
        """
        self._model.setStates(states)
        if not (result_states is None):
            self._model.setResultStates(result_states)
        self._model.setActions(actions)
        if not (rewards is None):
            self._model.setRewards(rewards)
            """
            
    def setGradTarget(self, grad):
        self._fd_grad_target_shared.set_value(grad)
        
    def getGrads(self, states, actions, result_states, v_grad=None, alreadyNormed=False):
        if ( alreadyNormed == False ):
            states = np.array(norm_state(states, self._state_bounds), dtype=self.getSettings()['float_type'])
            actions = np.array(norm_action(actions, self._action_bounds), dtype=self.getSettings()['float_type'])
            result_states = np.array(norm_state(result_states, self._state_bounds), dtype=self.getSettings()['float_type'])
        # result_states = np.array(result_states, dtype=self.getSettings()['float_type'])
        # self.setData(states, actions, result_states)
        # if (v_grad != None):
        # print ("states shape: ", states.shape, " actions shape: ", actions.shape, " v_grad.shape: ", v_grad.shape)
        self.setGradTarget(v_grad)
        # print ("states shape: ", states.shape, " actions shape: ", actions.shape)
        # grad = self._get_grad([states, actions])[0]
        grad = np.zeros_like(states)
        print ("grad: ", grad)
        return grad
    
    def getRewardGrads(self, states, actions, alreadyNormed=False):
        # states = np.array(states, dtype=self.getSettings()['float_type'])
        # actions = np.array(actions, dtype=self.getSettings()['float_type'])
        if ( alreadyNormed is False ):
            states = np.array(norm_state(states, self._state_bounds), dtype=self.getSettings()['float_type'])
            actions = np.array(norm_action(actions, self._action_bounds), dtype=self.getSettings()['float_type'])
            # rewards = np.array(norm_state(rewards, self._reward_bounds), dtype=self.getSettings()['float_type'])
        # self.setData(states, actions)
        return self._get_grad_reward([states, actions, 0])[0]
                
    def train(self, states, actions, result_states, rewards):
        # rewards = rewards * (1.0/(1.0-self.getSettings()['discount_factor'])) # scale rewards
        # self.setData(states, actions, result_states, rewards)
        # print ("Performing Critic trainning update")
        #if (( self._updates % self._weight_update_steps) == 0):
        #    self.updateTargetModel()
        self._updates += 1
        # all_paramsActA = lasagne.layers.helper.get_all_param_values(self._l_outActA)
        if ( self._train_combined_loss ):
            pass
            # loss = self._train_combined()
            # loss = self._train_combined()
        else:
            score = self._model.getForwardDynamicsNetwork().fit([states, actions], result_states,
              nb_epoch=1, batch_size=states.shape[0],
              verbose=0
              # callbacks=[early_stopping],
              )
            loss = score.history['loss'][0]
            if ( self.getSettings()['train_reward_predictor']):
                # print ("self._reward_bounds: ", self._reward_bounds)
                # print( "Rewards, predicted_reward, difference, model diff, model rewards: ", np.concatenate((rewards, self._predict_reward(), self._predict_reward() - rewards, self._reward_error(), self._reward_values()), axis=1))
                score = self._model.getRewardNetwork().fit([states, actions], rewards,
                  nb_epoch=1, batch_size=states.shape[0],
                  verbose=0
                  # callbacks=[early_stopping],
                  )
                lossReward = score.history['loss'][0]
                if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
                    print ("Loss Reward: ", lossReward)
            if ( 'train_state_encoding' in self.getSettings() and (self.getSettings()['train_state_encoding'])):
                pass
                # lossEncoding = self._train_state_encoding()
                # if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
                #    print ("Loss Encoding: ", lossEncoding)     
        # This undoes the Actor parameter updates as a result of the Critic update.
        # print (diff_)
        return loss
    
    def predict(self, state, action):
        state = np.array(norm_state(state, self._state_bounds), dtype=self.getSettings()['float_type'])
        action = np.array(norm_action(action, self._action_bounds), dtype=self.getSettings()['float_type'])
        state_ = scale_state(self.fd([state, action,0])[0], self._state_bounds)
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
        state = np.array(norm_state(state, self._state_bounds), dtype=self.getSettings()['float_type'])
        action = np.array(norm_action(action, self._action_bounds), dtype=self.getSettings()['float_type'])
        state_ = self._forwardDynamics_std() * (action_bound_std(self._state_bounds))
        return state_
    
    def predict_reward(self, state, action):
        # states = np.zeros((self._batch_size, self._self._state_length), dtype=theano.config.floatX)
        # states[0, ...] = state
        state = np.array(norm_state(state, self._state_bounds), dtype=self.getSettings()['float_type'])
        action = np.array(norm_action(action, self._action_bounds), dtype=self.getSettings()['float_type'])
        predicted_reward = self.reward([state, action, 0])[0]
        reward_ = scale_reward(predicted_reward, self.getRewardBounds()) # * (1.0 / (1.0- self.getSettings()['discount_factor']))
        # reward_ = scale_reward(predicted_reward, self.getRewardBounds())[0] * (1.0 / (1.0- self.getSettings()['discount_factor']))
        # reward_ = scale_state(predicted_reward, self._reward_bounds)
        # print ("reward, predicted reward: ", reward_, predicted_reward)
        return reward_
    
    def predict_batch(self, states, actions):
        ## These input should already be normalized.
        return self.fd([states, actions, 0])[0]
    
    def predict_reward_batch(self, states, actions):
        """
            This data should already be normalized
        """
        # states = np.zeros((self._batch_size, self._self._state_length), dtype=theano.config.floatX)
        # states[0, ...] = state
        predicted_reward = self.reward([states, actions, 0])[0]
        return predicted_reward

    def bellman_error(self, states, actions, result_states, rewards):
        self.setData(states, actions, result_states, rewards)
        predicted_y = self.predict(states, actions)
        diff = np.mean(np.abs(predicted_y - result_states))
        return diff
    
    def reward_error(self, states, actions, result_states, rewards):
        # rewards = rewards * (1.0/(1.0-self.getSettings()['discount_factor'])) # scale rewards
        predicted_y = self.predict_reward(states, actions)
        diff = np.mean(np.abs(predicted_y - result_states))
        return diff
