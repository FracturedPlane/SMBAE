import theano
from theano import tensor as T
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
# from DeepCACLA import DeepCACLA

class CACLA_KERAS(AlgorithmInterface):
    
    def __init__(self, model, n_in, n_out, state_bounds, action_bounds, reward_bound, settings_):

        super(CACLA_KERAS,self).__init__(model, n_in, n_out, state_bounds, action_bounds, reward_bound, settings_)
        
        ## primary network
        self._model = model

        # print ("Loss ", self._model.getActorNetwork().total_loss)
        
        ## Target network
        self._modelTarget = copy.deepcopy(model)
        # self._modelTarget = model
        
        self._q_valsActA = self._model.getActorNetwork()(self._model._stateInput)
        self._q_valsActTarget = self._modelTarget.getActorNetwork()(self._model._stateInput)
        self._q_valsActASTD = ( T.ones_like(self._q_valsActA)) * self.getSettings()['exploration_rate']
        self._q_valsActTargetSTD = (T.ones_like(self._q_valsActTarget)) * self.getSettings()['exploration_rate']
        
        self._actor_buffer_states=[]
        self._actor_buffer_result_states=[]
        self._actor_buffer_actions=[]
        self._actor_buffer_rewards=[]
        self._actor_buffer_falls=[]
        self._actor_buffer_diff=[]
        
        CACLA_KERAS.compile(self)
        
    def compile(self):
        # sgd = SGD(lr=0.001, momentum=0.9)
        sgd = keras.optimizers.Adam(lr=np.float32(self.getSettings()['critic_learning_rate']), beta_1=np.float32(0.9), beta_2=np.float32(0.999), epsilon=np.float32(self._rms_epsilon), decay=np.float32(0.0))
        print ("Clipping: ", sgd.decay)
        print("sgd, critic: ", sgd)
        self._model.getCriticNetwork().compile(loss='mse', optimizer=sgd)
        # sgd = SGD(lr=0.0005, momentum=0.9)
        sgd = keras.optimizers.Adam(lr=np.float32(self.getSettings()['learning_rate']), beta_1=np.float32(0.9), beta_2=np.float32(0.999), epsilon=np.float32(self._rms_epsilon), decay=np.float32(0.0))
        print("sgd, actor: ", sgd)
        print ("Clipping: ", sgd.decay)
        self._model.getActorNetwork().compile(loss='mse', optimizer=sgd)
        
        self._q_action_std = K.function([self._model._stateInput], [self._q_valsActASTD])
        
    def updateTargetModel(self):
        print ("Updating target Model")
        """
            Target model updates
        """
        self._modelTarget.getCriticNetwork().set_weights( copy.deepcopy(self._model.getCriticNetwork().get_weights()))
        self._modelTarget.getActorNetwork().set_weights( copy.deepcopy(self._model.getActorNetwork().get_weights()))
    
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
            params[0][i] = np.array(params[0][i], dtype=self._settings['float_type'])
            """
        self._model.getCriticNetwork().set_weights(params[0])
        self._model.getActorNetwork().set_weights( params[1] )
        self._modelTarget.getCriticNetwork().set_weights( params[2])
        self._modelTarget.getActorNetwork().set_weights( params[3])
    
    def setData(self, states, actions, rewards, result_states, fallen):
        pass
        # _targets = rewards + (self._discount_factor * self._q_valsTargetNextState )
        
    def trainCritic(self, states, actions, rewards, result_states, falls):
        self.setData(states, actions, rewards, result_states, falls)
        # print ("Performing Critic trainning update")
        
        if (( self._updates % self._weight_update_steps) == 0):
            self.updateTargetModel()
        self._updates += 1
        # print ("Falls:", falls)
        # print ("Rewards: ", rewards)
        # print ("Target Values: ", self._get_target())
        # print ("V Values: ", np.mean(self._q_val()))
        # print ("diff Values: ", np.mean(self._get_diff()))
        # data = np.append(falls, self._get_target()[0], axis=1)
        # print ("Rewards, Falls, Targets:", np.append(rewards, data, axis=1))
        # print ("Rewards, Falls, Targets:", [rewards, falls, self._get_target()])
        # print ("Actions: ", actions)
        y_ = self._modelTarget.getCriticNetwork().predict(result_states, batch_size=states.shape[0])
        v = self._model.getCriticNetwork().predict(states, batch_size=states.shape[0])
        # target_ = rewards + ((self._discount_factor * y_) * falls)
        target_ = rewards + ((self._discount_factor * y_))
        target_ = np.array(target_, dtype=self._settings['float_type'])
        # states = np.array(states, dtype=self._settings['float_type'])
        # print ("target type: ", target_.dtype)
        # print ("states type: ", states.dtype)
        # print ("Critic Target: ", np.concatenate((v, target_, rewards, y_) ,axis=1) )
        score = self._model.getCriticNetwork().fit(states, target_,
              nb_epoch=1, batch_size=32,
              verbose=0
              # callbacks=[early_stopping],
              )
        loss = score.history['loss'][0]
        # print(" Critic loss: ", loss)
        
        return loss
    
    def trainActor(self, states, actions, rewards, result_states, falls, advantage, exp_actions=None, forwardDynamicsModel=None):
        lossActor = 0
        
        diff_ = self.bellman_error(states, actions, rewards, result_states, falls)
        # print ("Action diff: ", diff_)
        # print ("Diff")
        # print (diff_)
        for i in range(len(diff_)):
            if ( diff_[i][0] > 0.0 and (exp_actions[i] == 1)):
                # if (('dont_use_advantage' in self.getSettings()) and self.getSettings()['dont_use_advantage']):
                self._actor_buffer_diff.append([1.0])
                    #  print("Not using advantage")
                # else:
                    # self._actor_buffer_diff.append(diff_[i])
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
            tmp_diff = self._actor_buffer_diff[:self.getSettings()['batch_size']]
            # self.setData(tmp_states, tmp_actions, tmp_rewards, tmp_result_states, tmp_falls)
            # self._tmp_diff_shared.set_value(tmp_diff)
            # print ("Actor diff: ", np.mean(np.array(self._get_diff()) / (1.0/(1.0-self._discount_factor))))
            score = self._model.getActorNetwork().fit(np.array(tmp_states), np.array(tmp_actions),
              nb_epoch=1, batch_size=len(tmp_actions),
              verbose=0
              # callbacks=[early_stopping],
              )
            lossActor = score.history['loss'][0]
            if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['train']):
                print( "Length of positive actions: " , str(len(tmp_actions)), " Actor loss: ", lossActor, " actor buffer size: ", len(self._actor_buffer_actions))
                
            if (self.getSettings()["print_levels"][self.getSettings()["print_level"]] >= self.getSettings()["print_levels"]['debug']):
                actions_ = self._model.getActorNetwork().predict(states, batch_size=states.shape[0])
                print("Mean action: ", np.mean(actions_, axis=0), " std ", np.std(actions_, axis=0))
            ### Remove batch from buffer
            self._actor_buffer_states=self._actor_buffer_states[self.getSettings()['batch_size']:]
            self._actor_buffer_actions = self._actor_buffer_actions[self.getSettings()['batch_size']:]
            self._actor_buffer_rewards = self._actor_buffer_rewards[self.getSettings()['batch_size']:]
            self._actor_buffer_result_states = self._actor_buffer_result_states[self.getSettings()['batch_size']:]
            self._actor_buffer_falls =self._actor_buffer_falls[self.getSettings()['batch_size']:]
            self._actor_buffer_diff = self._actor_buffer_diff[self.getSettings()['batch_size']:]
        
            # print ("Actor diff: ", np.mean(np.array(self._get_diff()) / (1.0/(1.0-self._discount_factor))))
            # lossActor, _ = self._trainActor()
            # print( "Length of positive actions: " , str(len(tmp_actions)), " Actor loss: ", lossActor)
            # print( " Actor loss: ", lossActor)
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
    
    def predict_std(self, state, deterministic_=True):
        state = norm_state(state, self._state_bounds)   
        state = np.array(state, dtype=self._settings['float_type'])
        
        # action_std = self._model.getActorNetwork().predict(state, batch_size=1)[:,self._action_length:] * (action_bound_std(self._action_bounds))
        action_std = self._q_action_std([state])[0] * action_bound_std(self._action_bounds)
        # print ("Policy std: ", action_std)
        return action_std
    
    def predictWithDropout(self, state, deterministic_=True):
        # states = np.zeros((self._batch_size, self._state_length), dtype=self._settings['float_type'])
        # states[0, ...] = state
        state = np.array(state, dtype=self._settings['float_type'])
        state = norm_state(state, self._state_bounds)
        self._model.setStates(state)
        # action_ = lasagne.layers.get_output(self._model.getActorNetwork(), state, deterministic=deterministic_).mean()
        # action_ = scale_action(self._q_action()[0], self._action_bounds)
        # if deterministic_:
        action_ = scale_action(self._model.getActorNetwork().predict(states, batch_size=1)[0], self._action_bounds)
        # else:
        # action_ = scale_action(self._q_action()[0], self._action_bounds)
        # action_ = q_valsActA[0]
        return action_
    
    def q_value(self, state):
        # states = np.zeros((self._batch_size, self._state_length), dtype=self._settings['float_type'])
        # states[0, ...] = state
        state = norm_state(state, self._state_bounds)
        state = np.array(state, dtype=self._settings['float_type'])
        self._model.setStates(state)
        self._modelTarget.setStates(state)
        # return scale_reward(self._q_valTarget(), self.getRewardBounds())[0]
        value = scale_reward(self._model.getCriticNetwork().predict(state, batch_size=1), self.getRewardBounds()) * (1.0 / (1.0- self.getSettings()['discount_factor']))
        return value
        # return self._q_val()[0]
    
    def q_values(self, state):
        """
            For returning a vector of q values, state should already be normalized
        """
        state = np.array(state, dtype=self._settings['float_type'])
        self._model.setStates(state)
        self._modelTarget.setStates(state)
        return self._model.getCriticNetwork().predict(state, batch_size=state.shape[0])
    
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
        y_ = self._modelTarget.getCriticNetwork().predict(result_states, batch_size=32)
        target_ = rewards + ((self._discount_factor * y_))
        values =  self._model.getCriticNetwork().predict(states, batch_size=32)
        bellman_error = target_ - values
        return bellman_error
        # return self._bellman_errorTarget()
