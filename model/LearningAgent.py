"""
    An interface class for Agents to be used in the system.

"""
from multiprocessing import Process, Queue
# from pathos.multiprocessing import Pool
import threading
import time
from model.AgentInterface import AgentInterface
from model.ModelUtil import *
import os
import numpy
import copy
import time
import datetime
# numpy.set_printoptions(threshold=numpy.nan)

class LearningAgent(AgentInterface):
    
    def __init__(self, n_in, n_out, state_bounds, action_bounds, reward_bound, settings_):
        super(LearningAgent,self).__init__(n_in, n_out, state_bounds, action_bounds, reward_bound, settings_)
        self._useLock = False
        if self._useLock:
            self._accesLock = threading.Lock()
        self._pol = None
        self._fd = None
        self._sampler = None
        self._expBuff = None
        self._expBuff_FD = None
        
    def getPolicy(self):
        if self._useLock:
            self._accesLock.acquire()
        pol = self._pol
        if self._useLock:
            self._accesLock.release()
        return pol
        
    def setPolicy(self, pol):
        if self._useLock:
            self._accesLock.acquire()
        self._pol = pol
        if (not self._sampler == None ):
            self._sampler.setPolicy(pol)
        if self._useLock:
            self._accesLock.release()
        
    def getForwardDynamics(self):
        if self._useLock:
            self._accesLock.acquire()
        fd = self._fd
        if self._useLock:
            self._accesLock.release()
        return fd
                
    def setForwardDynamics(self, fd):
        if self._useLock:
            self._accesLock.acquire()
        self._fd = fd
        if self._useLock:
            self._accesLock.release()
        
    def setSettings(self, settings):
        self._settings = settings
        
    def setExperience(self, experienceBuffer):
        self._expBuff = experienceBuffer 
    def getExperience(self):
        return self._expBuff
    
    def setFDExperience(self, experienceBuffer):
        self._expBuff_FD = experienceBuffer 
    def getFDExperience(self):
        return self._expBuff_FD  
    
    def train(self, _states, _actions, _rewards, _result_states, _falls, _advantage=None, _exp_actions=None):
        if self._useLock:
            self._accesLock.acquire()
        cost = 0
        
        # print ("Bounds comparison: ", self._pol.getStateBounds(), " exp mem: ", 
        #        self._expBuff.getStateBounds())
        # print ("Bounds comparison: ", self._pol.getActionBounds(), " exp mem: ", 
        #        self._expBuff.getActionBounds())
        
        # print("_exp_actions: ", _exp_actions)
        # print ("Shapes of things: ")
        # print ("State: ", np.array(_states).shape)
        # print ("Actions: ", np.array(_actions).shape)
        # print ("Advantage: ", np.array(_advantage).shape)
        if ("value_function_batch_size" in self._settings):
            value_function_batch_size = self._settings['value_function_batch_size']
        else:
            value_function_batch_size = self._settings["batch_size"]
        if self._settings['on_policy']:
            if ( ('clear_exp_mem_on_poli' in self._settings) 
                 and (self._settings['clear_exp_mem_on_poli'] == True)):
                self._expBuff.clear()
            if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['debug']):
                print("Start of Learning Agent Update")
                _actions__ = np.array(_actions)
                print("Actions:     ", np.mean(_actions__, axis=0), " shape: ", _actions__.shape)
                print("Actions std:  ", np.std(_actions__, axis=0) )
            
            ### Validate data
            tmp_states = []
            tmp_actions = []
            tmp_result_states = [] 
            tmp_rewards = []
            tmp_falls = []
            tmp_advantage = []
            tmp_exp_action = []
            # print ("Advantage:", _advantage)
            # print ("rewards:", _rewards)
            # print("Batch size: ", len(_states), len(_actions), len(_result_states), len(_rewards), len(_falls), len(_advantage))
            ### validate every tuples before giving them to the learning method
            num_samples_=0
            t0 = time.time()
            self.getExperience()._settings["keep_running_mean_std_for_scaling"] = False
            for (state__, action__, next_state__, reward__, fall__, advantage__, exp_action__) in zip(_states, _actions, _result_states, _rewards, _falls, _advantage, _exp_actions):
                if (checkValidData(state__, action__, next_state__, reward__)):
                    tmp_states.append(state__)
                    tmp_actions.append(action__)
                    tmp_result_states.append(next_state__)
                    tmp_rewards.append(reward__)
                    tmp_falls.append(fall__)
                    tmp_advantage.append(advantage__)
                    tmp_exp_action.append(exp_action__)
                    # print("adv__:", advantage__)
                    tup = ([state__], [action__], [next_state__], [reward__], [fall__], [advantage__], [exp_action__])
                    self.getExperience().insertTuple(tup)
                    if ( 'keep_seperate_fd_exp_buffer' in self._settings and (self._settings['keep_seperate_fd_exp_buffer'])):
                        self.getFDExperience().insertTuple(tup)
                    num_samples_ = num_samples_ + 1
                # else:
                    # print ("Tuple invalid:")
            t1 = time.time()
            if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['debug']):
                sim_time_ = datetime.timedelta(seconds=(t1-t0))
                print ("Pushing data into exp buffer complete in " + str(sim_time_) + " seconds")
                        
            if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):        
                print ("self._expBuff.samples(): ", self._expBuff.samples())
            _states = np.array(norm_action(np.array(tmp_states), self._pol.getStateBounds()), dtype=self._settings['float_type'])
            # print("Learning Agent: Get state bounds: ", self._pol.getStateBounds())
            # print ("ExpMem: Get state bounds ", self._expBuff.getStateBounds())
            # print("ExpMem: Get state bounds: ", self.getExperience().getStateBounds())
            if ( ('disable_parameter_scaling' in self._settings) and (self._settings['disable_parameter_scaling'])):
                _actions = np.array(tmp_actions, dtype=self._settings['float_type'])
                # _actions = np.array(norm_action(np.array(tmp_actions), self._action_bounds), dtype=self._settings['float_type'])
            else: ## Doesn't really matter for on policy methods
                _actions = np.array(norm_action(np.array(tmp_actions), self._pol.getActionBounds()), dtype=self._settings['float_type'])
            _result_states = np.array(norm_action(np.array(tmp_result_states), self._pol.getStateBounds()), dtype=self._settings['float_type'])
            # reward.append(norm_state(self._reward_history[i] , self._reward_bounds ) * ((1.0-self._settings['discount_factor']))) # scale rewards
            _rewards = np.array(norm_state(tmp_rewards , self._reward_bounds ) * ((1.0-self._settings['discount_factor'])), dtype=self._settings['float_type'])
            _rewards = np.reshape(_rewards, (len(tmp_states), 1))
            _falls = np.array(tmp_falls, dtype='int8')
            _advantage = np.array(tmp_advantage, dtype=self._settings['float_type'])
            _exp_action = np.array(tmp_exp_action, dtype=self._settings['float_type'])
            # print("Not Falls: ", _falls)
            # print("Rewards: ", _rewards)
            # print ("Actions after: ", _actions)
            cost = 0
            additional_on_poli_trianing_updates = 1
            if ( "additional_on-poli_trianing_updates" in self._settings 
                 and (self._settings["additional_on-poli_trianing_updates"] != False)):
                additional_on_poli_trianing_updates = self._settings["additional_on-poli_trianing_updates"]
                ### The data should be seen ~ 4 times
                additional_on_poli_trianing_updates = int(((self._settings["num_on_policy_rollouts"] * self._settings["max_epoch_length"] * 1) / value_function_batch_size) * additional_on_poli_trianing_updates)
                print ("additional_on_poli_trianing_updates: ", additional_on_poli_trianing_updates)
                
                if ( additional_on_poli_trianing_updates < 1 ):
                    additional_on_poli_trianing_updates = 1  
                
            for ii__ in range(additional_on_poli_trianing_updates):
                if (self._settings['train_critic']):
                    t0 = time.time()
                    if (self._settings['critic_updates_per_actor_update'] > 1):
                        
                        for i in range(self._settings['critic_updates_per_actor_update']):
                            if ( self._settings['agent_name'] == "algorithm.QProp.QProp"
                              or (self._settings['agent_name'] == 'algorithm.QPropKeras.QPropKeras')
                              ):
                                states__, actions__, result_states__, rewards__, falls__, G_ts__, exp_actions__ = self._expBuff.getNonMBAEBatch(min(value_function_batch_size, self._expBuff.samples()))
                                loss = self._pol.trainOnPolicyCritic(states=states__, actions=actions__, rewards=rewards__, result_states=result_states__, falls=falls__)
                            # loss = self._pol.trainOnPolicyCritic(states=tmp_states, actions=tmp_actions, rewards=tmp_rewards, result_states=tmp_result_states, falls=tmp_falls)
                            # print ("Number of samples:", self._expBuff.samples())
                            if ( 'give_mbae_actions_to_critic' in self._settings and 
                                 (self._settings['give_mbae_actions_to_critic'] == False)):
                                # if ( np.random.random() >= self._settings['model_based_action_omega']):
                                if ( np.random.random() >= -1.0):
                                    states__, actions__, result_states__, rewards__, falls__, G_ts__, exp_actions__ = self._expBuff.getNonMBAEBatch(min(value_function_batch_size, self._expBuff.samples()))
                                    loss = self._pol.trainCritic(states=states__, actions=actions__, rewards=rewards__, result_states=result_states__, falls=falls__)
                                else:
                                    # print('off-policy action update')
                                    states__, actions__, result_states__, rewards__, falls__, G_ts__, exp_actions__ = self._expBuff.get_batch(min(value_function_batch_size, self._expBuff.samples()))
                                    actions____ = self._pol.predict_batch(states=result_states__) 
                                    predicted_result_states__ = self._fd.predict_batch(states=result_states__, actions=actions____)
                                    rewards____ = self._fd.predict_reward_batch(states=result_states__, actions=actions____)
                                    loss = self._pol.trainCritic(states=result_states__, actions=actions____, rewards=rewards____, result_states=predicted_result_states__, falls=falls__)
                            else:
                                if ( 'keep_seperate_fd_exp_buffer' in self._settings 
                                     and (self._settings['keep_seperate_fd_exp_buffer'] == True)
                                    and ('train_critic_with_fd_data' in self._settings) 
                                     and (self._settings['train_critic_with_fd_data'] == True)
                                     ):
                                    # print ("Using seperate (off-policy) exp mem for Q model")
                                    states__, actions__, result_states__, rewards__, falls__, G_ts__, exp_actions__ = self.getFDExperience().get_batch(value_function_batch_size)
                                else:
                                    states__, actions__, result_states__, rewards__, falls__, G_ts__, exp_actions__ = self._expBuff.get_batch(min(value_function_batch_size, self._expBuff.samples()))
                                loss = self._pol.trainCritic(states=states__, actions=actions__, rewards=rewards__, result_states=result_states__, falls=falls__)
                            if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
                                print("Critic loss: ", loss)
                            if not np.isfinite(cost) or (cost > 500) :
                                numpy.set_printoptions(threshold=numpy.nan)
                                print ("States: " + str(states__) + " ResultsStates: " + str(result_states__) + " Rewards: " + str(rewards__) + " Actions: " + str(actions__))
                                print ("Training cost is Odd: ", cost)
                            
                    else:
                        _states, _actions, _result_states, _rewards, _falls, _advantage, exp_actions__ = self._expBuff.get_batch(self._settings["batch_size"])
                        cost = self._pol.trainCritic(states=_states, actions=_actions, rewards=_rewards, result_states=_result_states, falls=_falls)
                        if not np.isfinite(cost) or (cost > 500) :
                            numpy.set_printoptions(threshold=numpy.nan)
                            print ("States: " + str(_states) + " ResultsStates: " + str(_result_states) + " Rewards: " + str(_rewards) + " Actions: " + str(_actions))
                            print ("Training cost is Odd: ", cost)
                    
                    t1 = time.time()
                    if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['debug']):
                        sim_time_ = datetime.timedelta(seconds=(t1-t0))
                        print ("Critic training complete in " + str(sim_time_) + " seconds")
                if (self._settings['train_forward_dynamics']):
                    t0 = time.time()
                    if (self._settings['critic_updates_per_actor_update'] > 1):
                        for i in range(self._settings['critic_updates_per_actor_update']):
                            if ( 'keep_seperate_fd_exp_buffer' in self._settings and (self._settings['keep_seperate_fd_exp_buffer'])):
                                # print ("Using seperate (off-policy) exp mem for FD model")
                                states__, actions__, result_states__, rewards__, falls__, G_ts__, exp_actions__ = self.getFDExperience().get_batch(value_function_batch_size)
                            else:
                                states__, actions__, result_states__, rewards__, falls__, G_ts__, exp_actions__ = self.getExperience().get_batch(value_function_batch_size)
                            dynamicsLoss = self._fd.train(states=states__, actions=actions__, result_states=result_states__, rewards=rewards__)
                            if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
                                print ("Forward Dynamics Loss: ", dynamicsLoss)
                            
                                # cost = self._pol.trainDyna(predicted_states=predicted_result_states__, actions=actions__, rewards=rewards__, result_states=result_states__, falls=falls__)
                            if (self._settings['train_critic_on_fd_output'] and 
                                (( self._pol.numUpdates() % self._settings['dyna_update_lag_steps']) == 0) and 
                                ( ( self._pol.numUpdates() %  self._settings['steps_until_target_network_update']) >= (self._settings['steps_until_target_network_update']/10)) and
                                ( ( self._pol.numUpdates() %  self._settings['steps_until_target_network_update']) <= (self._settings['steps_until_target_network_update'] - (self._settings['steps_until_target_network_update']/10)))
                                ):
                                predicted_result_states__ = self._fd.predict_batch(states=states__, actions=actions__)
                                cost = self._pol.trainDyna(predicted_states=predicted_result_states__, actions=actions__, rewards=rewards__, result_states=result_states__, falls=falls__)
                                
                                if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
                                    print("Performing Dyna Update, loss: ", cost)
                                # print("Updated params: ", self._pol.getNetworkParameters()[0][0][0])
                    else:
                        dynamicsLoss = self._fd.train(states=_states, actions=_actions, rewards=_rewards, result_states=_result_states)
                        if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
                                print ("Forward Dynamics Loss: ", dynamicsLoss)
                                
                        if (self._settings['train_critic_on_fd_output'] and 
                            (( self._pol.numUpdates() % self._settings['dyna_update_lag_steps']) == 0) and 
                            ( ( self._pol.numUpdates() %  self._settings['steps_until_target_network_update']) >= (self._settings['steps_until_target_network_update']/10)) and
                            ( ( self._pol.numUpdates() %  self._settings['steps_until_target_network_update']) <= (self._settings['steps_until_target_network_update'] - (self._settings['steps_until_target_network_update']/10)))
                            ):
                            predicted_result_states__ = self._fd.predict_batch(states=_states, actions=_actions)
                            cost = self._pol.trainDyna(predicted_states=predicted_result_states__, actions=_actions, rewards=_rewards, result_states=_result_states, falls=_falls)
                            
                            if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
                                print("Performing Dyna Update, loss: ", cost)
                            # print("Updated params: ", self._pol.getNetworkParameters()[0][0][0])
                    
                    t1 = time.time()
                    if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['debug']):
                        sim_time_ = datetime.timedelta(seconds=(t1-t0))
                        print ("FD training complete in " + str(sim_time_) + " seconds")

                if (self._settings['train_actor']):
                    t1 = time.time()
                    if ( 'use_multiple_policy_updates' in self._settings and 
                         ( self._settings['use_multiple_policy_updates'] == True) ):
                        for i in range(self._settings['critic_updates_per_actor_update']):
                        
                            if ( ('anneal_on_policy' in self._settings) and self._settings['anneal_on_policy'] and False):
                                _states, _actions, _result_states, _rewards, _falls, _advantage, exp_actions__ = self._expBuff.get_exporation_action_batch(self._settings["batch_size"])
                            else:
                                _states, _actions, _result_states, _rewards, _falls, _advantage, exp_actions__ = self._expBuff.get_batch(self._settings["batch_size"])
                            
                            cost_ = self._pol.trainActor(states=_states, actions=_actions, rewards=_rewards, result_states=_result_states, falls=_falls, 
                                                         advantage=_advantage, exp_actions=exp_actions__, forwardDynamicsModel=self._fd)
                    else:
                        cost_ = self._pol.trainActor(states=_states, actions=_actions, rewards=_rewards, result_states=_result_states, falls=_falls, 
                                                     advantage=_advantage, exp_actions=exp_actions__, forwardDynamicsModel=self._fd)
                    t1 = time.time()
                    if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['debug']):
                        sim_time_ = datetime.timedelta(seconds=(t1-t0))
                        print ("Policy training complete in " + str(sim_time_) + " seconds")
                dynamicsLoss = 0 
                               
            ## Update scaling values
            ### Updating the scaling values after the update(s) will help make things more accurate
            if ('keep_running_mean_std_for_scaling' in self._settings and (self._settings["keep_running_mean_std_for_scaling"])):
                self.getExperience()._updateScaling()
                self.setStateBounds(self.getExperience().getStateBounds())
                self.setActionBounds(self.getExperience().getActionBounds())
                self.setRewardBounds(self.getExperience().getRewardBounds())
                if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['debug']):
                    print("Learner, Scaling State params: ", self.getStateBounds())
                    print("Learner, Scaling Action params: ", self.getActionBounds())
                    print("Learner, Scaling Reward params: ", self.getRewardBounds())
        else: ## Off-policy
            
            for update in range(self._settings['training_updates_per_sim_action']): ## Even more training options...
                t0 = time.time()
                for i in range(self._settings['critic_updates_per_actor_update']):
                    
                    if ( 'give_mbae_actions_to_critic' in self._settings and 
                         (self._settings['give_mbae_actions_to_critic'] == False)):
                        # if ( np.random.random() >= self._settings['model_based_action_omega']):
                        _states, _actions, _result_states, _rewards, _falls, _G_ts, _exp_actions = self._expBuff.getNonMBAEBatch(value_function_batch_size)
                        loss = self._pol.trainCritic(states=_states, actions=_actions, rewards=_rewards, result_states=_result_states, falls=_falls)
                        _states, _actions, _result_states, _rewards, _falls, _G_ts, _exp_actions = self._expBuff.get_batch(value_function_batch_size)
                    else:
                        _states, _actions, _result_states, _rewards, _falls, _G_ts, _exp_actions = self._expBuff.get_batch(value_function_batch_size)
                        loss = self._pol.trainCritic(states=_states, actions=_actions, rewards=_rewards, result_states=_result_states, falls=_falls)
                        if ('rebatch_data' in self._settings 
                            and (self._settings['rebatch_data'] == True)
                            ):
                            _states, _actions, _result_states, _rewards, _falls, _G_ts, _exp_actions = self._expBuff.get_batch(value_function_batch_size)

                    if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
                        print("Critic loss: ", loss)
                    if not np.isfinite(cost) or (cost > 500) :
                        numpy.set_printoptions(threshold=numpy.nan)
                        print ("States: " + str(_states) + " ResultsStates: " + str(_result_states) + " Rewards: " + str(_rewards) + " Actions: " + str(_actions))
                        print ("Training cost is Odd: ", cost)
                        
                t1 = time.time()
                if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['debug']):
                    sim_time_ = datetime.timedelta(seconds=(t1-t0))
                    print ("Crtic training complete in " + str(sim_time_) + " seconds")
                    
                if (self._settings['train_actor']):
                    t1 = time.time()
                    cost_ = self._pol.trainActor(states=_states, actions=_actions, rewards=_rewards, result_states=_result_states, falls=_falls, advantage=_advantage, exp_actions=_exp_actions, forwardDynamicsModel=self._fd)
                    t1 = time.time()
                    if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['debug']):
                        sim_time_ = datetime.timedelta(seconds=(t1-t0))
                        print ("Policy training complete in " + str(sim_time_) + " seconds")
                dynamicsLoss = 0 
                if (self._settings['train_forward_dynamics']):
                    t1 = time.time()
                    if ( 'keep_seperate_fd_exp_buffer' in self._settings and (self._settings['keep_seperate_fd_exp_buffer'])):
                        # print ("Using seperate (off-policy) exp mem for FD model")
                        _states, _actions, _result_states, _rewards, _falls, _G_ts, _exp_actions = self.getFDExperience().get_batch(value_function_batch_size)
                        
                    dynamicsLoss = self._fd.train(states=_states, actions=_actions, result_states=_result_states, rewards=_rewards)
                    if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
                        print ("Forward Dynamics Loss: ", dynamicsLoss)
                    
                    if ( 'give_mbae_actions_to_critic' in self._settings and 
                         (self._settings['give_mbae_actions_to_critic'] == False)):
                        ### perform a Q like update
                        actions____ = self._pol.predict_batch(states=_result_states) ### I think these could have noise added to them.
                        predicted_result_states__ = self._fd.predict_batch(states=_result_states, actions=actions____)
                        rewards____ = self._fd.predict_reward_batch(states=_result_states, actions=actions____)
                        loss = self._pol.trainCritic(states=_result_states, actions=actions____, rewards=rewards____, result_states=predicted_result_states__, falls=_falls)
                        
                    if (self._settings['train_critic_on_fd_output'] and 
                        (( self._pol.numUpdates() % self._settings['dyna_update_lag_steps']) == 0) and 
                        ( ( self._pol.numUpdates() %  self._settings['steps_until_target_network_update']) >= (self._settings['steps_until_target_network_update']/10)) and
                        ( ( self._pol.numUpdates() %  self._settings['steps_until_target_network_update']) <= (self._settings['steps_until_target_network_update'] - (self._settings['steps_until_target_network_update']/10)))
                        ):
                        
                        predicted_result_states__ = self._fd.predict_batch(states=_states, actions=_actions)
                        cost = self._pol.trainDyna(predicted_states=predicted_result_states__, actions=_actions, rewards=_rewards, result_states=_result_states, falls=_falls)
                        if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['train']):
                            print ( "Dyna training cost: ", cost)
                        if not np.isfinite(cost) or (cost > 500) :
                            numpy.set_printoptions(threshold=numpy.nan)
                            print ("States: " + str(_states) + " ResultsStates: " + str(_result_states) + " Rewards: " + str(_rewards) + " Actions: " + str(_actions))
                            print ("Training cost is Odd: ", cost)
                    
                    t1 = time.time()
                    if (self._settings["print_levels"][self._settings["print_level"]] >= self._settings["print_levels"]['debug']):
                        sim_time_ = datetime.timedelta(seconds=(t1-t0))
                        print ("FD training complete in " + str(sim_time_) + " seconds")
                            
            # import lasagne
            # val_params = lasagne.layers.helper.get_all_param_values(self._pol.getModel().getCriticNetwork())
            # pol_params = lasagne.layers.helper.get_all_param_values(self._pol.getModel().getActorNetwork())
            # fd_params = lasagne.layers.helper.get_all_param_values(self._fd.getModel().getForwardDynamicsNetwork())
            # print ("Learning Agent: Model pointers: val, ", self._pol.getModel(), " poli, ", self._pol.getModel(),  " fd, ", self._fd.getModel())
            # print("pol first layer params: ", pol_params[1])
            # print("val first layer params: ", val_params[1])
            # print("fd first layer params: ", fd_params[1])               
                            
        if self._useLock:
            self._accesLock.release()
        return (cost, dynamicsLoss) 
    
    def predict(self, state, evaluation_=False, p=None, sim_index=None, bootstrapping=False):
        if self._useLock:
            self._accesLock.acquire()
        act = self._pol.predict(state, evaluation_=evaluation_, p=p, sim_index=sim_index, bootstrapping=bootstrapping)
        if self._useLock:
            self._accesLock.release()
        return act
    
    def predict_std(self, state, evaluation_=False):
        if self._useLock:
            self._accesLock.acquire()
        std = self._pol.predict_std(state)
        if self._useLock:
            self._accesLock.release()
        return std
    
    def predictWithDropout(self, state):
        if self._useLock:
            self._accesLock.acquire()
        act = self._pol.predictWithDropout(state)
        if self._useLock:
            self._accesLock.release()
        return act
    
    def predictNextState(self, state, action):
        return self._fd.predict(state, action)
    
    def q_value(self, state):
        if self._useLock:
            self._accesLock.acquire()
        q = self._pol.q_value(state)
        if self._useLock:
            self._accesLock.release()
        return q
    
    def q_values(self, state):
        if self._useLock:
            self._accesLock.acquire()
        q = self._pol.q_values(state)
        if self._useLock:
            self._accesLock.release()
        return q
    
    def bellman_error(self, state, action, reward, result_state, fall):
        if self._useLock:
            self._accesLock.acquire()
        err = self._pol.bellman_error(state, action, reward, result_state, fall)
        if self._useLock:
            self._accesLock.release()
        return err
        
    def initEpoch(self, exp_):
        if (not (self._sampler == None ) ):
            self._sampler.initEpoch(exp_)
    
    def setSampler(self, sampler):
        self._sampler = sampler
    def getSampler(self):
        return self._sampler
    
    def setEnvironment(self, exp):
        if (not self._sampler == None ):
            self._sampler.setEnvironment(exp)
            
    def getStateBounds(self):
        return self.getPolicy().getStateBounds()
    def getActionBounds(self):
        return self.getPolicy().getActionBounds()
    def getRewardBounds(self):
        return self.getPolicy().getRewardBounds()
    
    def setStateBounds(self, bounds):
        self.getPolicy().setStateBounds(bounds)
        if (self._settings['train_forward_dynamics']):
            self.getForwardDynamics().setStateBounds(bounds)
    def setActionBounds(self, bounds):
        self.getPolicy().setActionBounds(bounds)
        if (self._settings['train_forward_dynamics']):
            self.getForwardDynamics().setActionBounds(bounds)
    def setRewardBounds(self, bounds):
        self.getPolicy().setRewardBounds(bounds)
        if (self._settings['train_forward_dynamics']):
            self.getForwardDynamics().setRewardBounds(bounds)

import copy
# class LearningWorker(threading.Thread):
class LearningWorker(Process):
    def __init__(self, input_exp_queue, agent, random_seed_=23):
        super(LearningWorker, self).__init__()
        self._input_queue= input_exp_queue
        # self._namespace = namespace
        self._agent = agent
        self._process_random_seed = random_seed_
        
    def setLearningNamespace(self, learningNamespace):    
        self._learningNamespace = learningNamespace
    
    # @profile(precision=5)    
    def run(self):
        _profile=False
        print ('Worker started')
        if ( _profile ):
            import cProfile, pstats, io
            pr = cProfile.Profile()
            pr.enable()
        np.random.seed(self._process_random_seed)
        if (self._agent._settings['on_policy']):
            self._agent._expBuff.clear()
        # do some initialization here
        step_ = 0
        iterations_=0
        # self._agent._expBuff = self.__learningNamespace.experience
        while True:
            tmp = self._input_queue.get()
            if tmp == None:
                break
            #if len(tmp) == 6:
                # self._input_queue.put(tmp)
            if tmp == "clear":
                if (self._agent._settings["print_levels"][self._agent._settings["print_level"]] >= self._agent._settings["print_levels"]['train']):
                    print ("Clearing exp memory")
                self._agent._expBuff.clear()
                continue
            #    continue # don't learn from eval tuples
            # (state_, action, reward, resultState, fall, G_t) = tmp
            # print ("Learner Size of state input Queue: " + str(self._input_queue.qsize()))
            # self._agent._expBuff = self.__learningNamespace.experience
            if self._agent._settings['action_space_continuous']:
                self._agent._expBuff.insertTuple(tmp)
                if ( 'keep_seperate_fd_exp_buffer' in self._agent._settings and (self._agent._settings['keep_seperate_fd_exp_buffer'])):
                    self._agent._expBuff_FD.insertTuple(tmp)
                # print ("Experience buffer size: " + str(self.__learningNamespace.experience.samples()))
                # print ("Reward Scale: ", self._agent._reward_bounds)
                # print ("Reward Scale Model: ", self._agent._pol.getRewardBounds())
            else:
                self._agent._expBuff.insertTuple(tmp)
                if ( 'keep_seperate_fd_exp_buffer' in self._agent._settings and (self._agent._settings['keep_seperate_fd_exp_buffer'])):
                    self._agent._expBuff_FD.insertTuple(tmp)
            # print ("Learning agent experience size: " + str(self._agent._expBuff.samples()))
            step_ += 1
            if self._agent._expBuff.samples() > self._agent._settings["batch_size"] and ((step_ >= self._agent._settings['sim_action_per_training_update']) ):
                __states, __actions, __result_states, __rewards, __falls, __G_ts, __exp_actions = self._agent._expBuff.get_batch(self._agent._settings["batch_size"])
                # print ("States: " + str(__states) + " ResultsStates: " + str(__result_states) + " Rewards: " + str(__rewards) + " Actions: " + str(__actions))
                (cost, dynamicsLoss) = self._agent.train(_states=__states, _actions=__actions, _rewards=__rewards, _result_states=__result_states, _falls=__falls)
                # print ("Master Agent Running training step, cost: " + str(cost) + " PID " + str(os.getpid()))
                # print ("Updated parameters: " + str(self._agent._pol.getNetworkParameters()[3]))
                if not np.isfinite(cost):
                    print ("States: " + str(__states) + " ResultsStates: " + str(__result_states) + " Rewards: " + str(__rewards) + " Actions: " + str(__actions))
                    print ("Training cost is Nan: ", cost)
                    sys.exit()
                # if (step_ % 10) == 0: # to help speed things up
                # self._learningNamespace.agentPoly = self._agent.getPolicy().getNetworkParameters()
                data = (self._agent._expBuff, self._agent.getPolicy().getNetworkParameters())
                if (self._agent._settings['train_forward_dynamics']):
                    # self._learningNamespace.forwardNN = self._agent.getForwardDynamics().getNetworkParameters()
                    data = (self._agent._expBuff, self._agent.getPolicy().getNetworkParameters(), self._agent.getForwardDynamics().getNetworkParameters())
                    if ( 'keep_seperate_fd_exp_buffer' in self._agent._settings and (self._agent._settings['keep_seperate_fd_exp_buffer'])):
                        data = (self._agent._expBuff, self._agent.getPolicy().getNetworkParameters(), self._agent.getForwardDynamics().getNetworkParameters(), self._agent._expBuff_FD)
                # self._learningNamespace.experience = self._agent._expBuff
                self._agent.setStateBounds(self._agent.getExperience().getStateBounds())
                self._agent.setActionBounds(self._agent.getExperience().getActionBounds())
                self._agent.setRewardBounds(self._agent.getExperience().getRewardBounds())
                # if (self._agent._settings["print_levels"][self._agent._settings["print_level"]] >= self._agent._settings["print_levels"]['train']):
                    # print("Learner, Scaling State params: ", self._agent.getStateBounds())
                    # print("Learner, Scaling Action params: ", self._agent.getActionBounds())
                    # print("Learner, Scaling Reward params: ", self._agent.getRewardBounds())
                ## put and do not block
                try:
                    # print ("Sending network params:")
                    if (not (self._output_message_queue.full())):
                        self._output_message_queue.put(data, False)
                    else:
                        ## Pull out (discard) an old one
                        self._output_message_queue.get(False)
                        self._output_message_queue.put(data, False)
                except Exception as inst:
                    if (self._agent._settings["print_levels"][self._agent._settings["print_level"]] >= self._agent._settings["print_levels"]['train']):
                        print ("LearningAgent: output model parameter message queue full: ", self._output_message_queue.qsize())
                step_=0
            iterations_+=1
            # print ("Done one update:")
        self._output_message_queue.close()
        self._output_message_queue.cancel_join_thread()
        print ("Learning Worker Complete:")
        
        if ( _profile ):
            pr.disable()
            f = open('x.prof', 'a')
            pstats.Stats(pr).sort_stats('time').print_stats()
            pstats.Stats(pr, stream=f).sort_stats('time').print_stats()
            f.close()
            
        return
        
    def updateExperience(self, experience):
        self._agent._expBuff = experience
        
    def setMasterAgentMessageQueue(self, queue):
        self._output_message_queue = queue
        
    # @profile(precision=5)  
    def updateModel(self):
        print ("Updating model to: ", self._learningNamespace.model)
        old_poli = self._agent.getPolicy()
        self._agent.setPolicy(self._learningNamespace.model)
        del old_poli
        