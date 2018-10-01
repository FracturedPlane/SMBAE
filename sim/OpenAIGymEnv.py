"""
"""
import numpy as np
import math
from sim.SimInterface import SimInterface 
import copy 
# import scipy.integrate as integrate
# import matplotlib.animation as animation
import gym
from gym import wrappers
from gym import envs
# import roboschool
from OpenGL import GL
print(envs.registry.all())

from model.ModelUtil import getOptimalAction, getMBAEAction


class OpenAIGymEnv(SimInterface):

    def __init__(self, exp, settings):
        #------------------------------------------------------------
        # set up initial state
        super(OpenAIGymEnv,self).__init__(exp, settings)
        self._previous_observation=None
        self._end_of_episode=False
        
        ## Should print the type of actions space, continuous/discrete, how many parameters
        print(self.getEnvironment().action_space)
        ## Should print the type of state space, continuous/discrete, how many parameters
        print(self.getEnvironment().observation_space)
        """
        if ( settings['sim_config_file'] == 'RoboschoolHopper-v1'):
            self._state_param_mask = [  True ,  False       ,  False      ,  True ,  False        ,
                                        True,  False        ,  True,  True,  True,
                                        True,  True,  True ,  True,  True]
        else:
            self._state_param_mask = [   True] * len(settings['state_bounds'][0])
        """
    def init(self):
        # self.getEnvironment().init()
        self._previous_observation = self.getEnvironment().reset()
        self._end_of_episode = False
            
    def initEpoch(self):
        self._previous_observation = self.getEnvironment().reset()
        self._end_of_episode = False
        
    def endOfEpoch(self):
        return self._end_of_episode

    def finish(self):   
        self._exp.finish()
    
    def generateValidation(self, data, epoch):
        pass
        # self.initEpoch()
    
    def generateEnvironmentSample(self):
        pass
        # self.initEpoch()
        
    def getEvaluationData(self):
        return self.getEnvironment().getEvaluationData()
    
    def finish(self):
        # self._exp.finish()
        pass
        
    def step(self, action):
        action_ = np.array(action)
        if (self.getSettings()['render']):
            self.getEnvironment().render()
        observation, reward, done, info = self.getEnvironment().step(action_[0])
        # print ("observation: ", observation)
        self._end_of_episode = done
        self._previous_observation = observation
        return reward
    
    def getState(self):
        # state = np.array(self._exp.getState())
        # observation, reward, done, info = env.step(action)
        # self._previous_observation = observation
        
        state_ = np.array(self._previous_observation)
        """
        ### Because some of the state parameters from the sim are always the same number.
        state_idx=0
        state__=[]
        for i in range(len(self._previous_observation)): 
            if (self._state_param_mask[i] == True):
                state__.append(state_[i] )
        """
        state = np.array(state_)
        
        state = np.reshape(state, (-1, len(state_)))
        
        return state
    
    def setState(self, st):
        self._agent = st
        self._box.state[0,0] = st[0]
        self._box.state[0,1] = st[1]
        
    def visualizeNextState(self, next_state_, action):
        _t_length = self.getEnvironment()._game_settings['num_terrain_samples']
        terrain = next_state_[:_t_length]
        terrain_dx = next_state_[_t_length]
        terrain_dy = next_state_[_t_length+1]
        character_features = next_state_[_t_length+2:]
        self.getEnvironment().visualizeNextState(terrain, action, terrain_dx)  

    def generateValidationEnvironmentSample(self, numb):
        pass
    
    def needUpdatedAction(self):
        return True
    
    def updateAction(self, action):
        self.step(action)

    def update(self):
        pass
    
    def display(self):
        pass
    