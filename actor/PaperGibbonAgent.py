import sys
import os
import math
from actor.ActorInterface import ActorInterface
import numpy as np
from model.ModelUtil import clampAction 
from model.ModelUtil import _scale_reward 
from model.ModelUtil import randomExporation, randomUniformExporation, reward_smoother


class PaperGibbonAgent(ActorInterface):
    
    def __init__(self, settings_, experience):
        super(PaperGibbonAgent,self).__init__(settings_, experience)
        self._target_vel_weight=self._settings["target_velocity_decay"]
        self._target_vel = self._settings["target_velocity"]
    
    
    # @profile(precision=5)
    def actContinuous(self, exp, action_, bootstrapping=False):
        import characterSim
        # print ("Executing action")
        action_ = np.array(action_, dtype='float64')
        # Actor should be FIRST here
        # print "Action: " + str(action_)
        action = characterSim.Action()
        # samp = paramSampler.generateRandomSample()
        action.setParams(action_)
        reward = exp.getEnvironment().act(action)
        reward = reward_smoother(reward, self._settings, self._target_vel_weight)
        if ( not np.isfinite(reward)):
            print ("Found some bad reward: ", reward, " for action: ", action_)
            reward = 0
        self._reward_sum = self._reward_sum + reward
        return reward
    
    def init(self):
        self._agent.init()
        self._reward_sum=0
        
    def initEpoch(self):
        # self._agent.initEpoch()
        self._reward_sum=0

    def getEvaluationData(self):
        return self._reward_sum
    
    def hasNotFallen(self, exp):
        return not exp.getEnvironment().hasFallen()
    