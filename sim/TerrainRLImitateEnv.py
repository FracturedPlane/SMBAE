"""
"""
import numpy as np
import math
from sim.TerrainRLEnv import TerrainRLEnv
import sys
from actor.DoNothingActor import DoNothingActor
# sys.path.append("../simbiconAdapter/")

# import scipy.integrate as integrate
# import matplotlib.animation as animation


class TerrainRLImitateEnv(TerrainRLEnv):

    def __init__(self, exp, settings):
        #------------------------------------------------------------
        # set up initial state
        super(TerrainRLImitateEnv,self).__init__(exp, settings)
        self._num_updates_since_last_action=0

    
    def getState(self):
        """
            Want just the character state at the end.
        """
        state_ = self.getEnvironment().getState()
        # print ("state_: ", state_)
        # state = np.array(state_)[200:]
        # state = np.reshape(state, (-1, len(state_)-200))
        state = np.array(state_)
        state = np.reshape(state, (-1, len(state_)))
        return state
    
    def update(self):
        for i in range(1):
            self.getEnvironment().update()
            self._num_updates_since_last_action+=1
        # self.getEnvironment().display()
            
    def updateAction(self, action_):
        
        self.getActor().updateAction(self, action_)
        self._num_updates_since_last_action = 0

    def needUpdatedAction(self):
        
        if ( self._num_updates_since_last_action >= 1):
            return True
        else:
            return False
        return 
            
    def generateValidationEnvironmentSample(self, epoch):
        pass
    def generateEnvironmentSample(self):
        pass
        # self._exp.getEnvironment().generateEnvironmentSample()