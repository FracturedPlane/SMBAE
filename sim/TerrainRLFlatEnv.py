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


class TerrainRLFlatEnv(TerrainRLEnv):

    def __init__(self, exp, settings):
        #------------------------------------------------------------
        # set up initial state
        super(TerrainRLFlatEnv,self).__init__(exp, settings)

    
    def getState(self):
        """
            Want just the character state at the end.
        """
        state_ = self.getEnvironment().getState()
        # print ("state_: ", state_)
        state = np.array(state_)[200:]
        state = np.reshape(state, (-1, len(state_)-200))
        # state = np.array(state_)
        # state = np.reshape(state, (-1, len(state_)))
        return state
    
    def generateValidationEnvironmentSample(self, epoch):
        pass
    def generateEnvironmentSample(self):
        pass
        # self._exp.getEnvironment().generateEnvironmentSample()
    