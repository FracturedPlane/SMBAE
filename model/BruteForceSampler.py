import numpy as np
import lasagne
import sys
sys.path.append('../')
from model.ModelUtil import *

# For debugging
# theano.config.mode='FAST_COMPILE'
from model.Sampler import Sampler

class BruteForceSampler(Sampler):
    """
        This model using a forward dynamics network to compute the reward directly
    """
    def __init__(self):

        super(BruteForceSampler,self).__init__()
        
    def sampleModel(self, model, forwardDynamics, current_state):
        self._samples = []
        self._bestSample=[[0],[-10000000]]
        # print ("Suggested Action: " + str(action) + " for state: " + str(current_state))
        samples = self.generateSamples(self._pol._action_bounds, num_samples=5)
        for sample in samples:
            pa = sample
            prediction = forwardDynamics.predict(state=current_state, action=pa)
            y = reward(current_state, prediction)
            # y = self._game._reward(self._game._computeHeight(i+current_state[1]))
            # print (pa, y)
            # self._samples.append([[i],[y]])
            if y > self._bestSample[1][0]:
                self._bestSample[1][0] = y
                self._bestSample[0] = pa
