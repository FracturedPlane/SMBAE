import sys
import math
from actor.ActorInterface import ActorInterface

class ImitationActor(ActorInterface):
    
    def __init__(self, settings_, experience):
        super(ImitationActor,self).__init__(settings_, experience)
        self._target_vel_weight=self._settings["target_velocity_decay"]
        self._target_vel = self._settings["target_velocity"]
        
    # @profile(precision=5)
    def act(self, exp, action_, bootstrapping=False):
        samp = self.getActionParams(action_)
        
        reward = self.actContinuous(exp, samp_, bootstrapping)
        return reward
    
    # @profile(precision=5)
    def actContinuous(self, exp, action_, bootstrapping=False):
        # Actor should be FIRST here
        # print "Action: " + str(action_)
        balh = exp.getEnvironment().act(action_)
        reward = exp.getEnvironment().calcReward()
        self._reward_sum = self._reward_sum + reward
        return reward
    
    def getEvaluationData(self):
        return self._reward_sum
    
    def hasNotFallen(self, exp):
        if ( exp.getEnvironment().agentHasFallen() ) :
            return 0
        else:
            return 1
        # return not exp.getEnvironment().agentHasFallen()