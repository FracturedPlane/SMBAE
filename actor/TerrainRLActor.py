import sys
import math
from actor.ActorInterface import ActorInterface
import numpy as np

class TerrainRLActor(ActorInterface):
    
    def __init__(self, settings_, experience):
        super(TerrainRLActor,self).__init__(settings_, experience)
        self._target_vel_weight=self._settings["target_velocity_decay"]
        self._target_vel = self._settings["target_velocity"]
        self._default_action = [ 0.082500, 0.251474, 2.099796, 0.000000, 0.000000, -0.097252, -0.993935, 0.273527, 0.221481, 1.100288, -3.076833, 0.180141, -0.176967, 0.310372, -1.642646, -0.406771, 1.240827, -1.773369, -0.508333, -0.170533, -0.063421, -2.091676, -1.418455, -1.242994, -0.262842, 0.453321, -0.170533, -0.366870, -1.494344, 0.794701, -1.408623, 0.655703, 0.634434]
        self._param_mask = [    False,        True,        True,        False,        False,    
        True,        True,        True,        True,        True,        True,        True,    
        True,        True,        True,        True,        True,        True,        True,    
        False,        True,        True,        True,        True,        True,        True,    
        False,        True,        True,        True,        True,        True,        True]
        
    # @profile(precision=5)
    def act(self, exp, action_, bootstrapping=False):
        samp = self.getActionParams(action_)
        
        reward = self.actContinuous(exp, samp, bootstrapping)
        return reward
    
    def updateAction(self, sim, action_):
        action_ = np.array(action_, dtype='float64')
        action_idx=0
        action__=[]
        vel_sum=0
        for i in range(len(self._default_action)): # because the use of parameters can be switched on and off.
            if (self._param_mask[i] == True):
                action__.append(action_[action_idx] )
                action_idx+=1
            else:
                action__.append(self._default_action[i])
        action_= np.array(action__, dtype='float64')
        sim.getEnvironment().updateAction(action_)
    
    # @profile(precision=5)
    def actContinuous(self, sim, action_, bootstrapping=False):
        # Actor should be FIRST here
        # print "Action: " + str(action_)
        # reward = exp.getEnvironment().act(action_)
        # mask some parameters
        ## Need to make sure this is an vector of doubles
        action_ = np.array(action_, dtype='float64')
        action_idx=0
        action__=[]
        vel_sum=0
        for i in range(len(self._default_action)): # because the use of parameters can be switched on and off.
            if (self._param_mask[i] == True):
                action__.append(action_[action_idx] )
                action_idx+=1
            else:
                action__.append(self._default_action[i])
        action_= np.array(action__, dtype='float64')
        sim.getEnvironment().act(action_)
        updates_=0
        stumble_count=0
        torque_sum=0
        while (not sim.getEnvironment().needUpdatedAction() and (updates_ < 500)
               and (not sim.getEnvironment().agentHasFallen())
               ):
            sim.getEnvironment().update()
            if (self._settings["shouldRender"]):
                sim.display()
            vel_sum += sim.getEnvironment().calcVelocity()
            updates_+=1
            if ( sim.getEnvironment().hasStumbled() ):
                stumble_count+=1
            torque_sum += sim.getEnvironment().jointTorque()
            # print("Update #: ", updates_)
        """    
        if (updates_ == 1):
            print("Action update did not go well....")
        else:
            print("Action update Okay!")
        """    
        if (updates_ == 0): #Something went wrong...
            return 0.0
        
        torque_reward = torque_sum/float(updates_)
        
        avg_stumble = float(stumble_count) / float(updates_);
        stumble_gamma = 10.0;
        stumble_reward = 1.0 / (1 + stumble_gamma * avg_stumble);
            
        averageSpeed = vel_sum / float(updates_)
        vel_diff = self._target_vel - averageSpeed
        vel_reward_ = math.exp((vel_diff*vel_diff)*self._target_vel_weight) # optimal is 0
        # reward_ = sim.getEnvironment().calcReward()   
        # print ("averageSpeed: ", averageSpeed)
        # print ("vel_reward_: ", vel_reward_, " stumble: ", stumble_reward, " torque: ", torque_reward)
        reward_ = ((vel_reward_ * 0.8) + 
                   (stumble_reward * 0.2 )+
                   (torque_reward * -0.1)
                   )
        self._reward_sum = self._reward_sum + reward_
        # print ("Reward: ", reward_)
        return reward_
    
    def getEvaluationData(self):
        return self._reward_sum
    
    def hasNotFallen(self, sim):
        if ( sim.getEnvironment().agentHasFallen() ) :
            return 0
        else:
            return 1
        # return not exp.getEnvironment().agentHasFallen()
        
        