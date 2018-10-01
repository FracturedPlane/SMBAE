import sys
import math
from actor.ActorInterface import ActorInterface
import numpy as np
import dill

class TerrainRLHLCActor(ActorInterface):
    
    def __init__(self, settings_, experience):
        super(TerrainRLHLCActor,self).__init__(settings_, experience)
        self._target_vel_weight=self._settings["target_velocity_decay"]
        self._target_vel = self._settings["target_velocity"]
        """
        self._default_action = [ 0.082500, 0.251474, 2.099796, 0.000000, 0.000000, -0.097252, -0.993935, 0.273527, 0.221481, 1.100288, -3.076833, 0.180141, -0.176967, 0.310372, -1.642646, -0.406771, 1.240827, -1.773369, -0.508333, -0.170533, -0.063421, -2.091676, -1.418455, -1.242994, -0.262842, 0.453321, -0.170533, -0.366870, -1.494344, 0.794701, -1.408623, 0.655703, 0.634434]
        self._param_mask = [    False,        True,        True,        False,        False,    
        True,        True,        True,        True,        True,        True,        True,    
        True,        True,        True,        True,        True,        True,        True,    
        False,        True,        True,        True,        True,        True,        True,    
        False,        True,        True,        True,        True,        True,        True]
        """
        
        print ("Loading pre compiled network")
        file_name=self._settings['llc_policy_model_path']
        f = open(file_name, 'rb')
        model = dill.load(f)
        # model.setSettings(settings_)
        f.close()
        self._llc_policy = model
        self._sim = None
    # @profile(precision=5)
    
    def updateAction(self, sim, action_):
        action_ = np.array(action_, dtype='float64')
        sim.getEnvironment().updateAction(action_)
        
    def updateLLCAction(self, sim, action_):
        action_ = np.array(action_, dtype='float64')
        sim.getEnvironment().updateLLCAction(action_)
        
    def act(self, exp, action_, bootstrapping=False):
        samp = self.getActionParams(action_)
        
        reward = self.actContinuous(exp, samp, bootstrapping)
        return reward
    
    # @profile(precision=5)
    def actContinuous(self, sim, action_, bootstrapping=False):
        self._sim = sim
        # Actor should be FIRST here
        # print "Action: " + str(action_)
        # reward = exp.getEnvironment().act(action_)
        # mask some parameters
        ## Need to make sure this is an vector of doubles
        action_ = np.array(action_, dtype='float64')
        action_idx=0
        action__=[]
        vel_sum=0
        """
        for i in range(len(self._default_action)): # because the use of parameters can be switched on and off.
            if (self._param_mask[i] == True):
                action__.append(action_[action_idx] )
                action_idx+=1
            else:
                action__.append(self._default_action[i])
        action_=action__
        """
        sim.updateAction(action_)
        updates_=0
        stumble_count=0
        torque_sum=0
        tmp_reward_sum=0
        # print("Acting")
        while (not sim.needUpdatedAction() and (updates_ < 100)
               and (not sim.getEnvironment().agentHasFallen())
               ):
            # sim.updateAction(action_)
            self.updateActor(sim, action_)
            updates_+=1
            # print("Update #: ", updates_)
        if (updates_ == 0): #Something went wrong...
            print("There were no updates... This is bad")
            return 0.0
        
        # reward_ = tmp_reward_sum/float(updates_)  
        reward_ = sim.getEnvironment().calcReward() 
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
        
    def updateActor(self, sim, action_):
        # llc_state = sim.getState()[:,self._settings['num_terrain_features']:]
        llc_state = sim.getLLCState()
        # print("LLC state: ", llc_state)
        # action__ = np.array([[action_[0], action_[1], 0.0, action_[2], action_[3], 0.0, action_[4]]])
        action__ = np.array([[action_[4], action_[0], 0.0, action_[1], action_[2], 0.0, action_[3]]])
        # print ("llc pose state: ", llc_state.shape, repr(llc_state))
        # print ("hlc action: ", action__.shape, repr(action__))
        # llc_state = np.concatenate((llc_state, action__), axis=1)
        llc_state[:,-7:] = action__
        # print ("llc_state: ", llc_state.shape, llc_state)
        llc_action = self._llc_policy.predict(llc_state)
        # print("llc_action: ", llc_action.shape, llc_action)
        sim.updateLLCAction(llc_action)
        sim.update()
        if (self._settings["shouldRender"]):
            sim.display()
        # rw_ = sim.getEnvironment().calcReward()
        # tmp_reward_sum=tmp_reward_sum + rw_
        # print("reward: ", rw_, " reward_sum:, ", tmp_reward_sum)
        