import sys
sys.path.append("../characterSimAdapter/")
import math
import numpy as np

class ActorInterface(object):
    """
    _actions = [[0.6,-0.4,0.75],
                [0.6,-0.4,0.25],
                [0.6,-0.4,0.5],
                [0.8,-0.45,0.75],
                [0.8,-0.45,0.5],
                [0.8,-0.45,0.25],
                [0.2,-0.5,0.75],
                [0.2,-0.5,0.5],
                [0.2,-0.5,0.25]]
    """         
    
    def __init__(self, settings_, experience):
        self._settings = settings_
        self._actions = np.array(self._settings["discrete_actions"])
        self._experience = experience
        self._reward_sum=0
        self._agent = None
        self._action_bounds = np.array(self._settings["action_bounds"], dtype=float)
        
        
    def init(self):
        self._reward_sum=0
        
    def initEpoch(self):
        self._reward_sum=0
        
    def hasNotFallen(self, exp):
        return 1
        
    def getActionParams(self, index):
        return self._actions[index]
    
    # @profile(precision=5)
    def act(self, exp, action_, bootstrapping=False):
        import characterSim
        samp = self.getActionParams(action_)
        
        reward = self.actContinuous(exp, samp, bootstrapping=bootstrapping)
        
        return reward
    
    # @profile(precision=5)
    def actContinuous(self, exp, action_, bootstrapping=False):
        import characterSim
        action_ = np.array(action_, dtype='float64')
        # Actor should be FIRST here
        # print "Action: " + str(action_)
        action = characterSim.Action()
        # samp = paramSampler.generateRandomSample()
        action.setParams(action_)
        reward = exp.getEnvironment().act(action)
        if ( not np.isfinite(reward)):
            print ("Found some bad reward: ", reward, " for action: ", action_)
        self._reward_sum = self._reward_sum + reward
        return reward
    
    def getEvaluationData(self):
        return self._reward_sum
    
    def setAgent(self, agent):
        self._agent = agent
        
    def updateActor(self, sim, action_):
        pass
        
        
        
def reward(previous_state, current_state):
    index_start=3
    v1_x = previous_state[index_start]
    v3_x = current_state[index_start]
    v2_x = previous_state[index_start+2]
    v5_x = previous_state[index_start+4]
    v4_x = current_state[index_start+2]
    v8_x = v5_x - v2_x
    v7_x = v4_x - v8_x
    dx = v7_x - v3_x 
    
    index_start=4
    v1_y = previous_state[index_start]
    v3_y = current_state[index_start]
    v2_y = previous_state[index_start+2]
    v5_y = previous_state[index_start+4]
    v4_y = current_state[index_start+2]
    v8_y = v5_y - v2_y
    v7_y = v4_y - v8_y
    dy = v7_y - v3_y
    
    d = math.sqrt((dx*dx)+(dy*dy))
    return -d
    
def travelDist(previous_state, current_state):
    index_start=3
    a0x_in_s0 = previous_state[index_start+2]
    a1x_in_s0 = previous_state[index_start+4]
    s0x_in_s0 = previous_state[index_start]
    s0x_in_s1 = current_state[index_start]
    a0x_in_s1 = current_state[index_start+2]
    dx = (a1x_in_s0-s0x_in_s0) - (a0x_in_s1-s0x_in_s1) 
    
    index_start=4
    a0y_in_s0 = previous_state[index_start+2]
    a1y_in_s0 = previous_state[index_start+4]
    s0y_in_s0 = previous_state[index_start]
    s0y_in_s1 = current_state[index_start]
    a0y_in_s1 = current_state[index_start+2]
    dy = (a1y_in_s0-s0y_in_s0) - (a0y_in_s1-s0y_in_s1) 
    
    d = math.sqrt((dx*dx)+(dy*dy))
    return d
    
def armDistFromTarget(previous_state):
    index_start=3
    a0x_in_s0 = previous_state[index_start+2]
    s0x_in_s0 = previous_state[index_start]
    x = (a0x_in_s0-s0x_in_s0)
    
    index_start=4
    a0y_in_s0 = previous_state[index_start+2]
    s0y_in_s0 = previous_state[index_start]
    y = (a0y_in_s0-s0y_in_s0)
    
    return np.array([x, y, 0])
    
def anchorDist(a0, a1, anchors):
    return np.array(anchors[a1]) - np.array(anchors[a0])
    
def goalDistance(current_state, anchors, goal_anchor):
    """
        Computes the Euclidean distance between the current state and the goal anchor 
    """ 
    current_state_ = current_state.getParams()
    # distance to current target anchor
    armDist = armDistFromTarget(current_state_)
    
    # distance from current target to goal
    anchor_dist = anchorDist(current_state.getID(), goal_anchor, anchors) 
    
    # total distance
    dx = armDist[0] + anchor_dist[0]
    dy = armDist[1] + anchor_dist[1]
    
    d = math.sqrt((dx*dx)+(dy*dy))
    return d
            
    
    
