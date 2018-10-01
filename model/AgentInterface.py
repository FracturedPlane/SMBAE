"""
    An interface class for Agents to be used in the system.

"""

class AgentInterface(object):
    
    def __init__(self, n_in, n_out, state_bounds, action_bounds, reward_bound, settings_):
        AgentInterface.setActionBounds(self, action_bounds) 
        AgentInterface.setStateBounds(self, state_bounds) 
        AgentInterface.setRewardBounds(self, reward_bound)    
        self._state_length = n_in
        self._action_length = n_out
        self._settings = settings_ 
    
    def train(self, states, actions, rewards, result_states):
        pass
    
    def predict(self, state):
        pass
    
    def q_value(self, state):
        pass
    
    def bellman_error(self, state, action, reward, result_state):
        pass
    
    def setStateBounds(self, bounds):
        self._state_bounds = bounds

    def setActionBounds(self, bounds):
        self._action_bounds = bounds
    
    def setRewardBounds(self, bounds):
        self._reward_bounds = bounds
        
    def initEpoch(self, exp):
        pass
    
    def init(state_length, action_length, state_bounds, action_bounds, actor, exp, settings):
        pass
    
    def getSettings(self):
        return self._settings
    
    def setEnvironment(self, exp):
        pass
