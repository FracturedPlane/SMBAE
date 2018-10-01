"""
"""
import numpy as np
import math
from sim.SimInterface import SimInterface 
import copy 
# import scipy.integrate as integrate
# import matplotlib.animation as animation

from model.ModelUtil import getOptimalAction, getMBAEAction


class ParticleSimEnv(SimInterface):

    def __init__(self, exp, settings):
        #------------------------------------------------------------
        # set up initial state
        super(ParticleSimEnv,self).__init__(exp, settings)

    def generateValidation(self, data, epoch):
        self.getEnvironment().generateValidationEnvironmentSample(epoch)
    
    def generateEnvironmentSample(self):
        self.getEnvironment().generateEnvironmentSample()
        
    def getEvaluationData(self):
        return self.getEnvironment().getEvaluationData()
    
    def getActor(self):
        return self._exp
    
    def finish(self):
        self._exp.finish()
    
    def getState(self):
        # state = np.array(self._exp.getState())
        state_ = np.array(self._exp.getState())
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
    
    def updateViz(self, actor, agent, directory, p=1.0):
        if (self.getSettings()['shouldRender']):
            U = []
            V = []
            Q = []
            U_mbae = []
            V_mbae = []
            R_mbae = []
            U_fd = []
            V_fd = []
            R_fd = []
            
            s_length = len(self.getSettings()['state_bounds'][0])
            ## This is a sampled grid in 2D
            (X,Y) = self.getEnvironment().getStateSamples()
            for x_,y_ in zip(X,Y):
                for x,y in zip(x_,y_):
                    ## Policy action
                    state_ = np.array([[x,y] + ([0]*(s_length-2))])
                    action1 = agent.predict(state_)[0]
                    action1_cp = copy.deepcopy(action1)
                    # print ("Action copy: ", action1_cp)
                    next_state_true_ = state_ + action1_cp
                    action1 = action1[:2]
                    # action1 = getOptimalAction(agent.getForwardDynamics(), agent.getPolicy(), state_)
                    ## normalize
                    ## action1 = action1/(np.sqrt((action1*action1).sum(axis=0)))
                    U.append(action1[0])
                    V.append(action1[1])
                    v = agent.q_value(state_)[0]
                    Q.append(v)
                    if (self.getSettings()['train_forward_dynamics']):
                        (action_, value_diff) = getOptimalAction(agent.getForwardDynamics(),
                                                                  agent.getPolicy(), state_, 
                                                                  action_lr=self.getSettings()['action_learning_rate']*p)[:2]
                        action_ = action_[0]                                  
                        # action_ = getMBAEAction(agent.getForwardDynamics(), agent.getPolicy(), state_)
                        ### How to change this action...
                        action_ = (action_[:2] - (action1_cp[:2]))
                        if ('use_stochastic_forward_dynamics' in self.getSettings() and 
                            (self.getSettings()['use_stochastic_forward_dynamics'] == "dropout")):
                            # print("Getting fd dropout sample:")
                            next_state = agent.getForwardDynamics().predictWithDropout(state_, [action1_cp])
                        else:
                            next_state = agent.getForwardDynamics().predict(state_, [action1_cp])
                        # print ("next_state: ", next_state)
                        fd_error_ = (next_state - next_state_true_)[0]
                        # print ("forward_dynamics error: ", action_)
                        action_ = action_/(np.sqrt((action_*action_).sum(axis=0)))
                        # action_ = action_ - action1
                        if ( np.all(np.isfinite(action_))):
                            U_mbae.append(action_[0])
                            V_mbae.append(action_[1])
                            U_fd.append(fd_error_[0])
                            V_fd.append(fd_error_[1])
                            # r = agent.getForwardDynamics().predict_reward(state_, np.array(action1_cp))
                            # print ("Predicted reward: ", r)
                            R_mbae.append(value_diff[0])
                            R_fd.append(value_diff[0])
                        else:
                            U_mbae.append(0.0)
                            V_mbae.append(0.0)
                            U_fd.append(0.0)
                            V_fd.append(0.0)
                            # r = agent.getForwardDynamics().predict_reward(state_, np.array(action1_cp))
                            # print ("Predicted reward: ", r)
                            R_mbae.append(0.0)
                            R_fd.append(0.0)
                            
            U = np.array(U)
            V = np.array(V)
            Q = np.array(Q)
            U_mbae = np.array(U_mbae)
            V_mbae = np.array(V_mbae)
            R_mbae = np.array(R_mbae)
            U_fd = np.array(U_fd)
            V_fd = np.array(V_fd)
            R_fd = np.array(R_fd)
            if (self.getSettings()['print_level'] == 'debug'):
                print( "U: ", U)
                print( "V: ", V)
                print( "Q: ", Q)
            self.getEnvironment().updatePolicy(U, V, Q)
            if (self.getSettings()['train_forward_dynamics']):
                if (self.getSettings()['print_level'] == 'debug'):
                    print( "U_mbae: ", U_mbae)
                    print( "V_mbae: ", V_mbae)
                    print( "R_mbae: ", R_mbae)
                self.getEnvironment().updateMBAE(U_mbae, V_mbae, R_mbae)
                self.getEnvironment().updateFD(U_fd, V_fd, R_fd)
            self.getEnvironment().saveVisual(directory+"/navAgent")
        
