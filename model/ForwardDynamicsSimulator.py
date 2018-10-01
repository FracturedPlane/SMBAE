import theano
from theano import tensor as T
import numpy as np
import lasagne
import sys
sys.path.append('../')
sys.path.append("../characterSimAdapter/")
from model.ModelUtil import *

from model.AgentInterface import AgentInterface

class ForwardDynamicsSimulator(AgentInterface):
    
    def __init__(self, state_length, action_length, state_bounds, action_bounds, actor, exp, settings):
        # import characterSim
        super(ForwardDynamicsSimulator,self).__init__(state_length, action_length, state_bounds, action_bounds, 0, settings)
        self._exp = exp ## Only used to pull some data from
        self._reward=0
        
        self._actor = actor
        # self.initSim(settings)
        self._sim = exp 
        
    def initSim(self, settings):
        from util.SimulationUtil import validateSettings, createEnvironment, createRLAgent, createActor
        sim = createEnvironment(str(settings["forwardDynamics_config_file"]), str(settings['environment_type']), settings)
        ## The real simulator that is used for predictions
        
        self._sim = sim
        
        self._sim.getActor().init()   
        self._sim.init()
        
    def setActor(self, actor):
        self._actor = actor
    def setEnvironment(self, exp):
        self._exp = exp
        self._sim = exp
    """    
    def setEnvironment(self, sim):
        self._sim = sim # The real simulator that is used for predictions
       """ 
    def train(self, states, actions,  result_states):
        return 0
    
    def bellman_error(self, state, action, result_state):
        return 0
    
    def endOfEpoch(self):
        return self._sim.getEnvironment().endOfEpoch()
    
    def setSimState(self, state_):
        self._sim.setSimState(state_)
        
    def initEpoch(self, exp):
        print ("Init FD epoch: ")
        self._sim.getActor().initEpoch()
        # self._sim.getEnvironment().clear()
        """
        for anchor in range(self.getSettings()['max_epoch_length']):
            # print (_anchor)
            anchor_ = exp.getEnvironment().getAnchor(anchor)
            self._sim.getEnvironment().addAnchor(anchor_.getX(), anchor_.getY(), anchor_.getZ())
        """
        simState = self._exp.getEnvironment().getSimState()
        self._sim.setSimState(simState)
        # self._sim.generateEnvironmentSample()
        self._sim.initEpoch()
        print ("Number of anchors is " + str(self._sim.getEnvironment().numAnchors()))
        
    def predict(self, state, action):
        """
            This is the normal prediction using the reduced state
        """
        # state = norm_state(state, self._state_bounds)
        # action = norm_action(action, self._action_bounds)
        # print ("Action: " + str(action))
        # print ("State: " + str(state._id))
        # self._exp.getEnvironment().setState(state)
        # current_state = self._exp._exp.getEnvironment().getSimInterface().getController().getControllerStateVector()
        c_state = self._sim.getSimState()
        reward = self._actor.actContinuous(self._sim,action)
        # print ("State: " + str(state.getParams()))
        state_ = self._sim.getState()
        # print ("State: " + str.(state))
        # restore previous state
        # self._exp._exp.getEnvironment().getSimInterface().getController().setControllerStateVector(current_state)
        self._sim.setSimState(c_state)
        # print ("State: " + str(state))
        return state_
    
    def _predict(self, state__c, action):
        """
            This particular prediction sets the internal state of the simulator before executing the action
            state__c: is some kind of global state of the simulator
        """
        # state = norm_state(state, self._state_bounds)
        # action = norm_action(action, self._action_bounds)
        # print ("Action: " + str(action))
        # print ("State: " + str(state._id))
        # state__ = self._sim.getSimState()
        
        self._sim.setSimState(state__c)
        # current_state = self._exp._exp.getEnvironment().getSimInterface().getController().getControllerStateVector()
        # c_state = self._sim.getEnvironment().getState()
        reward = self._actor.actContinuous(self._sim,action)
        # print("_Predict reward: ", reward)
        # print ("State: " + str(state.getParams()))
        state___ = self._sim.getSimState()
        # print ("State: " + str(state))
        # restore previous state
        # self._exp._exp.getEnvironment().getSimInterface().getController().setControllerStateVector(current_state)
        # self._sim.setSimState(state__)
        # print ("State: " + str(state))
        return (state___, reward)
