import theano
from theano import tensor as T
import numpy as np
import lasagne
import sys
sys.path.append('../')
sys.path.append("../characterSimAdapter/")
from model.ModelUtil import *

from model.AgentInterface import AgentInterface
from model.ForwardDynamicsSimulator import ForwardDynamicsSimulator
from multiprocessing import Queue, Process
# from util.SimulationUtil import createEnvironment

class ForwardDynamicsSimulatorProcess(Process):

    def __init__(self, state_length, action_length, state_bounds, action_bounds, actor, exp, settings, input_state_queue,
                 outpout_state_queue):
        # import characterSim
        super(ForwardDynamicsSimulatorProcess, self).__init__()
        # super(ForwardDynamicsSimulatorProcess,self).__init__(state_length, action_length, state_bounds, action_bounds, 0, settings)
        self._input_queue= input_state_queue
        self._output_state_queue = outpout_state_queue
        # self._exp = exp # Only used to pull some data from
        # self._c = characterSim.Configuration(str(settings['forwardDynamics_config_file']))
        # c = characterSim.Configuration("../data/epsilon0Config.ini")
        
        # this is the process that selects which game to play
        # sim = characterSim.Experiment(self._c)
        self._settings = settings
        
        self._actor = actor
        
        # self._sim = sim # The real simulator that is used for predictions
    
    def setActor(self, actor):
        self._actor = actor
    def setEnvironment(self, sim):
        self._sim = sim # The real simulator that is used for predictions
        
    def run(self):
        from util.SimulationUtil import createEnvironment
        # import characterSim
        # sim = characterSim.Experiment(self._c)
        sim = createEnvironment(str(self._settings["forwardDynamics_config_file"]), str(self._settings['environment_type']), self._settings, render=False)
        sim.getActor().init()   
        sim.init()
        self._sim = sim # The real simulator that is used for predictions
        print ('ForwardDynamicsSimulatorProcess started')
        # do some initialization here
        step_ = 0
        while True:
            tmp = self._input_queue.get()
            if tmp == None:
                break
            elif (tmp[0] == 'init'):
                print ("Init Epoch in FDSP:")
                self._sim.getActor().initEpoch()
                self._sim.getEnvironment().clear()
                for anchor_ in tmp[1]:
                    # print (_anchor)
                    # anchor_ = self._exp.getEnvironment().getAnchor(anchor)
                    self._sim.getEnvironment().addAnchor(anchor_[0], anchor_[1], anchor_[2])
                # simState = self._exp.getSimState()
                # self._sim.setSimState(simState)
                # self._sim.generateEnvironmentSample()
                self._sim.initEpoch()
                print ("Number of anchors is " + str(self._sim.getEnvironment().numAnchors()))
                
            else:
                (state__c, action) = tmp
                ## get current state of sim
                state__ = self._sim.getSimState()
                # print ("Sampling State:" + str(state__c))
                # print ("State: " + str(state_c) + " sim " + str(self._sim.getEnvironment()))
                ## Set sim to given state
                self._sim.setSimState(state__c)
                # print ("State: " + str(state_c) + " Action: " + str(action))
                reward = self._actor.actContinuous(self._sim,action)
                # print ("Reward: ", reward)
                # print ("State: " + str(state.getParams()))
                ## Get new state after action
                state_ = self._sim.getSimState()
                ## Set back to original state (maybe not needed)...
                self._sim.setSimState(state__)
                # characterSim.State(current_state_copy.getID(), current_state_copy.getParams())
                
                self._output_state_queue.put((reward, state_))
            

class ForwardDynamicsSimulatorParallel(ForwardDynamicsSimulator):
    
    def __init__(self, state_length, action_length, state_bounds, action_bounds, actor, exp, settings):

        super(ForwardDynamicsSimulatorParallel,self).__init__(state_length, action_length, state_bounds, action_bounds, actor, exp, settings)
        self._exp = exp # Only used to pull some data from
        self._reward=0
        
        self._output_state_queue = Queue(1)
        self._input_state_queue = Queue(1)
        
    def init(self, state_length, action_length, state_bounds, action_bounds, actor, exp, settings):
        
        self._worker = ForwardDynamicsSimulatorProcess(state_length, action_length, state_bounds, action_bounds, actor, exp, settings,
                                                       self._output_state_queue, self._input_state_queue)
        
        self._worker.start()
        
   
    def initEpoch(self, exp_):
        print ("Init FD epoch: ")
        # self._sim.getActor().initEpoch()
        # self._sim.getEnvironment().clear()
        anchors = []
        for anchor in range(exp_.getEnvironment().numAnchors()):
            # print (_anchor)
            anchor_ = exp_.getEnvironment().getAnchor(anchor)
            anchors.append([anchor_.getX(), anchor_.getY(), anchor_.getZ()])
            # self._sim.getEnvironment().addAnchor(anchor_.getX(), anchor_.getY(), anchor_.getZ())
            
        
        # simState = self._exp.getSimState()
        # self._sim.setSimState(simState)
        # self._sim.initEpoch()
        self._output_state_queue.put(('init',anchors)) 

    def _predict(self, state__c, action):
        """
            This particular prediction sets the internal state of the simulator before executing the action
            state__c: is some kind of global state of the simulator
        """
        # state = norm_state(state, self._state_bounds)
        # action = norm_action(action, self._action_bounds)
        # print ("Action: " + str(action))
        # print ("State: " + str(state._id))
        # state__ = self._sim.getEnvironment().getSimState()
        
        # self._sim.getEnvironment().setSimState(state__c)
        # current_state = self._exp._exp.getEnvironment().getSimInterface().getController().getControllerStateVector()
        # c_state = self._sim.getEnvironment().getState()
        # reward = self._actor.actContinuous(self._sim,action)
        ## Send in current state
        self._output_state_queue.put((state__c, action))
        ## get back out next state
        (reward, state___) = self._input_state_queue.get()
        # print("_Predict reward: ", reward)
        # print ("State: " + str(state.getParams()))
        # state___ = self._sim.getEnvironment().getSimState()
        # print ("State: " + str(state))
        # restore previous state
        # self._exp._exp.getEnvironment().getSimInterface().getController().setControllerStateVector(current_state)
        ## Set the state to that of the simm on the other process, (if, for example, we want to check endOfEpoch())
        self._sim.getEnvironment().setSimState(state___)
        # print ("State: " + str(state))
        return (state___, reward)
        """
        ## Send in current state
        self._output_state_queue.put([state__c.getID(), state__c.getParams(), action])
        ## get back out next state
        state__ = self._input_state_queue.get()
        state__ = characterSim.State(state__[0], state__[1])
        return state__
        """
