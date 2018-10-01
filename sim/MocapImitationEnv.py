"""
"""
import numpy as np
import math
from sim.SimInterface import SimInterface
import sys
sys.path.append("../simbiconAdapter/")
from actor.DoNothingActor import DoNothingActor
import json
import copy
# import scipy.integrate as integrate
# import matplotlib.animation as animation

def reOrderMocapData(data, current_order, desired_order):
    """
    
    -------
    Parameters
    
    data: is a vector of doubles the makes up a pose
    current_order: is a list of dicts of the form
    [{"root_pos":
        {"start_index":1,
        "data_length":3}},
    {"root_rot":
        {"start_index":4,
        "data_length":4}},
    {"pelvis_torso":
        {"start_index":8,
        "data_length":4}},
    {"rHip":
        {"start_index":12,
        "data_length":4}},
    ...
    ]
    desired_order: is a list of strings that match the names of the joints in the current_order
    dicts
    """
    old_data = copy.deepcopy(data)
    current_index = 0
    for joint in desired_order:
        data[current_index:current_index+current_order[joint]['data_length']] = old_data[current_order[joint]['start_index']:(current_order[joint]['start_index']+current_order[joint]['data_length'])] 
        current_index = current_index + current_order[joint]['data_length']
        # print ("Current index: ", current_index, " data: ", data)
        
    return data

class MocapImitationEnv(SimInterface):

    def __init__(self, exp, settings):
        #------------------------------------------------------------
        # set up initial state
        super(MocapImitationEnv,self).__init__(exp, settings)
        self._action_dimension=3
        self._range = 5.0

    def initEpoch(self):
        self.getEnvironment().initEpoch()
        # self.getAgent().initEpoch()
        
    def init(self):
        import simbiconAdapter
        self.getEnvironment().init()
                ### Read motion file.
        motionFileName = self.getSettings()['motion_file']
        motionFile = open(motionFileName)
        motionFileData = json.load(motionFile)
        print ("motionFileData: " + str(json.dumps(motionFileData)))
        motionFile.close()
        motion = simbiconAdapter.Motion(len(motionFileData['Frames']), len(motionFileData['Frames'][0]))
        print( "There are: ", len(motionFileData['Frames']), " frames of length: ", len(motionFileData['Frames'][0]))
        desired_joint_order = ["knot", "root_pos", "root_rot", "pelvis_torso", "rHip", "lHip", "rKnee", "lKnee", "rAnkle", "lAnkle"]
        ## To switch the ordering of the joints, a hack to produce the walking step from the other foot as well
        # desired_joint_order = ["knot", "root_pos", "root_rot", "pelvis_torso", "lHip", "lKnee", "lAnkle", "rHip", "rKnee", "rAnkle"]
        motionData = []
        for rowIndex in range(len(motionFileData['Frames'])):
            row = motionFileData['Frames'][rowIndex]
            fixed_row = reOrderMocapData(row, motionFileData['joint_order'], desired_joint_order)
            print ("Row length: ", len(fixed_row), " data: ", fixed_row)
            motion.addFrame(fixed_row, rowIndex)
            motionData.append(fixed_row)
        
        print ("MotionData: ", motionData)
        motion.setFrameStep(0.0333);
        self.getEnvironment().setMotion(motion)
                                   
                                   
                                   
    def getEnvironment(self):
        return self._exp
    
    def getEvaluationData(self):
        return self.getEnvironment().getEvaluationData()
    
    def generateValidation(self, data, epoch):
        """
            Do nothing for now
        """
        pass
        # print (("Training on validation set: ", epoch, " Data: ", data))
        # self.getEnvironment().clear()
        # print (("Done clear"))
        # for i in range(len(data)):
            # data_ = data[i]
            # print (("Adding anchor: ", data_, " index ", i))
            # self.getEnvironment().addAnchor(data_[0], data_[1], data_[2])
            
        # print (("Done adding anchors"))

    def generateValidationEnvironmentSample(self, epoch):
        pass
    def generateEnvironmentSample(self):
        pass
        # self._exp.getEnvironment().generateEnvironmentSample()
        
    def updateAction(self, action_):
        # print("Simbicon updating action:")
        self.getActor().updateAction(self, action_)
    
    def needUpdatedAction(self):
        return self.getEnvironment().needUpdatedAction()
        
    def update(self):
        self.getEnvironment().update()
            
    def display(self):
        pass
        # self.getEnvironment().display()

    def finish(self):
        self._exp.finish()
    
    def getState(self):
        """
            I like the state in this shape, a row
        """
        state_ = list(self.getEnvironment().getState())
        state_.extend(list([self.getEnvironment().getImitationPhase()]))    
        state_.extend(list(self.getEnvironment().getImitationPose()))
        state = np.array(state_)
        state = np.reshape(state, (-1, len(state_)))
        if ( self._settings["use_parameterized_control"] ):
            state = np.append(state, [self.getActor().getControlParameters()], axis=1)
        return state
    
    def getControllerBackOnTrack(self):
        import characterSim
        """
            Push controller back into a good state space
        """
        pass
        
    def setTargetChoice(self, i):
        # need to find which target corresponds to this bin.
        pass
    
    
    def getStateFromSimState(self, simState):
        """
            Converts a detailed simulation state to a state better suited for learning
        """
        return self.getEnvironment().getStateFromSimState(simState)
    
    def getSimState(self):
        """
            Gets a more detailed state that can be used to re-initilize the state of the character back to this state later.
        """
        return self.getEnvironment().getSimState()
    
    def setSimState(self, state_):
        """
            Sets the state of the simulation to the given state
        """
        return self.getEnvironment().setSimState(state_)
        
