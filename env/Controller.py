#!/usr/bin/env python
""" generated source for module Controller """
# 
#  * Controller.java
#  *
#  * Created on June 10, 2007, 11:51 PM
#  *
#  * To change this template, choose Tools | Template Manager
#  * and open the template in the editor.
#  
# 
#  *
#  * @author Stel-l
#  
class Controller(object):
    """ generated source for class Controller """
    MaxStates = 100
    MaxJoints = 10
    MaxGroups = 30
    MaxJumps = 20
    NONE = -1
    state = [None]*MaxStates

    #  states, index-access by groups
    groups = [None]*MaxGroups

    #  groups
    nrStates = 0

    #  number of states
    nrGroups = 0

    #  number of groups
    kp = [None]*MaxJoints
    kd = [None]*MaxJoints
    targetLimit = [None]*2

    #  target joint angles min,max limits
    torqLimit = [None]*2

    #  torque min, max limits
    jointLimit = [None]*2

    #  joint angle limits
    fsmState = 0

    #  current state
    stateTime = 0

    #  time spent within state
    currentGroupNumber = 0
    desiredGroupNumber = 0

    #  lookup group by name
    def findGroup(self, name):
        """ generated source for method findGroup """
        return None

    #  advance FSM state
    def advance(self, b):
        """ generated source for method advance """
        transition = False
        s = self.state[self.fsmState]
        oldStance = s.leftStance
        if s.timeFlag and (self.stateTime > s.transTime):
            transition = True
        elif not s.timeFlag and (b.FootState[s.sensorNum] != 0):
            #  sensor-based transition ?
            transition = True
        if transition:
            self.stateTime = 0
            #  reset state time
            self.fsmState = s.next
            #  normal state transition to next state
            if self.currentGroupNumber != self.desiredGroupNumber:
                self.currentGroupNumber = self.desiredGroupNumber
                # find out the local group number
                self.fsmState = self.groups[self.currentGroupNumber].stateOffset + localfsmState
        newStance = self.state[self.fsmState].leftStance
        newStep = True if (newStance != oldStance) else False
        return newStep

    def __init__(self):
        """ generated source for method __init__ """
        i = 0
        while i < self.MaxJoints:
            self.kp[i] = 300
            self.kd[i] = 30
            self.torqLimit[0][i] = -1000
            self.torqLimit[1][i] = 1000
            self.targetLimit[0][i] = -float(Math.PI)
            self.targetLimit[1][i] = float(Math.PI)
            self.jointLimit[0][i] = -float(Math.PI)
            self.jointLimit[1][i] = float(Math.PI)
            i += 1
        self.jointLimit[0][1] = -1
        self.jointLimit[1][1] = 3
        self.jointLimit[0][3] = -1
        self.jointLimit[1][3] = 3
        self.jointLimit[0][2] = -3
        self.jointLimit[1][2] = -0.02
        self.jointLimit[0][4] = -3
        self.jointLimit[1][4] = -0.02
        self.targetLimit[0][1] = -0.4
        self.targetLimit[1][1] = 1.6
        self.targetLimit[0][3] = -0.4
        self.targetLimit[1][3] = 1.6
        self.nrStates = 0
        self.nrGroups = 0

    def addWalkingController(self):
        """ generated source for method addWalkingController """
        self.state[self.nrStates] = ConState()
        self.state[self.nrStates].num = self.nrStates
        self.state[self.nrStates].localNum = 0
        self.state[self.nrStates].next = self.nrStates + 1
        self.state[self.nrStates].timeFlag = True
        self.state[self.nrStates].leftStance = True
        self.state[self.nrStates].poseStance = False
        self.state[self.nrStates].transTime = 0.3
        self.state[self.nrStates].sensorNum = 0
        self.state[self.nrStates].setThThDThDD(0, 0, 0, 0)
        self.state[self.nrStates].setThThDThDD(1, 0.4, 0, 0.2)
        self.state[self.nrStates].setThThDThDD(2, -1.1, 0, 0)
        self.state[self.nrStates].setThThDThDD(3, 0, 0, 0)
        self.state[self.nrStates].setThThDThDD(4, -0.05, 0, 0)
        self.state[self.nrStates].setThThDThDD(5, 0.2, 0, 0)
        self.state[self.nrStates].setThThDThDD(6, 0.2, 0, 0)
        self.nrStates += 1
        self.state[self.nrStates] = ConState()
        self.state[self.nrStates].num = self.nrStates
        self.state[self.nrStates].localNum = 1
        self.state[self.nrStates].next = self.nrStates + 1
        self.state[self.nrStates].timeFlag = False
        self.state[self.nrStates].leftStance = True
        self.state[self.nrStates].poseStance = False
        self.state[self.nrStates].transTime = 0
        self.state[self.nrStates].sensorNum = 0
        self.state[self.nrStates].setThThDThDD(0, 0, 0, 0)
        self.state[self.nrStates].setThThDThDD(1, -0.7, 2.2, 0)
        self.state[self.nrStates].setThThDThDD(2, -0.05, 0, 0)
        self.state[self.nrStates].setThThDThDD(3, 0, 0, 0)
        self.state[self.nrStates].setThThDThDD(4, -0.1, 0, 0)
        self.state[self.nrStates].setThThDThDD(5, 0.2, 0, 0)
        self.state[self.nrStates].setThThDThDD(6, 0.2, 0, 0)
        self.nrStates += 1
        self.state[self.nrStates] = ConState()
        self.state[self.nrStates].num = self.nrStates
        self.state[self.nrStates].localNum = 2
        self.state[self.nrStates].next = self.nrStates + 1
        self.state[self.nrStates].timeFlag = True
        self.state[self.nrStates].leftStance = False
        self.state[self.nrStates].poseStance = False
        self.state[self.nrStates].transTime = 0.3
        self.state[self.nrStates].sensorNum = 0
        self.state[self.nrStates].setThThDThDD(0, 0, 0, 0)
        self.state[self.nrStates].setThThDThDD(1, 0, 0, 0)
        self.state[self.nrStates].setThThDThDD(2, -0.05, 0, 0)
        self.state[self.nrStates].setThThDThDD(3, 0.4, 0, 0.2)
        self.state[self.nrStates].setThThDThDD(4, -1.1, 0, 0)
        self.state[self.nrStates].setThThDThDD(5, 0.2, 0, 0)
        self.state[self.nrStates].setThThDThDD(6, 0.2, 0, 0)
        self.nrStates += 1
        self.state[self.nrStates] = ConState()
        self.state[self.nrStates].num = self.nrStates
        self.state[self.nrStates].localNum = 1
        self.state[self.nrStates].next = self.nrStates - 3
        self.state[self.nrStates].timeFlag = False
        self.state[self.nrStates].leftStance = False
        self.state[self.nrStates].poseStance = False
        self.state[self.nrStates].transTime = 0
        self.state[self.nrStates].sensorNum = 6
        self.state[self.nrStates].setThThDThDD(0, 0, 0, 0)
        self.state[self.nrStates].setThThDThDD(1, 0, 0, 0)
        self.state[self.nrStates].setThThDThDD(2, -0.1, 0, 0)
        self.state[self.nrStates].setThThDThDD(3, -0.7, 2.2, 0)
        self.state[self.nrStates].setThThDThDD(4, -0.05, 0, 0)
        self.state[self.nrStates].setThThDThDD(5, 0.2, 0, 0)
        self.state[self.nrStates].setThThDThDD(6, 0.2, 0, 0)
        self.nrStates += 1
        self.groups[self.nrGroups] = Group()
        self.groups[self.nrGroups].num = 0
        self.groups[self.nrGroups].stateOffset = self.nrStates - 4
        self.groups[self.nrGroups].nStates = 4
        self.nrGroups += 1

    def addRunningController(self):
        """ generated source for method addRunningController """
        self.state[self.nrStates] = ConState()
        self.state[self.nrStates].num = self.nrStates
        self.state[self.nrStates].localNum = 0
        self.state[self.nrStates].next = self.nrStates + 1
        self.state[self.nrStates].timeFlag = True
        self.state[self.nrStates].leftStance = True
        self.state[self.nrStates].poseStance = False
        self.state[self.nrStates].transTime = 0.21
        self.state[self.nrStates].sensorNum = 0
        self.state[self.nrStates].setThThDThDD(0, 0, 0, 0)
        self.state[self.nrStates].setThThDThDD(1, 0.8, 0, 0.2)
        self.state[self.nrStates].setThThDThDD(2, -1.84, 0, 0)
        self.state[self.nrStates].setThThDThDD(3, 0, 0, 0)
        self.state[self.nrStates].setThThDThDD(4, -0.05, 0, 0)
        self.state[self.nrStates].setThThDThDD(5, 0.2, 0, 0)
        self.state[self.nrStates].setThThDThDD(6, 0.27, 0, 0)
        self.nrStates += 1
        self.state[self.nrStates] = ConState()
        self.state[self.nrStates].num = self.nrStates
        self.state[self.nrStates].localNum = 1
        self.state[self.nrStates].next = self.nrStates + 1
        self.state[self.nrStates].timeFlag = True
        self.state[self.nrStates].leftStance = True
        self.state[self.nrStates].poseStance = False
        self.state[self.nrStates].transTime = 0
        self.state[self.nrStates].sensorNum = 0
        self.state[self.nrStates].setThThDThDD(0, -0.22, 0, 0)
        self.state[self.nrStates].setThThDThDD(1, 1.08, 0.0, 0.2)
        self.state[self.nrStates].setThThDThDD(2, -2.18, 0, 0)
        self.state[self.nrStates].setThThDThDD(3, 0, 0, 0)
        self.state[self.nrStates].setThThDThDD(4, -0.05, 0, 0)
        self.state[self.nrStates].setThThDThDD(5, 0.2, 0, 0)
        self.state[self.nrStates].setThThDThDD(6, 0.27, 0, 0)
        self.nrStates += 1
        self.state[self.nrStates] = ConState()
        self.state[self.nrStates].num = self.nrStates
        self.state[self.nrStates].localNum = 2
        self.state[self.nrStates].next = self.nrStates + 1
        self.state[self.nrStates].timeFlag = True
        self.state[self.nrStates].leftStance = False
        self.state[self.nrStates].poseStance = False
        self.state[self.nrStates].transTime = 0.21
        self.state[self.nrStates].sensorNum = 0
        self.state[self.nrStates].setThThDThDD(0, 0, 0, 0)
        self.state[self.nrStates].setThThDThDD(1, 0, 0, 0)
        self.state[self.nrStates].setThThDThDD(2, -0.05, 0, 0)
        self.state[self.nrStates].setThThDThDD(3, 0.8, 0, 0.2)
        self.state[self.nrStates].setThThDThDD(4, -1.84, 0, 0)
        self.state[self.nrStates].setThThDThDD(5, 0.27, 0, 0)
        self.state[self.nrStates].setThThDThDD(6, 0.2, 0, 0)
        self.nrStates += 1
        self.state[self.nrStates] = ConState()
        self.state[self.nrStates].num = self.nrStates
        self.state[self.nrStates].localNum = 1
        self.state[self.nrStates].next = self.nrStates - 3
        self.state[self.nrStates].timeFlag = True
        self.state[self.nrStates].leftStance = False
        self.state[self.nrStates].poseStance = False
        self.state[self.nrStates].transTime = 0
        self.state[self.nrStates].sensorNum = 6
        self.state[self.nrStates].setThThDThDD(0, -0.22, 0, 0)
        self.state[self.nrStates].setThThDThDD(1, 0, 0, 0)
        self.state[self.nrStates].setThThDThDD(2, -0.05, 0, 0)
        self.state[self.nrStates].setThThDThDD(3, 1.08, 0.0, 0.2)
        self.state[self.nrStates].setThThDThDD(4, -2.18, 0, 0)
        self.state[self.nrStates].setThThDThDD(5, 0.27, 0, 0)
        self.state[self.nrStates].setThThDThDD(6, 0.2, 0, 0)
        self.nrStates += 1
        self.groups[self.nrGroups] = Group()
        self.groups[self.nrGroups].num = 1
        self.groups[self.nrGroups].stateOffset = self.nrStates - 4
        self.groups[self.nrGroups].nStates = 4
        self.nrGroups += 1

    def addCrouchWalkController(self):
        """ generated source for method addCrouchWalkController """
        self.state[self.nrStates] = ConState()
        self.state[self.nrStates].num = self.nrStates
        self.state[self.nrStates].localNum = 0
        self.state[self.nrStates].next = self.nrStates + 1
        self.state[self.nrStates].timeFlag = True
        self.state[self.nrStates].leftStance = True
        self.state[self.nrStates].poseStance = False
        self.state[self.nrStates].transTime = 0.3
        self.state[self.nrStates].sensorNum = 0
        self.state[self.nrStates].setThThDThDD(0, -0.18, 0, 0)
        self.state[self.nrStates].setThThDThDD(1, 1.1, 0, 0.2)
        self.state[self.nrStates].setThThDThDD(2, -2.17, 0, 0)
        self.state[self.nrStates].setThThDThDD(3, 0, 0, 0)
        self.state[self.nrStates].setThThDThDD(4, -0.97, 0, 0)
        self.state[self.nrStates].setThThDThDD(5, 0.62, 0, 0)
        self.state[self.nrStates].setThThDThDD(6, 0.44, 0, 0)
        self.nrStates += 1
        self.state[self.nrStates] = ConState()
        self.state[self.nrStates].num = self.nrStates
        self.state[self.nrStates].localNum = 1
        self.state[self.nrStates].next = self.nrStates + 1
        self.state[self.nrStates].timeFlag = False
        self.state[self.nrStates].leftStance = True
        self.state[self.nrStates].poseStance = False
        self.state[self.nrStates].transTime = 0
        self.state[self.nrStates].sensorNum = 0
        self.state[self.nrStates].setThThDThDD(0, -0.25, 0, 0)
        self.state[self.nrStates].setThThDThDD(1, -0.7, 2.2, 0.0)
        self.state[self.nrStates].setThThDThDD(2, -0.05, 0, 0)
        self.state[self.nrStates].setThThDThDD(3, 0, 0, 0)
        self.state[self.nrStates].setThThDThDD(4, -0.92, 0, 0)
        self.state[self.nrStates].setThThDThDD(5, 0.2, 0, 0)
        self.state[self.nrStates].setThThDThDD(6, 0.44, 0, 0)
        self.nrStates += 1
        self.state[self.nrStates] = ConState()
        self.state[self.nrStates].num = self.nrStates
        self.state[self.nrStates].localNum = 2
        self.state[self.nrStates].next = self.nrStates + 1
        self.state[self.nrStates].timeFlag = True
        self.state[self.nrStates].leftStance = False
        self.state[self.nrStates].poseStance = False
        self.state[self.nrStates].transTime = 0.3
        self.state[self.nrStates].sensorNum = 0
        self.state[self.nrStates].setThThDThDD(0, -0.18, 0, 0)
        self.state[self.nrStates].setThThDThDD(1, 0, 0, 0)
        self.state[self.nrStates].setThThDThDD(2, -0.97, 0, 0)
        self.state[self.nrStates].setThThDThDD(3, 1.1, 0, 0.2)
        self.state[self.nrStates].setThThDThDD(4, -2.17, 0, 0)
        self.state[self.nrStates].setThThDThDD(5, 0.44, 0, 0)
        self.state[self.nrStates].setThThDThDD(6, 0.62, 0, 0)
        self.nrStates += 1
        self.state[self.nrStates] = ConState()
        self.state[self.nrStates].num = self.nrStates
        self.state[self.nrStates].localNum = 1
        self.state[self.nrStates].next = self.nrStates - 3
        self.state[self.nrStates].timeFlag = False
        self.state[self.nrStates].leftStance = False
        self.state[self.nrStates].poseStance = False
        self.state[self.nrStates].transTime = 0
        self.state[self.nrStates].sensorNum = 6
        self.state[self.nrStates].setThThDThDD(0, -0.25, 0, 0)
        self.state[self.nrStates].setThThDThDD(1, 0, 0, 0)
        self.state[self.nrStates].setThThDThDD(2, -0.92, 0, 0)
        self.state[self.nrStates].setThThDThDD(3, -0.7, 2.2, 0.0)
        self.state[self.nrStates].setThThDThDD(4, -0.05, 0, 0)
        self.state[self.nrStates].setThThDThDD(5, 0.44, 0, 0)
        self.state[self.nrStates].setThThDThDD(6, 0.2, 0, 0)
        self.nrStates += 1
        self.groups[self.nrGroups] = Group()
        self.groups[self.nrGroups].num = 2
        self.groups[self.nrGroups].stateOffset = self.nrStates - 4
        self.groups[self.nrGroups].nStates = 4
        self.nrGroups += 1

