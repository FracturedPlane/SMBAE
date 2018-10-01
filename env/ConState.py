#!/usr/bin/env python
""" generated source for module ConState """
# 
#  * ConState.java
#  *
#  * Created on June 11, 2007, 12:05 AM
#  *
#  * To change this template, choose Tools | Template Manager
#  * and open the template in the editor.
#  
# 
#  *
#  * @author Stel-l
#  
class ConState(object):
    """ generated source for class ConState """
    # /////////////////////////////////////////////////
    #  FILE:      controller.h
    #  CONTAINS:  defs for FSM controller
    # /////////////////////////////////////////////////
    num = int()

    #  absolute state number
    localNum = int()

    #  local state number
    th = [None]*Controller.MaxJoints

    #  target angles
    thd = [None]*Controller.MaxJoints

    #  coeff for d
    thdd = [None]*Controller.MaxJoints

    #  coeff for d_dot
    next = int()

    #  next state
    timeFlag = bool()

    #  TRUE for time-based transitions
    leftStance = bool()

    #  TRUE if this is a state standing on left foot
    poseStance = bool()

    #  TRUE is this is an absolute pose state
    transTime = float()

    #  transition time
    sensorNum = int()

    #  transition sensor number
    #  Creates a new instance of ConState 
    def __init__(self):
        """ generated source for method __init__ """
        self.num = -1

    def setThThDThDD(self, index, t, tD, tDD):
        """ generated source for method setThThDThDD """
        self.th[index] = t
        self.thd[index] = tD
        self.thdd[index] = tDD

