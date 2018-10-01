import sys
import os
import math
from actor.ActorInterface import ActorInterface
import numpy as np
from model.ModelUtil import clampAction, clampActionWarn
from model.ModelUtil import _scale_reward 
from model.ModelUtil import randomExporation, randomUniformExporation, reward_smoother

class MocapImitationActor(ActorInterface):
    
    def __init__(self, settings_, experience):
        super(MocapImitationActor,self).__init__(settings_, experience)
        self._target_vel_weight=self._settings["target_velocity_decay"]
        self._target_vel = self._settings["target_velocity"]
        self._target_lean = 0.0
        self._target_torque = 0
        self._target_root_height = 1.02
        self._target_hand_pos = 0.0
        self._action_bounds = np.array(self._settings["action_bounds"], dtype=float)
        
    def updateAction(self, sim, action_):
        action_ = np.array(action_, dtype='float64')
        (action_, outside_bounds) = clampActionWarn(action_, self._action_bounds)
        sim.getEnvironment().updateAction(action_)
        
    # @profile(precision=5)
    def act(self, exp, action_, bootstrapping=False):
        samp = self.getActionParams(action_)
        
        reward = self.actContinuous(exp, samp, bootstrapping)
        # self._reward_sum = self._reward_sum + reward
        return reward
    
    # @profile(precision=5)
    def actContinuous(self, exp, action_, bootstrapping=False):
        # Actor should be FIRST here
        # print "Action: " + str(action_)
        ## Need to make sure this is an vector of doubles
        action_ = np.array(action_, dtype='float64')
        (action_, outside_bounds) = clampActionWarn(action_, self._action_bounds)
        position_root = np.array(exp.getEnvironment().getActor().getStateEuler()[0:][:3])
        # print ("Relative Right arm pos: ", right_hand_pos-position_root)
        exp.getEnvironment().updateAction(action_)
        steps_ = 0
        vel_error_sum= float(0)
        torque_sum= float(0)
        position_error_sum = float(0)
        pose_error_sum = float(0)
        while (not exp.getEnvironment().needUpdatedAction() or (steps_ == 0)):
            exp.getEnvironment().update()
            simData = exp.getEnvironment().getActor().getSimData()
            position_root = exp.getEnvironment().getActor().getStateEuler()[0:3]
            vel_root = exp.getEnvironment().getActor().getStateEuler()[6:9]
            if (self._settings["print_level"]== 'debug_details'):
                print ("avgSpeed: ", simData.avgSpeed, " grabed speed: ", vel_root)
            vel_error_sum += math.fabs(self._target_vel - vel_root[0])
            torque_sum += math.fabs( self._target_torque - simData.avgTorque)
            
            position_error_sum += math.fabs(self._target_root_height - position_root[1])
            
            pose_error_sum += exp.getEnvironment().calcImitationReward()
            
            steps_ += 1
        averageSpeedError = vel_error_sum / steps_
        averageTorque = torque_sum / steps_
        averagePositionError = position_error_sum / steps_
        averagePoseError = pose_error_sum / steps_
        
        
             
        # averageSpeed = exp.getEnvironment().act(action_)
        # print ("averageSpeed: ", averageSpeed)
        # if (averageSpeed < 0.0):
        #     return 0.0
        # if (exp.getEnvironment().agentHasFallen()):
        #     return 0
        
        # orientation = exp.getEnvironment().getActor().getStateEuler()[3:][:3]
        # position_root = exp.getEnvironment().getActor().getStateEuler()[0:][:3]
        # print ("Pos: ", position_root)
        # print ("Orientation: ", orientation)
        ## Reward for going the desired velocity
        vel_diff = averageSpeedError
        if (self._settings["print_level"]== 'debug_details'):
            print ("vel_diff: ", vel_diff)
        # if ( self._settings["use_parameterized_control"] ):
        vel_bounds = self._settings['controller_parameter_settings']['velocity_bounds']
        vel_diff = _scale_reward([vel_diff], vel_bounds)[0]
        vel_reward = reward_smoother(vel_diff, self._settings, self._target_vel_weight)
        if (self._settings["print_level"]== 'debug_details'):
            print ("vel_reward: ", vel_reward)
        ## Rewarded for using less torque
        torque_diff = averageTorque
        if (self._settings["print_level"]== 'debug_details'):
            print ("torque_diff: ", torque_diff)
        _bounds = self._settings['controller_parameter_settings']['torque_bounds']
        torque_diff = _scale_reward([torque_diff], _bounds)[0]
        torque_reward = reward_smoother(torque_diff, self._settings, self._target_vel_weight)
        if (self._settings["print_level"]== 'debug_details'):
            print ("torque_reward: ", torque_reward)
        ## Rewarded for keeping the y height of the root at a specific height 
        root_height_diff = (averagePositionError)
        if (self._settings["print_level"]== 'debug_details'):
            print ("root_height_diff: ", root_height_diff)
        # if ( self._settings["use_parameterized_control"] ):
        root_height_bounds = self._settings['controller_parameter_settings']['root_height_bounds']
        root_height_diff = _scale_reward([root_height_diff], root_height_bounds)[0]
        root_height_reward = reward_smoother(root_height_diff, self._settings, self._target_vel_weight)
        if (self._settings["print_level"]== 'debug_details'):
            print ("root_height_reward: ", root_height_reward)
        
        pose_error = (averagePoseError)
        if (self._settings["print_level"]== 'debug_details'):
            print ("pose_error: ", pose_error)
        # if ( self._settings["use_parameterized_control"] ):
        pose_error_bounds = self._settings['controller_parameter_settings']['pose_error_bounds']
        pose_error_diff = _scale_reward([pose_error], pose_error_bounds)[0]
        pose_error_reward = reward_smoother(pose_error_diff, self._settings, self._target_vel_weight)
        if (self._settings["print_level"]== 'debug_details'):
            print ("pose_error_reward: ", pose_error_reward)
        
        reward = ( 
                  (vel_reward * self._settings['controller_reward_weights']['velocity']) 
                  + (torque_reward * self._settings['controller_reward_weights']['torque']) 
                  + ((root_height_reward) * self._settings['controller_reward_weights']['root_height'])
                  + ((pose_error_reward) * self._settings['controller_reward_weights']['pose_error'])
                  )# optimal is 0
        if (self._settings["print_level"]== 'debug_details'):
            print ("Reward: ", reward)
        
        self._reward_sum = self._reward_sum + reward
        return reward
    
        
        # print("New target Velocity: ", self._target_vel)
        # if ( self._settings["use_parameterized_control"] )
            
    def getControlParameters(self):
        # return [self._target_vel, self._target_root_height, self._target_lean, self._target_hand_pos]
        return [self._target_vel]
        
    def getEvaluationData(self):
        return self._reward_sum
    
    def hasNotFallen(self, exp):
        position_root = exp.getEnvironment().getActor().getStateEuler()[0:][:3]
        if ( exp.getEnvironment().agentHasFallen() or (position_root[1] < 0.0) ) :
            return 0
        else:
            return 1
        # return not exp.getEnvironment().agentHasFallen()
        
