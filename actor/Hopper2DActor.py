import sys
import os
import math
from actor.ActorInterface import ActorInterface
import numpy as np
from model.ModelUtil import clampAction, clampActionWarn
from model.ModelUtil import _scale_reward 
from model.ModelUtil import randomExporation, randomUniformExporation, reward_smoother

class Hopper2DActor(ActorInterface):
    
    def __init__(self, settings_, experience):
        super(Hopper2DActor,self).__init__(settings_, experience)
        self._target_vel_weight=self._settings["target_velocity_decay"]
        self._target_vel = self._settings["target_velocity"]
        self._target_lean = 0.0
        self._target_torque = 0
        self._target_root_height = 1.02
        self._target_hand_pos = 0.0
        
    def updateAction(self, sim, action_):
        action_ = np.array(action_, dtype='float64')
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
        # right_hand_pos = np.array(exp.getEnvironment().getActor().getLinkPosition("rLowerarm"))
        position_root = np.array(exp.getEnvironment().getActor().getStateEuler()[0:][:3])
        # print ("Relative Right arm pos: ", right_hand_pos-position_root)
        exp.getEnvironment().updateAction(action_)
        steps_ = 0
        vel_sum= float(0)
        torque_sum= float(0)
        pitch_sum = float(0)
        position_sum = float(0)
        right_hand_x_sum = float(0)
        dist_x = float(0)
        while (not exp.getEnvironment().needUpdatedAction() or (steps_ == 0)):
            exp.getEnvironment().update()
            simData = exp.getEnvironment().getActor().getSimData()
            position_root_ = exp.getEnvironment().getActor().getStateEuler()[0:][:3]
            dist_x = dist_x + (position_root_[0] - position_root[0])
            # print ("avgSpeed: ", simData.avgSpeed)
            vel_sum += simData.avgSpeed
            print ("I don't calculate the velocity properly, should use character::getCOMVelocity() instead")
            torque_sum += math.fabs( self._target_torque - simData.avgTorque)
            
            # orientation = exp.getEnvironment().getActor().getStateEuler()[3:][:3]
            
            position_sum += math.fabs(self._target_root_height - position_root_[1])
            
            steps_ += 1
        
        # print("vel_x:", dist_x*30.0)
        averageSpeed = vel_sum / steps_
        averageTorque = torque_sum / steps_
        averagePosition = position_sum / steps_
        # averageSpeed = exp.getEnvironment().act(action_)
        # print ("root position: ", position_root)
        # print ("averageSpeed: ", averageSpeed)
        # if (averageSpeed < 0.0):
        #     return 0.0
        if (exp.getEnvironment().agentHasFallen()):
            return 0
        
        # orientation = exp.getEnvironment().getActor().getStateEuler()[3:][:3]
        # position_root = exp.getEnvironment().getActor().getStateEuler()[0:][:3]
        # print ("Pos: ", position_root)
        # print ("Orientation: ", orientation)
        ## Reward for going the desired velocity
        vel_diff = math.fabs(self._target_vel - averageSpeed)
        if (self._settings["print_level"]== 'debug'):
            print ("vel_diff: ", vel_diff)
        # if ( self._settings["use_parameterized_control"] ):
        vel_bounds = self._settings['controller_parameter_settings']['velocity_bounds']
        vel_diff = _scale_reward([vel_diff], vel_bounds)[0]
        if (self._settings["print_level"]== 'debug'):
            print ("vel_diff: ", vel_diff)
        vel_reward = reward_smoother(vel_diff, self._settings, self._target_vel_weight)
        ## Rewarded for using less torque
        torque_diff = averageTorque
        _bounds = self._settings['controller_parameter_settings']['torque_bounds']
        torque_diff = _scale_reward([torque_diff], _bounds)[0]
        torque_reward = reward_smoother(torque_diff, self._settings, self._target_vel_weight)
        ## Rewarded for keeping the y height of the root at a specific height 
        root_height_diff = (averagePosition)
        if (self._settings["print_level"]== 'debug'):
            print ("root_height_diff: ", root_height_diff)
        # if ( self._settings["use_parameterized_control"] ):
        root_height_bounds = self._settings['controller_parameter_settings']['root_height_bounds']
        root_height_diff = _scale_reward([root_height_diff], root_height_bounds)[0]
        if (self._settings["print_level"]== 'debug'):
            print ("root_height_diff: ", root_height_diff)
        root_height_reward = reward_smoother(root_height_diff, self._settings, self._target_vel_weight)
        
        # print ("vel reward: ", vel_reward, " torque reward: ", torque_reward )
        reward = ( 
                  (vel_reward * self._settings['controller_reward_weights']['velocity'])
                  # + (torque_reward * self._settings['controller_reward_weights']['torque']) +
                  # (lean_reward * self._settings['controller_reward_weights']['root_pitch']) + 
                  # + ((root_height_reward) * self._settings['controller_reward_weights']['root_height']) 
                  # (right_hand_pos_x_reward * self._settings['controller_reward_weights']['right_hand_x_pos'])
                  )# optimal is 0
        
        self._reward_sum = self._reward_sum + reward
        if ( self._settings["use_parameterized_control"] ):
            self.changeParameters()
        return reward
    
    def changeParameters(self):
        """
            Slowly modifies the parameters during training
        """
        move_scale = self._settings['average_parameter_change']
        ## Can change at most by +-move_scale between each action This does not seem to work as well = 0.1
        # r = ((r - 0.5) * 2.0) * move_scale
        vel_bounds = self._settings['controller_parameter_settings']['velocity_bounds']
        self._target_vel = randomUniformExporation(move_scale, [self._target_vel], vel_bounds)[0]
        self._target_vel = clampAction([self._target_vel], vel_bounds)[0]
        # print("New target velocity after action: ", self._target_vel)
        
        root_height_bounds = self._settings['controller_parameter_settings']['root_height_bounds']
        self._target_root_height = randomUniformExporation(move_scale, [self._target_root_height], root_height_bounds)[0]
        self._target_root_height = clampAction([self._target_root_height], root_height_bounds)[0]
        
        # print("New target Velocity: ", self._target_vel)
        # if ( self._settings["use_parameterized_control"] )
            
    def getControlParameters(self):
        # return [self._target_vel, self._target_root_height, self._target_lean, self._target_hand_pos]
        return []
        
    def initEpoch(self):
        super(Hopper2DActor,self).initEpoch()
        if ( self._settings["use_parameterized_control"] ):
            # print (os.getpid(), ", Old target velocity: ", self._target_vel)
            # _bounds = self._settings['controller_parameter_settings']['velocity_bounds']
            # self._target_vel = np.random.uniform(_bounds[0][0], _bounds[1][0])
            self._target_vel = self._settings["target_velocity"]
            # print (os.getpid(), ", New target velocity: ", self._target_vel)
            """
            _bounds = self._settings['controller_parameter_settings']['root_height_bounds']
            self._target_root_height = np.random.uniform(_bounds[0][0], _bounds[1][0])
            
            _bounds = self._settings['controller_parameter_settings']['root_pitch_bounds']
            self._target_lean = np.random.uniform(_bounds[0][0], _bounds[1][0])
            
            _bounds = self._settings['controller_parameter_settings']['right_hand_x_pos_bounds']
            self._target_hand_pos = np.random.uniform(_bounds[0][0], _bounds[1][0])
            """
    def getEvaluationData(self):
        return self._reward_sum
    
    def hasNotFallen(self, exp):
        position_root = exp.getEnvironment().getActor().getStateEuler()[0:][:3]
        if ( exp.getEnvironment().agentHasFallen() or (position_root[1] < 0.0) ) :
            return 0
        else:
            return 1
        # return not exp.getEnvironment().agentHasFallen()
        
