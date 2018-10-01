import sys
import math
from actor.ActorInterface import ActorInterface
from model.ModelUtil import randomExporation, randomUniformExporation, reward_smoother, clampAction, clampActionWarn
import numpy as np
import copy

class GapGame2DActor(ActorInterface):
    
    def __init__(self, discrete_actions, experience):
        super(GapGame2DActor,self).__init__(discrete_actions, experience)
        self._target_vel_weight=self._settings["target_velocity_decay"]
        self._target_vel = self._settings["target_velocity"]
        
    
    def updateAction(self, sim, action_):
        action_ = np.array(action_, dtype='float64')
        sim.getEnvironment().updateAction(action_)
    
    # @profile(precision=5)
    def act(self, exp, action_, bootstrapping=False):
        samp = self.getActionParams(action_)
        
        reward = self.actContinuous(exp, samp, bootstrapping=bootstrapping)
        
        return reward
    
    # @profile(precision=5)
    def actContinuous(self, exp, action_, bootstrapping=False):
        # Actor should be FIRST here
        # print "Action: " + str(action_)
        # action_ = copy.deepcopy(action_)
        action_ = np.array(action_, dtype='float64')
        (action_, outside_bounds) = clampActionWarn(action_, self._action_bounds)
        
        averageSpeed = exp.getEnvironment().actContinuous(action_, bootstrapping=bootstrapping)
        if (self.hasNotFallen(exp)):
            vel_dif = np.abs(self._target_vel - averageSpeed)
            # reward = math.exp((vel_dif*vel_dif)*self._target_vel_weight) # optimal is 0
            if ( self._settings["use_parameterized_control"] ):
                self.changeParameters()
            reward = reward_smoother(vel_dif, self._settings, self._target_vel_weight)
            if ( self._settings['print_level'] == 'debug'):
                print("target velocity: ", self._target_vel)
                print("velocity diff: ", vel_dif)
                print("reward: ", reward)
        else:
            return 0.0
        reward = reward
        self._reward_sum = self._reward_sum + reward
        # print("Reward Sum: ", self._reward_sum)
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
        
        """
        root_height_bounds = self._settings['controller_parameter_settings']['root_height_bounds']
        self._target_root_height = randomUniformExporation(move_scale, [self._target_root_height], root_height_bounds)[0]
        self._target_root_height = clampAction([self._target_root_height], root_height_bounds)[0]
        
        root_pitch_bounds = self._settings['controller_parameter_settings']['root_pitch_bounds']
        self._target_lean = randomUniformExporation(move_scale, [self._target_lean], root_pitch_bounds)[0]
        self._target_lean = clampAction([self._target_lean], root_pitch_bounds)[0]
        
        _bounds = self._settings['controller_parameter_settings']['right_hand_x_pos_bounds']
        self._target_hand_pos = randomUniformExporation(move_scale, [self._target_hand_pos], _bounds)[0]
        self._target_hand_pos = clampAction([self._target_hand_pos], _bounds)[0]
        """
        # print("New target Velocity: ", self._target_vel)
        # if ( self._settings["use_parameterized_control"] )
            
    def getControlParameters(self):
        # return [self._target_vel, self._target_root_height, self._target_lean, self._target_hand_pos]
        return [self._target_vel]
        
    def initEpoch(self):
        super(GapGame2DActor,self).initEpoch()
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
    
    def setTargetVelocity(self, exp, target_vel):
        self._target_vel = target_vel
        exp.getEnvironment().setTargetVelocity(self._target_vel)
    
    def hasNotFallen(self, exp):
        # if ( (not ( exp.getEnvironment().agentHasFallen() or exp.getEnvironment().hitWall())) and () ):
        # if ( exp.getEnvironment().agentHasFallen() or exp.getEnvironment().hitWall()) :
        if ( exp.getEnvironment().agentHasFallen() ):
            return 0
        else:
            return 1