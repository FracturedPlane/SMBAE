"""
"""
import numpy as np
import math
from sim.SimInterface import SimInterface 

# import scipy.integrate as integrate
# import matplotlib.animation as animation


class GapGame2DEnv(SimInterface):

    def __init__(self, exp, settings):
        #------------------------------------------------------------
        # set up initial state
        super(GapGame2DEnv,self).__init__(exp, settings)

    def generateValidation(self, data, epoch):
        self.getEnvironment().generateValidationEnvironmentSample(epoch)
    
    def generateValidationEnvironmentSample(self, epoch):
        self.getEnvironment().generateValidationEnvironmentSample(epoch)
        
    def generateEnvironmentSample(self):
        self.getEnvironment().generateEnvironmentSample()
        
    def getEvaluationData(self):
        return self.getEnvironment().getEvaluationData()
    
    def updateAction(self, action_):
        # print("Simbicon updating action:")
        self.getActor().updateAction(self, action_)
    
    def needUpdatedAction(self):
        return self.getEnvironment().needUpdatedAction()
            
    def display(self):
        pass
        # self.getEnvironment().display(
    
    def finish(self):
        self._exp.finish()
        
    def update(self):
        self.getEnvironment().update()
    
    def getState(self):
        # state = np.array(self._exp.getState())
        state_ = self._exp.getState()
        state = np.array(state_)
        state = np.reshape(state, (-1, len(state_)))
        
        if ( self._settings["use_parameterized_control"] ):
            state = np.append(state, [self.getActor().getControlParameters()], axis=1)
        
        return state
    
    def setState(self, st):
        self._agent = st
        self._box.state[0,0] = st[0]
        self._box.state[0,1] = st[1]
        
    def setTargetChoice(self, i):
        # need to find which target corresponds to this bin.
        _loc = np.linspace(-self._range, self._range, self._granularity)[i]
        min_dist = 100000000.0
        _choice = -1
        for i in range(int(self._choices)):
            _target_loc = self._targets[0][i][1]
            _tmp_dist = math.fabs(_target_loc - _loc)
            if ( _tmp_dist < min_dist):
                _choice = i
                min_dist = _tmp_dist
        self._target_choice = i
        self._target = self._targets[0][i]
        
    def visualizeNextState(self, next_state_, action):
        _t_length = self.getEnvironment()._game_settings['num_terrain_samples']
        # print ("next_state_: ", next_state_)
        # print ("Action: ", action)
        terrain = next_state_[:_t_length]
        terrain_dx = next_state_[_t_length]
        terrain_dy = next_state_[_t_length+1]
        character_features = next_state_[_t_length+2:]
        self.getEnvironment().visualizeNextState(terrain, action, terrain_dx)
        
        
#ani = animation.FuncAnimation(fig, animate, frames=600,
#                               interval=10, blit=True, init_func=init)


# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html
#ani.save('particle_box.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

# plt.show()

if __name__ == '__main__':
    
    np.random.seed(seed=10)
    game = BallGame1DChoiceState()

    game.enableRender()
    game._simulate=True
    # game._saveVideo=True
    print ("dt: " + str(game._dt))
    print ("BOX: " + str(game._box))
    game.init(np.random.rand(16,16),np.random.rand(16,16),np.random.rand(16,16))
    
    game.reset()
    game.setTarget(np.array([2,2]))
    num_actions=10
    scaling = 2.0
    game._box.state[0][1] = 0
    game.resetTarget()
    game.resetHeight()
    
    actions = (np.random.rand(num_actions,1)-0.5) * 2.0 * scaling
    for action in actions:
        # game.resetTarget()
        state = game.getState()
        print ("State: " + str(state))
        print ("Action: " + str(action))
        reward = game.actContinuous(action)
        print ("Reward: " + str(reward))
        game.resetTarget()
        game.resetHeight()

    game.finish()
