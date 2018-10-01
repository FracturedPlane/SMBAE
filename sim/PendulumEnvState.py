"""
"""
import numpy as np
import math

from sim.PendulumEnv import PendulumEnv

# import scipy.integrate as integrate
# import matplotlib.animation as animation


class PendulumEnvState(PendulumEnv):

    def __init__(self, exp, settings):
        #------------------------------------------------------------
        # set up initial state
        super(PendulumEnvState,self).__init__(exp, settings)
        self._granularity = 50 # This should be an even number
        self._range = 5.0
    
    def getState(self):
        state_ = super(PendulumEnvState,self).getState()
        state = np.zeros(self._granularity+len(state_))-1.0
        state[0:len(state_)] = state_
        # print ("sSelf: " + str(self._choices))
            # state = np.zeros(grandularity)
        delta = state[5]
        # print ("First delta: " + str(delta))
        delta = (delta)/(self._range/(self._granularity/2.0))
        # print ("Second delta: " + str(delta))
        index = int(delta+(self._granularity/2))
        # print ("Index is: " + str(index))
        if (index < 1):
            index = 0
        if (index >= self._granularity):
            index = self._granularity-1
        state[index+len(state_)] = 1.0
        
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
