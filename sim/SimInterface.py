"""
"""
import numpy as np
import math

from actor.DoNothingActor import DoNothingActor
# import scipy.integrate as integrate
# import matplotlib.animation as animation


class SimInterface(object):

    def __init__(self, exp, settings_):
        #------------------------------------------------------------
        # set up initial state
        # super(BallGame1DChoiceState,self).__init__()
        self._exp = exp
        self._settings = settings_
        self._actor = DoNothingActor()
        
    def getSettings(self):
        return self._settings
        
    def getEnvironment(self):
        return self._exp
    
    def endOfEpoch(self):
        return self.getEnvironment().endOfEpoch()

    def init(self):
        self.getEnvironment().init()
            
    def initEpoch(self):
        self.getEnvironment().initEpoch()
    
    def generateValidation(self, data, epoch):
        pass
    
    def generateEnvironmentSample(self):
        pass
    
    def getEvaluationData(self):
        pass
    
    def getActor(self):
        return self._actor
    
    def setActor(self, actor):
        self._actor = actor
    
    def finish(self):
        self._exp.finish()
    
    def getState(self):
        """
            I like the state in this shape (1, state_length)
        """
        state_ = self._exp.getState()
        state = np.array(state_)
        state = np.reshape(state, (1, len(state_)))
        return state
    
    def setState(self, st):
        pass
        
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
        
    def getStateFromSimState(self, simState):
        """
            Converts a detailed simulation state to a state better suited for learning
        """
        pass
    
    def getSimState(self):
        """
            Gets a more detailed state that can be used to re-initilize the state of the character back to this state later.
        """
        pass
    
    def setSimState(self, state_):
        """
            Sets the state of the simulation to the given state
        """
        pass
    
    def updateViz(self, actor, agent, directory, p=1.0):
        """
            Maybe the sim has some cool visualization of the policy or something.
            This will update that visualization
        """
        pass
    
    def setRandomSeed(self, seed):
        """
            Set the random seed for the simulator
            This is helpful if you are running many simulations in parallel you don't
            want them to be producing the same results if they all init their random number 
            generator the same.
        """
        pass
        
        
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
