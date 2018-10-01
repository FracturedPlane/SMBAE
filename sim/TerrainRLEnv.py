"""
"""
import numpy as np
import math
from sim.SimInterface import SimInterface
import sys
from actor.DoNothingActor import DoNothingActor
# sys.path.append("../simbiconAdapter/")

# import scipy.integrate as integrate
# import matplotlib.animation as animation


class TerrainRLEnv(SimInterface):

    def __init__(self, exp, settings):
        #------------------------------------------------------------
        # set up initial state
        super(TerrainRLEnv,self).__init__(exp, settings)
        self._action_dimension=3
        self._range = 5.0
        self._actor = DoNothingActor()

    def initEpoch(self):
        self.getEnvironment().initEpoch()
        # self.getAgent().initEpoch()
        
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

    def generateEnvironmentSample(self):
        self._exp.getEnvironment().generateEnvironmentSample()
        
    def generateValidationEnvironmentSample(self, epoch):
        pass
            
    def getActor(self):
        return self._actor
    
    def updateAction(self, action_):
        self.getActor().updateAction(self, action_)
        
    def update(self):
        for i in range(1):
            self.getEnvironment().update()
            
    def needUpdatedAction(self):
        return self.getEnvironment().needUpdatedAction()
            
    def display(self):
        self.getEnvironment().display()
    
    def finish(self):
        self._exp.finish()
    
    def getState(self):
        state_ = self.getEnvironment().getState()
        state = np.array(state_)
        state = np.reshape(state, (-1, len(state_)))
        return state
    
    def setState(self, st):
        self._agent = st
        self._box.state[0,0] = st[0]
        self._box.state[0,1] = st[1]
        
    def getControllerBackOnTrack(self):
        import characterSim
        """
            Push controller back into a good state space
        """
        state_ = self.getEnvironment().getState()
        state_params = list(state_.getParams())
        # move the body such that the current grab position will overlap the anchor
        # print (("Sim State: ", state_params))
        state_params[5] = 1.0* state_params[3]
        state_params[6] = 1.0* state_params[4]
        # print (("New Sim State: ", state_params)  )
        state__c = characterSim.State(state_.getID(), state_params)
        self._exp.getEnvironment().setState(state__c)
        
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
        
    def setRandomSeed(self, seed):
        """
            Set the random seed for the simulator
            This is helpful if you are running many simulations in parallel you don't
            want them to be producing the same results if they all init their random number 
            generator the same.
        """
        print ( "Setting random seed: ", seed )
        self.getEnvironment().setRandomSeed(seed)
        
        
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
