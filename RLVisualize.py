import matplotlib.pyplot as plt
# from matplotlib import mpl
import numpy as np
# import matplotlib.animation as animation
import random
import sys
import json

class RLVisualize(object):
    
    def __init__(self, title, settings=None):
        """
            Three plots
            bellman error
            average reward
            discounted reward error
        """
        if (settings != None):
            self._sim_iteration_scale = (settings['plotting_update_freq_num_rounds']*settings['max_epoch_length']*settings['epochs'])
            self._iteration_scale = ((self._sim_iteration_scale * settings['training_updates_per_sim_action']) / 
                                     settings['sim_action_per_training_update'])
            if ('on_policy' in settings and (settings['on_policy'])):
                self._sim_iteration_scale = self._sim_iteration_scale * settings['num_on_policy_rollouts']
                self._iteration_scale = ((self._sim_iteration_scale / (settings['max_epoch_length'] )) *
                                     settings['critic_updates_per_actor_update'])
        else:
            self._iteration_scale = 1
            self._sim_iteration_scale = 1
        self._title=title
        self._fig, (self._bellman_error_ax, self._reward_ax, self._discount_error_ax) = plt.subplots(3, 1, sharey=False, sharex=True)
        self._bellman_error, = self._bellman_error_ax.plot([], [], linewidth=2.0)
        self._bellman_error_std = self._bellman_error_ax.fill_between([0], [0], [1], facecolor='blue', alpha=0.5)
        self._bellman_error_ax.set_title('Bellman Error')
        self._bellman_error_ax.set_ylabel("Absolute Error")
        self._bellman_error_ax.grid(b=True, which='major', color='black', linestyle='--')
        self._reward, = self._reward_ax.plot([], [], linewidth=2.0)
        self._reward_std = self._reward_ax.fill_between([0], [0], [1], facecolor='blue', alpha=0.5)
        self._reward_ax.set_title('Mean Reward')
        self._reward_ax.set_ylabel("Reward")
        self._reward_ax.grid(b=True, which='major', color='black', linestyle='--')
        self._discount_error, = self._discount_error_ax.plot([], [], linewidth=2.0)
        self._discount_error_std = self._discount_error_ax.fill_between([0], [0], [1], facecolor='blue', alpha=0.5)
        self._discount_error_ax.set_title('Discount Error')
        self._discount_error_ax.set_ylabel("Absolute Error")
        self._discount_error_ax.grid(b=True, which='major', color='black', linestyle='--')
        plt.xlabel("Simulated Actions x" + str(self._sim_iteration_scale) + ", Training Updates x" + str(self._iteration_scale))
        
        self._fig.set_size_inches(8.0, 12.5, forward=True)
        
    def init(self):
        """
            Three plots
            bellman error
            average reward
            discounted reward error
        """
        self._fig, (self._bellman_error_ax, self._reward_ax, self._discount_error_ax) = plt.subplots(3, 1, sharey=False, sharex=True)
        self._bellman_error, = self._bellman_error_ax.plot([], [], linewidth=2.0)
        self._bellman_error_std = self._bellman_error_ax.fill_between([0], [0], [1], facecolor='blue', alpha=0.5)
        self._bellman_error_ax.set_title('Bellman Error')
        self._bellman_error_ax.set_ylabel("Absolute Error")
        self._bellman_error_ax.grid(b=True, which='major', color='black', linestyle='--')
        self._reward, = self._reward_ax.plot([], [], linewidth=2.0)
        self._reward_std = self._reward_ax.fill_between([0], [0], [1], facecolor='blue', alpha=0.5)
        self._reward_ax.set_title('Mean Reward')
        self._reward_ax.set_ylabel("Reward")
        self._reward_ax.grid(b=True, which='major', color='black', linestyle='--')
        self._discount_error, = self._discount_error_ax.plot([], [], linewidth=2.0)
        self._discount_error_std = self._discount_error_ax.fill_between([0], [0], [1], facecolor='blue', alpha=0.5)
        self._discount_error_ax.set_title('Discount Error')
        self._discount_error_ax.set_ylabel("Absolute Error")
        self._discount_error_ax.grid(b=True, which='major', color='black', linestyle='--')
        plt.xlabel("Simulated Actions x" + str(self._sim_iteration_scale) + ", Training Updates x" + str(self._iteration_scale))
        self._fig.suptitle(self._title, fontsize=18)
        
        # plt.grid(b=True, which='major', color='black', linestyle='--')
        # plt.grid(b=True, which='minor', color='g', linestyle='--'
        
        self._fig.set_size_inches(8.0, 12.5, forward=True)
        
    def updateBellmanError(self, error, std):
        self._bellman_error.set_data(np.arange(len(error)), error)
        # self._bellman_error.set_data(error)
        self._bellman_error_ax.collections.remove(self._bellman_error_std)
        self._bellman_error_std = self._bellman_error_ax.fill_between(np.arange(len(error)), error - std, error + std, facecolor='blue', alpha=0.5)
        
        
        self._bellman_error_ax.relim()      # make sure all the data fits
        self._bellman_error_ax.autoscale()
        
    def updateReward(self, reward, std):
        self._reward.set_xdata(np.arange(len(reward)))
        self._reward.set_ydata(reward)
        self._reward_ax.collections.remove(self._reward_std)
        self._reward_std = self._reward_ax.fill_between(np.arange(len(reward)), reward - std, reward + std, facecolor='blue', alpha=0.5)
        
        self._reward_ax.relim()      # make sure all the data fits
        self._reward_ax.autoscale()  # auto-scale
        
    def updateDiscountError(self, error, std):
        self._discount_error.set_xdata(np.arange(len(error)) )
        self._discount_error.set_ydata(error)
        self._discount_error_ax.collections.remove(self._discount_error_std)
        self._discount_error_std = self._discount_error_ax.fill_between(np.arange(len(error)), error - std, error + std, facecolor='blue', alpha=0.5)
        
        
        self._discount_error_ax.relim()      # make sure all the data fits
        self._discount_error_ax.autoscale()
        
    def show(self):
        plt.show()
        
    def redraw(self):
        self._fig.canvas.draw()
        
    def setInteractive(self):
        plt.ion()
        
    def setInteractiveOff(self):
        plt.ioff()
        
    def saveVisual(self, fileName):
        self._fig.savefig(fileName+".svg")
        self._fig.savefig(fileName+".png")
        
    def finish(self):
        """
            Closes the figure window
        """
        plt.close(self._fig)
        plt.close()
        
        
if __name__ == "__main__":
    
    datafile = sys.argv[1]
    file = open(datafile)
    trainData = json.load(file)
    # print "Training data: " + str(trainingData)
    file.close()
    settings = None
    length = len(trainData["mean_bellman_error"])
    if (len(sys.argv) == 3):
        datafile = sys.argv[2]
        file = open(datafile)   
        settings = json.load(file)
        file.close()
        
    
    """
    trainData["mean_reward"]=[]
    trainData["std_reward"]=[]
    trainData["mean_bellman_error"]=[]
    trainData["std_bellman_error"]=[]
    trainData["mean_discount_error"]=[]
    trainData["std_discount_error"]=[]
    
    """
    
    rlv = RLVisualize(datafile, settings)
    rlv.updateBellmanError(np.array(trainData["mean_bellman_error"][:length]), np.array(trainData["std_bellman_error"][:length]))
    rlv.updateReward(np.array(trainData["mean_eval"][:length]), np.array(trainData["std_eval"][:length]))
    rlv.updateDiscountError(np.fabs(trainData["mean_discount_error"][:length]), np.array(trainData["std_discount_error"][:length]))
    rlv.saveVisual("pendulum_agent")
    rlv.show()