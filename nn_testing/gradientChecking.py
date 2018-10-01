import theano
from theano import tensor as T
import numpy as np
import lasagne
import sys
sys.path.append('../')
from model.ModelUtil import *
from model.NeuralNetwork import NeuralNetwork
from util.ExperienceMemory import ExperienceMemory
import matplotlib.pyplot as plt
import math


def f(x):
    return (math.cos(x)-0.75)*(math.sin(x)+0.75)

def fcos(x):
    return (math.cos(x))

def fNoise(x):
    out = f(x)
    if (x > 1.0) and (x < 2.0):
        # print "Adding noise"
        r = random.choice([0,1])
        if r == 1:
            out = x
        else:
            out = out
    return out

if __name__ == '__main__':
    """
    State is the input state and Action is the desired output (y).
    """
    
    file = open(sys.argv[1])
    settings = json.load(file)
    print ("Settings: " + str(json.dumps(settings)))
    file.close()
    import os    
    os.environ['THEANO_FLAGS'] = "mode=FAST_RUN,device="+settings['training_processor_type']+",floatX="+settings['float_type']
    
    # state_bounds = np.array([[1.8],[3.0]])
    state_bounds = np.array([[-5.0],[8.0]])
    action_bounds = np.array([[-4.0],[2.0]])
    reward_bounds = np.array([[-3.0],[1.0]])
    experience_length = 200
    batch_size=32
    # states = np.repeat(np.linspace(0.0, 5.0, experience_length),2, axis=0)
    states = np.linspace(state_bounds[0], state_bounds[1], experience_length)
    # states = np.linspace(-5.0,-2.0, experience_length/2)
    # states = np.append(states, np.linspace(-1.0, 5.0, experience_length/2))
    # print states
    # actions = np.array(map(fNoise, states))
    actions = np.array(list(map(f, states)))
    actionsNoNoise = np.array(list(map(f, states)))
    
    # states2 = np.transpose(np.repeat([states], 2, axis=0))
    # print states2
    model = NeuralNetwork(len(state_bounds[0]), len(action_bounds[0]), state_bounds, action_bounds, settings)
    
    experience = ExperienceMemory(len(state_bounds[0]), len(action_bounds[0]), experience_length, continuous_actions=True, settings=settings)
    
    experience.setStateBounds(state_bounds)
    experience.setRewardBounds(reward_bounds)
    experience.setActionBounds(action_bounds)
    for i in range(experience_length):
        action_ = np.array([actions[i]])
        state_ = np.array([states[i]])
        # print "Action: " + str([actions[i]])
        experience.insert(state_, action_, state_, np.array([0]))
    
    errors=[]
    for i in range(10000):
        _states, _actions, _result_states, _rewards, fals_, _G_ts = experience.get_batch(batch_size)
        # print _actions 
        error = model.train(_states, _actions)
        errors.append(error)
        # print "Error: " + str(error)
    
    # model.predict does the normalizing and scaling internally
    predicted_actions = np.array(list(map(model.predict, states)))
    
    
    # print "var : " + str(predicted_actions_var)
    # print "act : " + str(predicted_actions)
    
    
    std = 1.0
    # _fig, (_bellman_error_ax, _reward_ax, _discount_error_ax) = plt.subplots(1, 1, sharey=False, sharex=True)
    _fig, (_bellman_error_ax, _training_error_ax) = plt.subplots(1, 2, sharey=False, sharex=False)
    _bellman_error, = _bellman_error_ax.plot(states, actions, linewidth=2.0, color='y', label="True function")
    # _bellman_error, = _bellman_error_ax.plot(states, predicted_actions_dropout, linewidth=2.0, color='r', label="Estimated function with dropout")
    _bellman_error, = _bellman_error_ax.plot(states, predicted_actions, linewidth=2.0, color='g', label="Estimated function")
    # _bellman_error, = _bellman_error_ax.plot(states, actionsNoNoise, linewidth=2.0, label="True function")
    
    
    # _bellman_error_std = _bellman_error_ax.fill_between(states, predicted_actions - predicted_actions_var,
    #                                                     predicted_actions + predicted_actions_var, facecolor='green', alpha=0.5)
    # _bellman_error_std = _bellman_error_ax.fill_between(states, lower_var, upper_var, facecolor='green', alpha=0.5)
    # _bellman_error_ax.set_title("True function")
    _bellman_error_ax.set_ylabel("function value: f(x)")
    _bellman_error_ax.set_xlabel("x")
    # Now add the legend with some customizations.
    legend = _bellman_error_ax.legend(loc='upper left', shadow=True)
    plt.setp(legend.get_texts(), fontsize='small')
    _bellman_error_ax.set_title("Predicted curves")
    _bellman_error_ax.grid(b=True, which='major', color='black', linestyle='-')
    _bellman_error_ax.grid(b=True, which='minor', color='g', linestyle='--')


    """
    _reward, = _reward_ax.plot([], [], linewidth=2.0)
    _reward_std = _reward_ax.fill_between([0], [0], [1], facecolor='blue', alpha=0.5)
    _reward_ax.set_title('Mean Reward')
    _reward_ax.set_ylabel("Reward")
    _discount_error, = _discount_error_ax.plot([], [], linewidth=2.0)
    _discount_error_std = _discount_error_ax.fill_between([0], [0], [1], facecolor='blue', alpha=0.5)
    _discount_error_ax.set_title('Discount Error')
    _discount_error_ax.set_ylabel("Absolute Error")
    plt.xlabel("Iteration")
    """
    _title = "Training function"
    _fig.suptitle(_title, fontsize=18)
    
    _fig.set_size_inches(8.0, 4.5, forward=True)
    # er = plt.figure(2)
    _training_error_ax.plot(range(len(errors)), errors)
    _training_error_ax.set_ylabel("Error")
    _training_error_ax.set_xlabel("Iteration")
    _training_error_ax.set_title("Training Error")
    _training_error_ax.grid(b=True, which='major', color='black', linestyle='-')
    _training_error_ax.grid(b=True, which='minor', color='g', linestyle='--')
    
    grads_ = model.getGrads([[states[0]]], [[actions[0]]])
    print ("Grads : ", len(grads_))
    print ("Grad: ", grads_)
    print ("Grad sum: ", np.sum(grads_[0], axis=1))
    grad_dirs=[]
    old_states_=[]
    predicted_actions_=[]
    space=1
    spaces_=0
    for s in range(0, len(states), 2):
        if (s % space) == 0:
            action_ = np.reshape(np.array([predicted_actions[s]-0.01]), (1,1))
            state_ = np.reshape(np.array([states[s]]), (1,1))
            grads_ = model.getGrads(state_, action_)
            print ("Grad: ", grads_[0])
            diff = model.bellman_error(state_, action_)
            print ("Diff, ", diff)
            grad_dir = np.sum(grads_[0], axis=1)
            """
            if (grad_dir > 0.0):
                grad_dir = 1.0
            else:
                grad_dir = -1.0
                """
            grad_dirs.append(grad_dir * 100.0)
            old_states_.append(states[s])
            predicted_actions_.append(predicted_actions[s])
    
    _bellman_error_ax.quiver(old_states_, predicted_actions_, grad_dirs, np.zeros((len(grad_dirs))), linewidth=0.5, pivot='tail', edgecolor='k', headaxislength=4, alpha=.5, angles='xy', linestyles='-', scale=10.0, label="gradient direction")
    
    # _fig.show()
    # er.show()
    plt.show()
    
    _fig.savefig("gradientChecking"+".svg")
    _fig.savefig("gradientChecking"+".png")