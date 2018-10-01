import numpy as np
# import lasagne
import sys
sys.path.append('../')
import matplotlib.pyplot as plt
import math
import random
import json



def f(x):
    return ((math.cos(x)-0.75)*(math.sin(x)+0.75))

def fNoise(x):
    out = f(x)
    if (x > -1.0) and (x < 0.0):
        # print "Adding noise"
        r = random.choice([0,1])
        n = np.random.normal(0, 0.2 * (np.abs(x)+1), 1)[0]
        out = out + n
    return out

if __name__ == '__main__':
    
    file = open(sys.argv[1])
    settings = json.load(file)
    print ("Settings: " + str(json.dumps(settings)))
    file.close()
    import os    
    os.environ['THEANO_FLAGS'] = "mode=FAST_RUN,device="+settings['training_processor_type']+",floatX="+settings['float_type']
        
    # import theano
    # from theano import tensor as T
    from model.ModelUtil import *
    from model.DropoutNetwork import DropoutNetwork
    from util.ExperienceMemory import ExperienceMemory
    
    state_bounds = np.array([[-5.0],[5.0]])
    action_bounds = np.array([[-4.0],[2.0]])
    reward_bounds = np.array([[-3.0],[1.0]])
    experience_length = 300
    batch_size=16
    # states = np.repeat([np.linspace(-5.0, 5.0, experience_length)],2, axis=0)
    states = np.linspace(-5.0,-1.0, experience_length/2)
    states = np.append(states, np.linspace(-1.0, 5.0, experience_length/2))
    old_states = states
    # print states
    actions = np.array(map(fNoise, states))
    actionsNoNoise = np.array(map(f, states))
    
    # states2 = np.transpose(np.repeat([states], 2, axis=0))
    # print states2
    model = DropoutNetwork(len(state_bounds[0]), len(action_bounds[0]), state_bounds, action_bounds, settings)
    
    experience = ExperienceMemory(len(state_bounds[0]), len(action_bounds[0]), experience_length, continuous_actions=True, settings=settings)
    experience.setStateBounds(state_bounds)
    experience.setRewardBounds(reward_bounds)
    experience.setActionBounds(action_bounds)
    experience.setSettings(settings)
    arr = range(experience_length)
    random.shuffle(arr)
    num_samples_to_keep=300
    given_actions=[]
    given_states=[]
    for i in range(num_samples_to_keep):
        action_ = np.array([actions[arr[i]]])
        given_actions.append(action_)
        state_ = np.array([states[arr[i]]])
        given_states.append(state_)
        # print "Action: " + str([actions[i]])
        experience.insert(state_, action_, state_, np.array([0]))
    
    errors=[]
    for i in range(10000):
        _states, _actions, _result_states, _rewards, fals_, _G_ts = experience.get_batch(batch_size)
        # print _actions 
        error = model.train(_states, _actions)
        errors.append(error)
        # print "Error: " + str(error)
    
    
    states = np.linspace(-5.0, 5.0, experience_length)
    actionsNoNoise = np.array(map(f, states))
    
    predicted_actions = np.array(map(model.predict, states))
    predicted_actions_dropout = np.array(map(model.predictWithDropout, states))
    predicted_actions_var = []
    
    # print ("modelPrecsionInv: ", modelPrecsionInv)
    predictions = []
    for i in range(len(states)):
        
        var_ = getModelPredictionUncertanty(model, state=states[i], length=0.5, num_samples=128)
        # print var_
        predicted_actions_var.append(var_[0])
        predictions=[]
    # predictions = model.predictWithDropout(samp_)
    predicted_actions_var = np.array(predicted_actions_var)
    # states=np.reshape(states, (experience_length,1))
    predicted_actions = np.reshape(predicted_actions, (experience_length,))
    print "states shape: " + str(states.shape)
    print "var shape: " + str(predicted_actions_var.shape)
    print "act shape: " + str(predicted_actions.shape)
    
    # print "var : " + str(predicted_actions_var)
    # print "act : " + str(predicted_actions)
    
    lower_var=[]
    upper_var=[]
    
    for i in range(len(states)):
        lower_var.append(predicted_actions[i]-predicted_actions_var[i])
        upper_var.append(predicted_actions[i]+predicted_actions_var[i])
     
    lower_var = np.array(lower_var)
    upper_var = np.array(upper_var)
    
    
    std = 1.0
    # _fig, (_bellman_error_ax, _reward_ax, _discount_error_ax) = plt.subplots(1, 1, sharey=False, sharex=True)
    _fig, (_bellman_error_ax) = plt.subplots(1, 1, sharey=False, sharex=True)
    _bellman_error, = _bellman_error_ax.plot(old_states, actions, linewidth=2.0, color='y', label="True function with noise")
    _bellman_error, = _bellman_error_ax.plot(states, predicted_actions_dropout, linewidth=2.0, color='r', label="Estimated function with dropout")
    _bellman_error, = _bellman_error_ax.plot(states, predicted_actions, linewidth=2.0, color='g', label="Estimated function")
    _bellman_error, = _bellman_error_ax.plot(states, actionsNoNoise, linewidth=2.0, label="True function")
    _bellman_error = _bellman_error_ax.scatter(given_states, given_actions, label="Data trained on")
    _bellman_error, = _bellman_error_ax.plot(states, predicted_actions_var, linewidth=2.0, label="Variance")
    
    # _bellman_error_std = _bellman_error_ax.fill_between(states, predicted_actions - predicted_actions_var,
    #                                                     predicted_actions + predicted_actions_var, facecolor='green', alpha=0.5)
    _bellman_error_std = _bellman_error_ax.fill_between(states, lower_var, upper_var, facecolor='green', alpha=0.5)
    # _bellman_error_ax.set_title("True function")
    _bellman_error_ax.set_ylabel("Absolute Error")
    # Now add the legend with some customizations.
    legend = _bellman_error_ax.legend(loc='lower right', shadow=True)
    plt.grid(b=True, which='major', color='black', linestyle='-')
    plt.grid(b=True, which='minor', color='g', linestyle='--')


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
    er = plt.figure(2)
    plt.plot(range(len(errors)), errors)
    
    print "Max var: " + str(np.max(predicted_actions_var))
    print "Min var: " + str(np.min(predicted_actions_var))
    
    # _fig.show()
    # er.show()
    plt.show()