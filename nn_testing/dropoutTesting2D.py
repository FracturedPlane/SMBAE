import numpy as np
# import lasagne
import sys
sys.path.append('../')
import matplotlib.pyplot as plt
import math
import random
import json

f2_scale = 5.0


def f(x):
    return ((math.cos(x)-0.75)*(math.sin(x)+0.75))

def f2(x):
    return ((math.cos(x*2)-0.75)*(math.sin(x/2)+0.75))*f2_scale

def fNoise(x):
    out = f(x)
    if (x > -1.0) and (x < 0.0):
        # print "Adding noise"
        r = random.choice([0,1])
        n = np.random.normal(0, 0.2 * (np.abs(x)+1), 1)[0]
        out = out + n
    return out

def f2Noise(x):
    out = f2(x)
    if (x > -2.0) and (x < -1.0):
        # print "Adding noise"
        r = random.choice([0,1])
        n = np.random.normal(0, 1.2 * (np.abs(x)+1), 1)[0]
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
    action_bounds = np.array([[-3.0,-6.0*f2_scale],[2.0, 1.0*f2_scale]])
    reward_bounds = np.array([[-3.0],[1.0]])
    experience_length = 300
    batch_size=32
    # states = np.repeat([np.linspace(-5.0, 5.0, experience_length)],2, axis=0)
    states = np.linspace(-5.0,-1.0, experience_length/2)
    states = np.append(states, np.linspace(-1.0, 5.0, experience_length/2))
    old_states = states
    # print states
    actions = np.array(map(fNoise, states))
    actions2 = np.array(map(f2Noise, states))
    allAction = np.transpose(np.concatenate(([actions], [actions2]), axis=0))
    actionsNoNoise = np.array(map(f, states))
    actionsNoNoise2 = np.array(map(f2, states))
    
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
        action_ = np.array(allAction[arr[i]])
        given_actions.append(action_)
        state_ = np.array([states[arr[i]]])
        given_states.append(state_)
        # print "Action: " + str([actions[i]])
        experience.insert(state_, action_, state_, np.array([0]))
    
    errors=[]
    for i in range(50000):
        _states, _actions, _result_states, _rewards, fals_, _G_ts = experience.get_batch(batch_size)
        # print _actions 
        error = model.train(_states, _actions)
        errors.append(error)
        # print "Error: " + str(error)
    
    
    states = np.linspace(-10.0, 10.0, experience_length)
    actionsNoNoise = np.array(map(f, states))
    
    predicted_actions_ = np.array(map(model.predict, states))
    # print ("Predicted actions: ", predicted_actions_)
    predicted_actions = predicted_actions_[:,0]
    predicted_actions2 = predicted_actions_[:,1]
    predicted_actions_dropout_ = np.array(map(model.predictWithDropout, states))
    predicted_actions_dropout = predicted_actions_dropout_[:,0]
    predicted_actions_dropout2 = predicted_actions_dropout_[:,1]
    predicted_actions_var_ = []
    
    # print ("modelPrecsionInv: ", modelPrecsionInv)
    predictions = []
    for i in range(len(states)):
        
        var_ = getModelPredictionUncertanty(model, state=states[i], length=0.5, num_samples=128)
        # print var_
        predicted_actions_var_.append(var_)
        predictions=[]
    # predictions = model.predictWithDropout(samp_)
    predicted_actions_var = np.array(predicted_actions_var_)[:,0]
    predicted_actions_var2 = np.array(predicted_actions_var_)[:,1]
    # states=np.reshape(states, (experience_length,1))
    print "states shape: " + str(states.shape)
    print "var shape: " + str(predicted_actions_var.shape)
    print "act shape: " + str(predicted_actions.shape)
    
    # print "var : " + str(predicted_actions_var)
    # print "act : " + str(predicted_actions)
    
    lower_var=[]
    upper_var=[]
    lower_var2=[]
    upper_var2=[]
    for i in range(len(states)):
        lower_var.append(predicted_actions[i]-predicted_actions_var[i])
        upper_var.append(predicted_actions[i]+predicted_actions_var[i])
        lower_var2.append(predicted_actions2[i]-predicted_actions_var2[i])
        upper_var2.append(predicted_actions2[i]+predicted_actions_var2[i])
     
    lower_var = np.array(lower_var)
    upper_var = np.array(upper_var)
    lower_var2 = np.array(lower_var2)
    upper_var2 = np.array(upper_var2)
    
    # print("given_actions: ", given_actions)
    std = 1.0
    # _fig, (_bellman_error_ax, _reward_ax, _discount_error_ax) = plt.subplots(1, 1, sharey=False, sharex=True)
    _fig, (_bellman_error_ax,_bellman_error_ax2) = plt.subplots(2, 1, sharey=False, sharex=True)
    _bellman_error, = _bellman_error_ax.plot(old_states, actions, linewidth=2.0, color='y', label="True function with noise")
    _bellman_error, = _bellman_error_ax.plot(states, predicted_actions_dropout, linewidth=2.0, color='r', label="Estimated function with dropout")
    _bellman_error, = _bellman_error_ax.plot(states, predicted_actions, linewidth=2.0, color='g', label="Estimated function")
    _bellman_error, = _bellman_error_ax.plot(states, actionsNoNoise, linewidth=2.0, label="True function")
    _bellman_error = _bellman_error_ax.scatter(given_states, np.array(given_actions)[:,0], label="Data trained on")
    _bellman_error, = _bellman_error_ax.plot(states, predicted_actions_var, linewidth=2.0, label="Variance")
    _bellman_error_std = _bellman_error_ax.fill_between(states, lower_var, upper_var, facecolor='green', alpha=0.25)
    plt.grid(b=True, which='major', color='black', linestyle='-')
    plt.grid(b=True, which='minor', color='g', linestyle='--')
    legend = _bellman_error_ax.legend(loc='lower right', shadow=True)
    _bellman_error_ax.set_ylabel("Absolute Error")
    # _bellman_error_std = _bellman_error_ax.fill_between(states, predicted_actions - predicted_actions_var,
    #                                                     predicted_actions + predicted_actions_var, facecolor='green', alpha=0.5)
    actionsNoNoise2 = np.array(map(f2, states)) 
    _bellman_error, = _bellman_error_ax2.plot(old_states, actions2, linewidth=2.0, color='y', label="True function with noise")
    _bellman_error, = _bellman_error_ax2.plot(states, predicted_actions_dropout2, linewidth=2.0, color='r', label="Estimated function with dropout")
    _bellman_error, = _bellman_error_ax2.plot(states, predicted_actions2, linewidth=2.0, color='g', label="Estimated function")
    _bellman_error, = _bellman_error_ax2.plot(states, actionsNoNoise2, linewidth=2.0, label="Action2")
    _bellman_error = _bellman_error_ax2.scatter(given_states, np.array(given_actions)[:,1], label="Data trained on")
    _bellman_error_std = _bellman_error_ax2.fill_between(states, lower_var2, upper_var2, facecolor='green', alpha=0.25)
    _bellman_error, = _bellman_error_ax2.plot(states, predicted_actions_var2, linewidth=2.0, label="Variance")
    # _bellman_error_ax.set_title("True function")
    # Now add the legend with some customizations.
    legend = _bellman_error_ax2.legend(loc='lower right', shadow=True)
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
    
    print ("Max var: " + str(np.max(predicted_actions_var_, axis=0)))
    print ("Min var: " + str(np.min(predicted_actions_var_, axis=0)))
    
    # _fig.show()
    # er.show()
    plt.show()