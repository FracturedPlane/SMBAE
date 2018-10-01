import theano
from theano import tensor as T
import numpy as np
import lasagne
import sys
sys.path.append('../learn/')
from ExperienceMemory import ExperienceMemory
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

def f2(x, y):
    return (math.cos(x)-0.75)*(math.sin(x)+0.75)+(math.sin(y))

def fNoise2(x, y):
    out = f2(x, y)
    if (y > -1.0) and (y < 1.0):
        # print "Adding noise"
        out = out + random.normalvariate(out,0.35)
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
    
    state_bounds = np.array([[-5.0, -5.0],[5.0, 5.0]])
    action_bounds = np.array([[-4.0],[2.0]])
    reward_bounds = np.array([[-3.0],[1.0]])
    experience_length = 50
    batch_size=32
    # states = np.repeat([np.linspace(-5.0, 5.0, experience_length)],2, axis=0)
    states = np.linspace(-5.0, 5.0, experience_length)
    X, Y = np.meshgrid(states, states)
    
    states_ = zip(X,Y)
    states_X = np.reshape(X, (-1,))
    states_Y = np.reshape(Y, (-1,))
    
    states_ = np.array(zip(states_X, states_Y))
    
    
    
    
    # print states_
    actions = np.array(map(fNoise2, states_X, states_Y))
    actionsNoNoise = np.array(map(f2, states_Y, states_Y))
    
    print "length: " + str(len(states_))
    print  states_[-10:]
    
    actions_ = np.reshape(actions, (experience_length,-1))
    
    model = DropoutNetwork(len(state_bounds[0]), len(action_bounds[0]), state_bounds, action_bounds)
    
    experience = ExperienceMemory(len(state_bounds[0]), len(action_bounds[0]), experience_length, continuous_actions=True)
    for i in range(len(states_)):
        action_ = np.array([actions[i]])
        state_ = np.array([states_[i]])
        # print "Action: " + str([actions[i]])
        experience.insert(norm_state(state_, state_bounds), norm_action(action_, action_bounds),
                           norm_state(state_, state_bounds), norm_reward(np.array([0]), reward_bounds))
    
    
    print "Experience samples: " + str(experience.samples())
    errors=[]
    for i in range(250000):
        _states, _actions, _result_states, _rewards = experience.get_batch(batch_size)
        # print _actions 
        error = model.train(_states, _actions)
        errors.append(error)
        # print "Error: " + str(error)
    
    
    predicted_actions = np.array(map(model.predict, states_))
    predicted_actions_dropout = np.array(map(model.predictWithDropout, states))
    predicted_actions_var = []
    
    l=0.1
    num_samples = 32
    modelPrecsion = ((l*l)*0.9)/ (2*num_samples*1e-4 )
    predictions = []
    for i in range(len(states_)):
        
        samp_ = np.repeat([states_[i]],num_samples, axis=0)
        # print "Sample: " + str(samp_)
        for sam in samp_:
            predictions.append(model.predictWithDropout(sam))
        # print "Predictions: " + str(predictions)
        var_ = (1.0/modelPrecsion)+np.mean(predictions) - predicted_actions[i]
        # print var_
        predicted_actions_var.append(var_[0])
        predictions=[]
    # predictions = model.predictWithDropout(samp_)
    predicted_actions_var = np.reshape(np.array(predicted_actions_var), (len(predicted_actions_var),-1))
    # states=np.reshape(states, (experience_length,1))
    print "states shape: " + str(states.shape)
    print "var shape: " + str(predicted_actions_var.shape)
    print "act shape: " + str(predicted_actions.shape)
    
    upper_var = predicted_actions + np.array(predicted_actions_var)
    lower_var = predicted_actions - np.array(predicted_actions_var)
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    predicted_actions_ = np.reshape(predicted_actions, (experience_length,-1))
    
    upper_var = np.reshape(upper_var, (experience_length,-1))
    lower_var = np.reshape(lower_var, (experience_length,-1))
    
    ax.plot_surface(X, Y, actions_, rstride=4, cstride=4, alpha=0.15)
    ax.plot_surface(X, Y, predicted_actions_, rstride=4, cstride=4, alpha=0.4, color='green')
    ax.plot_surface(X, Y, upper_var, rstride=4, cstride=4, alpha=0.3, color='yellow')
    ax.plot_surface(X, Y, lower_var, rstride=4, cstride=4, alpha=0.3, color='yellow')
    
    er = plt.figure(2)
    plt.plot(range(len(errors)), errors)
    
    print "Max var: " + str(np.max(predicted_actions_var))
    print "Min var: " + str(np.min(predicted_actions_var))
    
    plt.show()