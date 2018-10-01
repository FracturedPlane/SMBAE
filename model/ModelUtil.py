
import json
import random
import sys
# sys.path.append("../characterSimAdapter/")
# import characterSim
import numpy as np
import math
import copy 
import scipy

anchors__name="anchors"
# replace string print ([a-z|A-Z|0-9|\"| |:|(|)|\+|_|,|\.|-|\[|\]|\/]*)
def getAnchors(anchorFile):

    s = anchorFile.read()
    data = json.loads(s)
    return data[anchors__name]

def saveAnchors(anchors, fileName):
    data={}
    data[anchors__name] = anchors
    data_ = json.dumps(data)
    f = open(fileName, "w")
    f.write(data_)
    f.close()
    
def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i+n]
        
        
def anneal_value(d, settings_):
    """
        d is a value between 0 and 1.0 that is an indicator to 
        how close something is to the end or from the begining
        0 being the start and 1.0 being the end.
    """
    d = float(d)
    anneal_type = settings_['annealing_schedule']
    if (anneal_type == 'linear'):
        p = 1.0 - (d)
    elif (anneal_type == "log"):
        # p = ((0.1/math.log(((d))+1)))
        p = (1.0 - (math.log((d)+1.0)))**settings_['initial_temperature']
    elif (anneal_type == "square"):
        d = 1.0 - (d)
        p = (d**2)
    elif (anneal_type == "exp"):
        d = 1.0 - (d)
        p = (d**round_)
        
    return p
def discounted_rewards(rewards, discount_factor):
    from scipy import signal, misc
    """
    computes discounted sums along 0th dimension of x.
    inputs
    ------
    rewards: ndarray
    discount_factor: float
    outputs
    -------
    y: ndarray with same shape as x, satisfying
        y[t] = x[t] + gamma*x[t+1] + gamma^2*x[t+2] + ... + gamma^k x[t+k],
                where k = len(x) - t - 1
    """
    assert rewards.ndim >= 1
    return scipy.signal.lfilter([1],[1,-discount_factor],rewards[::-1], axis=0)[::-1]


    
def compute_advantage(discounted_rewards, rewards, discount_factor):
    """
    As 
    """
    adv = []
    for i in range(len(discounted_rewards)-1):
        # adv.append([((discount_factor * discounted_rewards[i+1]) + (rewards[i])) - discounted_rewards[j]])
        adv.append(((discounted_rewards[i+1])) - discounted_rewards[i])
        # adv.append([discounted_rewards[i] - (discounted_rewards[i+1] )])
        # print ("computing advantage discounts: ", adv)
    return adv

def compute_advantage_(vf, paths, gamma, lam):
    # Compute return, baseline, advantage
    for path in paths:
        path["return"] = discounted_rewards(path["reward"], gamma)
        # q_values
        # print("States shape: ", path['states'].shape)
        # print("States: ", path['states'])
        # print("reward shape: ", path['reward'].shape)
        # print("reward: ", path['reward'])
        b = path["baseline"] = vf.q_values(path['states'])
        # print("Baseline: ", b.shape)
        b1 = np.append(b, 0 if path["terminated"] else b[-1])
        b1 = np.reshape(b1, (-1,1))
        # print ("b1: ", b1.shape)
        deltas = path["reward"] + gamma*b1[1:] - b1[:-1]
        # print ("deltas: ", deltas.shape) 
        path["advantage"] = discounted_rewards(deltas, gamma * lam)
    # alladv = np.concatenate([path["advantage"] for path in paths])    
    # Standardize advantage
    """
    std = alladv.std()
    mean = alladv.mean()
    for path in paths:
        path["advantage"] = (path["advantage"] - mean) / std
    """
    return paths[0]["advantage"]

def btVectorToNumpy(vec):
    return np.array([vec.x(), vec.y(), vec.z()])

def thompsonExploration(model, exploration_rate, state_):
    """
        This exploration technique uses thompson exploration
        This generates and action wrt variance of the model and the mean
        a = mean + var.sample()
         
    """
    pa = model.predict(state_)
    pa_var = model.predictWithDropout(state_)
    diff = pa_var - pa
    print ("Thmopson Diff: ", diff)
    # out = pa + (diff * exploration_rate)
    return pa_var
    

def eGreedy(pa1, ra2, e):
    """
        epsilon greedy action select
        pa1 is best action from policy
        ra1 is the random action
        e is proabilty to select random action
        0 <= e < 1.0
    """
    r = random.random()
    if r < e:
        return ra2
    else:
        return pa1
    
def eOmegaGreedy(pa1, ra1, ra2, e, omega):
    """
        epsilon greedy action select
        pa1 is best action from policy
        ra1 is the noisy policy action action
        ra2 is the random action
        e is proabilty to select random action
        0 <= e < omega < 1.0
    """
    r = random.random()
    if r < e:
        return ra2
    elif r < omega:
        return ra1
    else:
        return pa1

def norm_reward(reward, reward_bounds):
    return norm_action(reward, reward_bounds)

def scale_reward(reward, reward_bounds):
    return scale_action(reward, reward_bounds)

def _scale_reward(diff, reward_bounds):
    var = (np.array(reward_bounds[1]) - np.array(reward_bounds[0]))/2.0
    return diff/var

def norm_state(state, max_state):
    return norm_action(action_=state, action_bounds_=max_state)

def scale_state(state, max_state):
    return scale_action(normed_action_=state, action_bounds_=max_state)

def norm_action(action_, action_bounds_):
    """
        
        Normalizes the action 
        Where the middle of the action bounds are mapped to 0
        upper bound will correspond to 1 and -1 to the lower
        from environment space to normalized space
        
        norm_action = ( action - mean ) / var
    """
    
    
    avg = (action_bounds_[0] + action_bounds_[1])/2.0
    std = (action_bounds_[1] - action_bounds_[0])/2.0
    # return (action_ - (avg)) / (action_bounds_[1]-avg)
    return (action_ - (avg)) / (std)

def norm_actions(actions_, action_bounds_):
    """
        
        Normalizes the action 
        Where the middle of the action bounds are mapped to 0
        upper bound will correspond to 1 and -1 to the lower
        from environment space to normalized space
        
        norm_action = ( action - mean ) / var
    """
    
    
    avg = (action_bounds_[0] + action_bounds_[1])/2.0
    std = (action_bounds_[1] - action_bounds_[0])/2.0
    # return (action_ - (avg)) / (action_bounds_[1]-avg)
    return (actions_ - (std)) / (var)

def scale_action(normed_action_, action_bounds_):
    """
        from normalize space back to environment space
        Normalizes the action 
        Where 0 in the action will be mapped to the middle of the action bounds
        1 will correspond to the upper bound and -1 to the lower
    """
    avg = (action_bounds_[0] + action_bounds_[1])/2.0
    std = (action_bounds_[1] - action_bounds_[0])/2.0
    # return normed_action_ * (action_bounds_[1] - avg) + avg
    return (normed_action_ * (std)) + avg

def rescale_action(normed_action_, action_bounds_):
    """
        from normalize space back to environment space
        Scales the action 
        Just scales the input wrt the given action bounds
    """
    
    std = (action_bounds_[1] - action_bounds_[0])/2.0
    # return normed_action_ * (action_bounds_[1] - avg) + avg
    return (normed_action_ * (std)) 

def action_bound_std(action_bounds_):
    # avg = (action_bounds_[0] + action_bounds_[1])/2.0
    std = (action_bounds_[1] - action_bounds_[0])/2.0
    return std

def getSettings(settingsFileName):
    file = open(settingsFileName)
    settings = json.load(file)
    # print ("Settings: " + str(json.dumps(settings)))
    file.close()
    
    return settings

def randomExporation(explorationRate, actionV):
    out = []
    for i in range(len(actionV)):
        out.append(actionV[i] + np.random.normal(0, explorationRate, 1)[0])
    return out

"""
def randomExporation(actionV, std, bounds):
    
    out = []
    for i in range(len(actionV)):
        scale = (bounds[1][i]-bounds[0][i])
        while True:
            ## resample noise that is greater than std*3 away
            n = np.random.normal(0, explorationRate, 1)[0]
            if (np.abs(n) < (explorationRate*3)):
                break
        n = n * scale
        out.append(actionV[i] + n)
    return out
"""

def randomExporationSTD(actionV, std, bounds=None):
    """
        This version scales the exploration noise wrt the action bounds
    """
    # print ("std: ", std)
    out = []
    for j in range(len(actionV)):
        out_ = [] 
        for i in range(len(actionV[j])):
            ## I think this should have a /2.0 want to map 1 - -1 to this interval
            # scale = (bounds[1][i]-bounds[0][i])/2.0
            while True:
                ## resample noise that is greater than std*3 away
                n = np.random.normal(0, std[j][i], 1)[0]
                if (np.abs(n) < (std[j][i]*3)):
                    break
            # n = (np.random.randn() * std[i]) 
                ## Scale std wrt action bounds
            # n = n * scale
            #    if (np.abs(n) < (std[i]*3)):
            #        break
            # n = n * scale
            
            out_.append(actionV[j][i] + n)
        out.append(out_)
    return out

def OUNoise(theta, sigma, previousNoise):
    """
        Ornstein–Uhlenbeck process
    
        d x t = θ ( μ − x t ) d t + σ d W t {\displaystyle dx_{t}=\theta (\mu -x_{t})\,dt+\sigma \,dW_{t}} {\displaystyle dx_{t}=\theta (\mu -x_{t})\,dt+\sigma \,dW_{t}}  
    """
    
    dWt = np.random.normal(0.0,1.0, size=previousNoise.shape[0])
    dx_t = theta * (0.0 - previousNoise) + (sigma * dWt)
    return dx_t

def randomUniformExporation(bounds):
    out = []
    for i in range(len(bounds[0])):
        out.append(np.random.uniform(bounds[0][i],bounds[1][i],1)[0])
    return out

def randomUniformExporation(explorationRate, actionV, bounds):
    out = []
    for i in range(len(bounds[0])):
        r = np.random.uniform(-1.0,1,1)[0] * explorationRate
        out.append( actionV[i] + (r * ( bounds[1][i] - bounds[0][i] ) ) )
    return out

def clampAction(actionV, bounds):
    """
    bounds[0] is lower bounds
    bounds[1] is upper bounds
    """
    actionV_ = copy.deepcopy(actionV)
    for i in range(len(actionV_)):
        if actionV_[i] < bounds[0][i]:
            actionV_[i] = bounds[0][i]
        elif actionV_[i] > bounds[1][i]:
            actionV_[i] = bounds[1][i]
    return actionV_

def clampActionWarn(actionV, bounds):
    """
    bounds[0] is lower bounds
    bounds[1] is upper bounds
    Returns True if the actionV was outside the bounds
    """
    out=False
    actionV_ = copy.deepcopy(actionV)
    for j in range(len(actionV_)): 
        for i in range(len(actionV_[j])):
            if actionV_[j][i] < bounds[0][i]:
                actionV_[j][i] = bounds[0][i]
                out=True
            elif actionV_[j][i] > bounds[1][i]:
                actionV_[j][i] = bounds[1][i]
                out=True
    return (actionV_, out)

def reward_smoother(diff_, settings, _weight):
    if (settings['reward_smoother'] == 'abs'):
        return np.exp(np.abs(diff_)*_weight)
    elif (settings['reward_smoother'] == 'gaussian'):
        return np.exp((diff_*diff_)*_weight)
    elif (settings['reward_smoother'] == 'linear'):
        return ((np.abs(diff_))*_weight)
    else:
        print("Reward smoother unknown: ", settings['reward_smoother'])
        sys.exit(-1)

def loglikelihood(a, mean0, std0, d):
    """
        d is the number of action dimensions
    """
    
    # exp[ -(a - mu)^2/(2*sigma^2) ] / sqrt(2*pi*sigma^2)
    diff = np.square(a - mean0)
    print ("norm0: ", (a[0]-mean0[0]))
    print ("a.shape: ", a.shape, " mean0.shape: ", mean0.shape)
    print ("diff: ", diff.shape, diff)
    print ("diff/std0: ", diff/std0)
    # print ("log(diff/std0): ", np.log(diff/std0))
    diff = (diff/std0).sum(axis=1) * -0.5
    print("np.log(std0).sum(axis=1): ", np.log(std0).sum(axis=1))
    print ("Final: ", diff - 0.5 * np.log(2.0 * np.pi) * d - np.log(std0).sum(axis=1))
    # print ("Final: ", - 0.5 * np.square(diff/std0) - 0.5 * np.log(2.0 * np.pi) * d - np.log(std0))
    # square_return = (- 0.5 * np.square((a - mean0) / std0) - 0.5 * np.log(2.0 * np.pi) * d - np.log(std0)).sum(axis=1)
    # print ("FInal2: ", square_return)
    return np.reshape(- 0.5 * np.square((a - mean0) / std0).sum(axis=1) - 0.5 * np.log(2.0 * np.pi) * d - np.log(std0).sum(axis=1), newshape=(a.shape[0], 1))


def likelihood(a, mean0, std0, d):
    return np.exp(loglikelihood(a, mean0, std0, d))


"""
def initSimulation(settings):

    print ("Sim config file name: " + str(settings["sim_config_file"]))
    c = characterSim.Configuration(str(settings["sim_config_file"]))
    # c = characterSim.Configuration("../data/epsilon0Config.ini")
    action_space_continuous=settings['action_space_continuous']
    if action_space_continuous:
        action_bounds = np.array(settings["action_bounds"], dtype=float)
        
    
    # this is the process that selects which game to play
    exp = characterSim.Experiment(c)
        
    
    if ( "Deep_NN2" == model_type):
        model = RLDeepNet(n_in=len(state_bounds[0]), n_out=num_actions, state_bounds=state_bounds, 
                          action_bounds=None, reward_bound=reward_bounds)
        train_forward_dynamics=False
    elif (model_type == "Deep_NN3" ):
        model = DeepRLNet3(n_in=len(state_bounds[0]), n_out=num_actions, state_bounds=state_bounds, 
                          action_bounds=None, reward_bound=reward_bounds)
        train_forward_dynamics=False
    elif (model_type == "Deep_CACLA" ):
        model = DeepCACLA(n_in=len(state_bounds[0]), n_out=len(action_bounds[0]), state_bounds=state_bounds, 
                          action_bounds=action_bounds, reward_bound=reward_bounds)
        omega = settings["omega"]
        exploration_rate = settings["exploration_rate"]
        train_forward_dynamics=True
    else:
        print ("Unknown model type: " + str(model_type))
    state = characterSim.State()
    
    exp.getActor().init()
    exp.init()
    
    output={}
    output['exp']=exp
    output['model']=model
    
    return output
"""

def getMBAEAction(forwardDynamicsModel, model, state):
    action = model.predict(state)
    return getMBAEAction2(forwardDynamicsModel, model, action, state)

def getMBAEAction2(forwardDynamicsModel, model, action, state):
    """
        Computes the optimal action to be taken given
        the forwardDynamicsModel f and
        the value function (model) v
    """
    learning_rate=model.getSettings()['action_learning_rate']
    num_updates=model.getSettings()['num_mbae_steps']
    state_length = model.getStateSize()
    init_value = model.q_value(state)
    """
    fake_state_ = copy.deepcopy(state)
    for i in range(num_updates):
        fake_state_ = fake_state_ + ( model.getGrads(fake_state_)[0] * learning_rate )
        print ("Fake state Value: ", model.q_value(fake_state_))
    """
    init_action = copy.deepcopy(action)
    init_reward = final_reward = forwardDynamicsModel.predict_reward(state, action)
    for i in range(num_updates):
        ## find next state with dynamics model
        next_reward = np.reshape(forwardDynamicsModel.predict_reward(state, action), (1, 1))
        ## Set modified next state as output for dynamicsModel
        ## Compute the grad to change the input to produce a better action
        dynamics_grads = forwardDynamicsModel.getRewardGrads(np.reshape(state, (1, model.getStateSize())), np.reshape(action, (1, model.getActionSize())), np.reshape(next_reward+0.1, (1, 1)))[0]
        ## Grab the part of the grads that is the action
        action_grads = dynamics_grads[:, state_length:] * learning_rate 
        action = action + action_grads
        action = action[0]
        # next_state_ = np.reshape(forwardDynamicsModel.predict(state, action), (1, model.getStateSize()))
        
    # final_value = model.q_value(next_state_)
    final_reward = forwardDynamicsModel.predict_reward(state, action)
        
        # repeat
    if (model.getSettings()["print_levels"][model.getSettings()["print_level"]] >= model.getSettings()["print_levels"]['debug']):
        print ("New action: ", action, " action diff: ", (action - init_action), " reward change: ", 
           (final_reward - init_reward))
    action = clampAction(action, model._action_bounds)
    return action

def getOptimalAction(forwardDynamicsModel, model, state, action_lr, use_random_action=False, p=1.0):

    if (model.getSettings()['num_mbae_steps'] > 1):
        new_action = sampleActions(forwardDynamicsModel, model, state, action_lr, use_random_action=use_random_action, p=p)
    else:
        new_action = getOptimalAction2(forwardDynamicsModel, model, state, action_lr, use_random_action=use_random_action, p=p)
    """
    diff__ = new_action[0] - action
    if ( ('print_level' in model.getSettings()) and (model.getSettings()["print_level"]== 'debug') ):
        print ("MBAE action change: ", (np.sqrt((diff__*diff__).sum())), " values: ",  new_action[0] - action)
    """
    return new_action

def sampleActions(forwardDynamicsModel, model, state, action_lr, use_random_action=False, p=1.0):
    """
        Computes the optimal action to be taken given
        the forwardDynamicsModel f and
        the value function (model) v
    """
    learning_rate=action_lr
    num_samples=model.getSettings()['num_mbae_steps']
    state_length = model.getStateSize()
    init_value = model.q_value(state)
    # print ("Initial value: ", init_value)
    action_grads_tmp = np.array(model.getSettings()["action_bounds"][0]) * 0.0
    action_tmp = np.array(model.getSettings()["action_bounds"][0]) * 0.0
    next_state_tmp = state * 0.0
    for i in range(num_samples):
        action = model.predict(state)
        if ( use_random_action ):
            action_bounds = np.array(model.getSettings()["action_bounds"], dtype=model.getSettings()["float_type"])
            std_ = model.predict_std(state)
            # print("Annealing random action explore for MBAE: ", p)
            std_ = std_ * p
            action = randomExporationSTD(action, std_, action_bounds)
        init_action = copy.deepcopy(action)
        ## find next state with dynamics model
        next_state = np.reshape(forwardDynamicsModel.predict(state, action), (1, model.getStateSize()))
        if (model.getSettings()["print_levels"][model.getSettings()["print_level"]] >= model.getSettings()["print_levels"]['debug']):
            print(" MBAE mean: ", next_state)
        """
        if ('use_stochastic_forward_dynamics' in model.getSettings() and 
            (model.getSettings()['use_stochastic_forward_dynamics'] == True)):
            std = forwardDynamicsModel.predict_std(state, action)
            if (model.getSettings()["print_levels"][model.getSettings()["print_level"]] >= model.getSettings()["print_levels"]['debug']):
                print ("SMBAE std: ", std)
            if ('num_stochastic_forward_dynamics_samples' in model.getSettings()):
                next_states = []
                for ns in range(model.getSettings()['num_stochastic_forward_dynamics_samples']):
                    next_states.append(randomExporationSTD(0, next_state, std))
                next_state = np.mean(next_states, axis=0)
            else:
                next_state = randomExporationSTD(0, next_state, std)
        elif ('use_stochastic_forward_dynamics' in model.getSettings() and 
            (model.getSettings()['use_stochastic_forward_dynamics'] == "dropout")):
            # print("Getting fd dropout sample:")
            next_state = np.reshape(forwardDynamicsModel.predictWithDropout(state, action), (1, model.getStateSize()))
            # next_state = forwardDynamicsModel.predictWithDropout
        """
        # value_ = model.q_value(next_state)
        # print ("next state q value: ", value_)
        # print ("Next State: ", next_state.shape)
        ## compute grad for next state wrt model, i.e. how to change the state to improve the value
        """
        if ( 'optimize_advantage_for_MBAE' in model.getSettings() and  model.getSettings()['optimize_advantage_for_MBAE'] ):
            next_state_grads = model.getAdvantageGrads(state, next_state)[0] # this uses the value function
            if (model.getSettings()["print_levels"][model.getSettings()["print_level"]] >= model.getSettings()["print_levels"]['debug']):
                print ( "Advantage grad: ", next_state_grads)
                next_state_grads__ = model.getGrads(next_state)[0] # this uses the value function
                print ( "Q-function grad: ", next_state_grads__)
        else:
            next_state_grads = model.getGrads(next_state)[0] # this uses the value function
        """
        next_state_grads = model.getGrads(next_state)[0] # this uses the value function
        ## normalize
        forwardDynamicsModel.setGradTarget(next_state_grads)
        # next_state_grads = (next_state_grads/(np.sqrt((next_state_grads*next_state_grads).sum()))) * (learning_rate)
        # if (model.getSettings()["print_levels"][model.getSettings()["print_level"]] >= model.getSettings()["print_levels"]['debug'])::
        #    print ("Next State Grad: ", next_state_grads)
        # next_state_grads = rescale_action(next_state_grads, model.getStateBounds())
        # next_state_grads = np.sum(next_state_grads, axis=1)
        # print ("Next State Grad shape: ", next_state_grads.shape)
        ## modify next state wrt increasing grad, this is the direction we want the next state to go towards 
        # next_state = next_state + next_state_grads
        # print ("Next State: ", next_state)
        # value_ = model.q_value(next_state)
        # print ("Updated next state q value: ", value_)
        # Set modified next state as output for dynamicsModel
        # print ("Next State2: ", next_state)
        # compute grad to find
        # next_state = np.reshape(next_state, (model.getStateSize(), 1))
        # uncertanty = getModelValueUncertanty(model, next_state[0])
        # print ("Uncertanty: ", uncertanty)
        ## Compute the grad to change the input to produce the new target next state
        ## We will want to use the negative of this grad because the cost function is L2, 
        ## the grad will make this bigger, user - to pull action towards target action using this loss function 
        dynamics_grads = forwardDynamicsModel.getGrads(np.reshape(state, (1, model.getStateSize())),
                                                        np.reshape(action, (1, model.getActionSize())),
                                                         np.reshape(next_state, (1, model.getStateSize())), 
                                                         v_grad=np.reshape(next_state_grads, (1, model.getStateSize())))
        dynamics_grads = dynamics_grads[0]
        # print ("action_grad1: ", action_grads)
        
        if (model.getSettings()["print_levels"][model.getSettings()["print_level"]] >= model.getSettings()["print_levels"]['debug']):
            print ("fd dynamics_grads: ", dynamics_grads)
            print ("fd dynamics_grads magnitude: ", np.sqrt((dynamics_grads*dynamics_grads).sum()))
        
        if ( model.getSettings()['train_reward_predictor']):
            reward_grad = forwardDynamicsModel.getRewardGrads(np.reshape(state, (1, model.getStateSize())),
                                                        np.reshape(action, (1, model.getActionSize())))[0]
            ## Need to shrink this grad down to the same scale as the value function
            reward_grad = (reward_grad * (1.0 - model.getSettings()['discount_factor']))
            """
            if ( model.getSettings()['optimize_advantage_for_MBAE'] ):
                state_grads_ = model.getGrads(state)[0] # this uses the value function
                if (model.getSettings()["print_levels"][model.getSettings()["print_level"]] >= model.getSettings()["print_levels"]['debug']):
                    print("State_Grad Raw: ", state_grads_)
                    print ("State_Grad magnitude: ", np.sqrt((state_grads_*state_grads_).sum()))
                dynamics_grads =  ((reward_grad) + (dynamics_grads *  model.getSettings()['discount_factor']) -
                                   state_grads_ * model.getSettings()['discount_factor'])
            else:    
            """                                    
            dynamics_grads =  (reward_grad) + (dynamics_grads *  model.getSettings()['discount_factor'])
            if (model.getSettings()["print_levels"][model.getSettings()["print_level"]] >= model.getSettings()["print_levels"]['debug']):
                print("Reward_Grad Raw: ", reward_grad)
                print ("Reward_Grad magnitude: ", np.sqrt((reward_grad*reward_grad).sum()))
                
            action_grads_tmp = action_grads_tmp + dynamics_grads
            action_tmp = action_tmp + action
        ## Grab the part of the grads that is the action
        # action_grads = dynamics_grads[:, state_length:] * learning_rate
        
        action_grads_tmp = action_grads_tmp / float(num_samples)
        action_grads = action_grads_tmp
        action_ = action_tmp / float(num_samples)
        """ 
        if ('use_dpg_grads_for_MBAE' in model.getSettings() and model.getSettings()['use_dpg_grads_for_MBAE']):
            action_grads = model.getActionGrads(state)[0]
            if (model.getSettings()["print_levels"][model.getSettings()["print_level"]] >= model.getSettings()["print_levels"]['debug']):
                print("Using DPG action grads for MBAE")
        """    
        if (model.getSettings()["print_levels"][model.getSettings()["print_level"]] >= model.getSettings()["print_levels"]['debug']):
            print( "Raw action grad: ", action_grads)
        ## Normalize action length
        
        
    action_grads = ( action_grads / np.std(action_grads) ) * (learning_rate)
    if ('randomize_MBAE_action_length' in model.getSettings() and ( model.getSettings()['randomize_MBAE_action_length'])):
        # action_grads = action_grads * np.random.uniform(low=0.0, high = 1.0, size=1)[0]
        action_grads = action_grads * (np.fabs(np.random.normal(loc=0.0, scale = 1.0, size=1)[0]))
        
    ## Scale action by action bounds
    action_grads = rescale_action(action_grads, model.getActionBounds())
    if (model.getSettings()["print_levels"][model.getSettings()["print_level"]] >= model.getSettings()["print_levels"]['debug']):
        print ("Applied action: ", action_grads)
        print ("Action magnitude: ", np.sqrt((action_grads*action_grads).sum()))
    # action_grads = action_grads * learning_rate
    # print ("action_grad2: ", action_grads)
    ## Use grad to update action parameters
    action = action_ + action_grads
    # print ("action_grad: ", action_grads, " new action: ", action)
    # print ( "Action shape: ", action.shape)
    # print (" Action diff: ", (action - init_action))
    next_state_ = np.reshape(forwardDynamicsModel.predict(state, action), (1, model.getStateSize()))
        
        # print ("Next_state: ", next_state_.shape, " values ", next_state_)
    final_value = model.q_value(next_state_)
    if ( model.getSettings()['train_reward_predictor']):
        reward = forwardDynamicsModel.predict_reward(state, action)
        # print ("Estimated reward: ", reward)
        final_value = reward + (model.getSettings()['discount_factor'] * final_value)

    
        # print ("Final Estimated Value: ", final_value)
        
        # repeat
    value_diff = final_value - init_value
    # if (model.getSettings()["print_levels"][model.getSettings()["print_level"]] >= model.getSettings()["print_levels"]['debug'])::
    #    print ("New action: ", action, " action diff: ", (action - init_action), " value change: ", 
    #           (value_diff))
    #    print ("dynamics_grads: ", dynamics_grads)
    # action = clampAction(action, model._action_bounds)
    if (checkDataIsValid(action)):
        ### Because there are some nan values coming out of here.
        return (action, value_diff)
    else:
        print("MBAE, action invalid: ", action)
        return (init_action, 0)

def getOptimalAction2(forwardDynamicsModel, model, state, action_lr, use_random_action=False, p=1.0):
    """
        Computes the optimal action to be taken given
        the forwardDynamicsModel f and
        the value function (model) v
    """
    action = model.predict(state)
    if ( use_random_action ):
        action_bounds = np.array(model.getSettings()["action_bounds"], dtype=model.getSettings()["float_type"])
        std_ = model.predict_std(state)
        # print("Annealing random action explore for MBAE: ", p)
        std_ = std_ * p
        action = randomExporationSTD(action, std_, action_bounds)

    learning_rate=action_lr
    num_updates=model.getSettings()['num_mbae_steps']
    state_length = model.getStateSize()
    init_value = model.q_value(state)
    """
    fake_state_ = copy.deepcopy(state)
    for i in range(num_updates):
        fake_state_ = fake_state_ + ( model.getGrads(fake_state_)[0] * learning_rate )
        print ("Fake state Value: ", model.q_value(fake_state_))
    """
    init_action = copy.deepcopy(action)
    for i in range(num_updates):
        ## find next state with dynamics model
        next_state = np.reshape(forwardDynamicsModel.predict(state, action), (1, model.getStateSize()))
        
        if (model.getSettings()["print_levels"][model.getSettings()["print_level"]] >= model.getSettings()["print_levels"]['debug']):
            print(" MBAE mean: ", next_state)
        if ('use_stochastic_forward_dynamics' in model.getSettings() and 
            (model.getSettings()['use_stochastic_forward_dynamics'] == True)):
            std = forwardDynamicsModel.predict_std(state, action)
            if (model.getSettings()["print_levels"][model.getSettings()["print_level"]] >= model.getSettings()["print_levels"]['debug']):
                print ("SMBAE std: ", std)
            if ('num_stochastic_forward_dynamics_samples' in model.getSettings()):
                next_states = []
                for ns in range(model.getSettings()['num_stochastic_forward_dynamics_samples']):
                    next_states.append(randomExporationSTD(0, next_state, std))
                next_state = np.mean(next_states, axis=0)
            else:
                next_state = randomExporationSTD(0, next_state, std)
        elif ('use_stochastic_forward_dynamics' in model.getSettings() and 
            (model.getSettings()['use_stochastic_forward_dynamics'] == "dropout")):
            # print("Getting fd dropout sample:")
            next_state = np.reshape(forwardDynamicsModel.predictWithDropout(state, action), (1, model.getStateSize()))
            # next_state = forwardDynamicsModel.predictWithDropout
        value_ = model.q_value(next_state)
        # print ("next state q value: ", value_)
        # print ("Next State: ", next_state.shape)
        ## compute grad for next state wrt model, i.e. how to change the state to improve the value
        if ( 'optimize_advantage_for_MBAE' in model.getSettings() and  model.getSettings()['optimize_advantage_for_MBAE'] ):
            next_state_grads = model.getAdvantageGrads(state, next_state)[0] # this uses the value function
            if (model.getSettings()["print_levels"][model.getSettings()["print_level"]] >= model.getSettings()["print_levels"]['debug']):
                print ( "Advantage grad: ", next_state_grads)
                next_state_grads__ = model.getGrads(next_state)[0] # this uses the value function
                print ( "Q-function grad: ", next_state_grads__)
        else:
            next_state_grads = model.getGrads(next_state)[0] # this uses the value function
        next_state_grads = model.getGrads(next_state)[0] # this uses the value function
        ## normalize
        forwardDynamicsModel.setGradTarget(next_state_grads)
        # next_state_grads = (next_state_grads/(np.sqrt((next_state_grads*next_state_grads).sum()))) * (learning_rate)
        if (model.getSettings()["print_levels"][model.getSettings()["print_level"]] >= model.getSettings()["print_levels"]['debug']):
            print ("Next State Grad: ", next_state_grads)
        ## modify next state wrt increasing grad, this is the direction we want the next state to go towards 
        value_ = model.q_value(next_state)
        # print ("Updated next state q value: ", value_)
        # Set modified next state as output for dynamicsModel
        # print ("Next State2: ", next_state)
        # compute grad to find
        # next_state = np.reshape(next_state, (model.getStateSize(), 1))
        # uncertanty = getModelValueUncertanty(model, next_state[0])
        # print ("Uncertanty: ", uncertanty)
        ## Compute the grad to change the input to produce the new target next state
        ## We will want to use the negative of this grad because the cost function is L2, 
        ## the grad will make this bigger, user - to pull action towards target action using this loss function 
        dynamics_grads = forwardDynamicsModel.getGrads(np.reshape(state, (1, model.getStateSize())),
                                                        np.reshape(action, (1, model.getActionSize())),
                                                         np.reshape(next_state, (1, model.getStateSize())), 
                                                         v_grad=np.reshape(next_state_grads, (1, model.getStateSize())))
        """
        if (model.getSettings()["print_levels"][model.getSettings()["print_level"]] >= model.getSettings()["print_levels"]['debug']):
            print("Fd network parameters: ", forwardDynamicsModel.getNetworkParameters()[0])
            print ("Full fd grads: ", dynamics_grads)
        """
        dynamics_grads = dynamics_grads[0]
        # print ("action_grad1: ", action_grads)
        
        if (model.getSettings()["print_levels"][model.getSettings()["print_level"]] >= model.getSettings()["print_levels"]['debug']):
            print ("fd dynamics_grads: ", dynamics_grads)
            print ("fd dynamics_grads magnitude: ", np.sqrt((dynamics_grads*dynamics_grads).sum()))
        
        if ( model.getSettings()['train_reward_predictor']):
            reward_grad = forwardDynamicsModel.getRewardGrads(np.reshape(state, (1, model.getStateSize())),
                                                        np.reshape(action, (1, model.getActionSize())))[0]
            ## Need to shrink this grad down to the same scale as the value function
            reward_grad = (reward_grad * (1.0 - model.getSettings()['discount_factor']))
            """
            if ( model.getSettings()['optimize_advantage_for_MBAE'] ):
                state_grads_ = model.getGrads(state)[0] # this uses the value function
                if (model.getSettings()["print_levels"][model.getSettings()["print_level"]] >= model.getSettings()["print_levels"]['debug']):
                    print("State_Grad Raw: ", state_grads_)
                    print ("State_Grad magnitude: ", np.sqrt((state_grads_*state_grads_).sum()))
                dynamics_grads =  ((reward_grad) + (dynamics_grads *  model.getSettings()['discount_factor']) -
                                   state_grads_ * model.getSettings()['discount_factor'])
            else:    
            """                                    
            dynamics_grads =  (reward_grad) + (dynamics_grads *  model.getSettings()['discount_factor'])
            if (model.getSettings()["print_levels"][model.getSettings()["print_level"]] >= model.getSettings()["print_levels"]['debug']):
                print("Reward_Grad Raw: ", reward_grad)
                print ("Reward_Grad magnitude: ", np.sqrt((reward_grad*reward_grad).sum()))
        ## Grab the part of the grads that is the action
        # action_grads = dynamics_grads[:, state_length:] * learning_rate
        action_grads = dynamics_grads 
        if ('use_dpg_grads_for_MBAE' in model.getSettings() and model.getSettings()['use_dpg_grads_for_MBAE']):
            action_grads = model.getActionGrads(state)[0]
            if (model.getSettings()["print_levels"][model.getSettings()["print_level"]] >= model.getSettings()["print_levels"]['debug']):
                print("Using DPG action grads for MBAE")
            
        if (model.getSettings()["print_levels"][model.getSettings()["print_level"]] >= model.getSettings()["print_levels"]['debug']):
            print( "Raw action grad: ", action_grads)
        ## Normalize action length
        # action_grads = ( (action_grads/(np.sqrt((action_grads*action_grads).sum())))/np.sqrt(np.mean(np.abs(action_grads)))) * (learning_rate)
        action_grads = ( action_grads / np.std(action_grads) ) * (learning_rate)
        # print ("MBAE learning rate: ", learning_rate, " ", model.getSettings()['randomize_MBAE_action_length'])
        if ('randomize_MBAE_action_length' in model.getSettings()
             and ( model.getSettings()['randomize_MBAE_action_length'] == True)):
            # print ("Adding noise to action grads")
            # action_grads = action_grads * np.random.uniform(low=0.0, high = 1.0, size=1)[0]
            action_grads = action_grads * (np.fabs(np.random.normal(loc=0.0, scale = 1.0, size=1)[0]))
            
        if (model.getSettings()["print_levels"][model.getSettings()["print_level"]] >= model.getSettings()["print_levels"]['debug']):
            print ("Applied action: ", action_grads)
            # print ("Action magnitude: ", np.sqrt((action_grads*action_grads).sum()), " mean, ", np.mean(np.abs(action_grads)))
            print ("Action magnitude: ", np.sqrt((action_grads*action_grads).sum()), " std, ", np.std(action_grads))
            
        ## Scale action by action bounds
        action_grads = rescale_action(action_grads, model.getActionBounds())
        if (model.getSettings()["print_levels"][model.getSettings()["print_level"]] >= model.getSettings()["print_levels"]['debug']):
            print ("Applied action: ", action_grads)
            print ("Action magnitude: ", np.sqrt((action_grads*action_grads).sum()))
        # action_grads = action_grads * learning_rate
        # print ("action_grad2: ", action_grads)
        ## Use grad to update action parameters
        action = action + action_grads
        action = action
        # print ("action_grad: ", action_grads, " new action: ", action)
        # print ( "Action shape: ", action.shape)
        # print (" Action diff: ", (action - init_action))
        next_state_ = np.reshape(forwardDynamicsModel.predict(state, action), (1, model.getStateSize()))
        
        # print ("Next_state: ", next_state_.shape, " values ", next_state_)
    final_value = model.q_value(next_state_)
    if ( model.getSettings()['train_reward_predictor']):
        reward = forwardDynamicsModel.predict_reward(state, action)
        # print ("Estimated reward: ", reward)
        final_value = reward + (model.getSettings()['discount_factor'] * final_value)

    
        # print ("Final Estimated Value: ", final_value)
        
        # repeat
    ### This should be higher
    value_diff = final_value - init_value
    if (model.getSettings()["print_levels"][model.getSettings()["print_level"]] >= model.getSettings()["print_levels"]['debug']):
        print ("New action: ", action, " action diff: ", (action - init_action), " value change: ", 
               (value_diff))
        print ("dynamics_grads: ", dynamics_grads)
    # action = clampAction(action, model._action_bounds)
    if (checkDataIsValid(action)):
        ### Because there are some nan values coming out of here.
        return (action, value_diff)
    else:
        print("MBAE, action invalid: ", action)
        return (init_action, 0)

def getModelPredictionUncertanty(model, state, length=4.1, num_samples=32):
    """
        Computes the optimal action to be taken given
        the forwardDynamicsModel f and
        the value function (model) v
    """
    lSquared =(length**2)
    modelPrecsionInv = ((lSquared * (1.0 - model.getSettings()['dropout_p'])) / 
                        (2*model.getSettings()['expereince_length']*
                         model.getSettings()['regularization_weight'] ))**-1
    # print "Model Precision Inverse:" + str(modelPrecsionInv)
    predictions_ = []
    samp_ = np.repeat(np.array(state),num_samples, axis=0)
    # print "Sample: " + str(samp_)
    for pi in range(num_samples):
        predictions_.append( (model.predictWithDropout(samp_[pi])))
    # print "Predictions: " + str(predictions_)
    variance__ = (modelPrecsionInv) + np.var(predictions_, axis=0)
    return variance__

def getModelValueUncertanty(model, state, length=4.1, num_samples=32):
    """
        Computes the optimal action to be taken given
        the forwardDynamicsModel f and
        the value function (model) v
    """
    lSquared =(length**2)
    modelPrecsionInv = ((lSquared * (1.0 - model.getSettings()['dropout_p'])) / 
                        (2*model.getSettings()['expereince_length']*
                         model.getSettings()['regularization_weight'] ))**-1
    # print "Model Precision Inverse:" + str(modelPrecsionInv)
    predictions_ = []
    samp_ = np.repeat(np.array(state),num_samples, axis=0)
    # print "Sample: " + str(samp_)
    for pi in range(num_samples):
        predictions_.append( (model.q_valueWithDropout(samp_[pi])))
    # print "Predictions: " + str(predictions_)
    variance__ = (modelPrecsionInv) + np.var(predictions_, axis=0)
    return variance__


def validBounds(bounds):
    """
        Checks to make sure bounds are valid
        max is > min
        and max - min > epsilon
    """
    valid = np.all(np.less(bounds[0], bounds[1]))
    if (not valid):
        less_ = np.less(bounds[0], bounds[1])
        bad_indecies = np.where(less_ == False)
        bad_values_low = bounds[0][bad_indecies]
        bad_values_high = bounds[1][bad_indecies]
        print ("Invalid bounds: ", bad_indecies )
        print ("Bad Values:", bad_values_low, bad_values_high)
        
    ##  bounds not too close to each other
    epsilon = 0.01
    bounds = np.array(bounds)
    diff = bounds[1]-bounds[0]
    valid = valid and np.all(np.greater(diff, epsilon))
    if (not valid):
        less_ = np.greater(diff, epsilon)
        bad_indecies = np.where(less_ == False)
        bad_values_low = bounds[0][bad_indecies]
        bad_values_high = bounds[1][bad_indecies]
        print ("Invalid bounds, bounds to small: ", bad_indecies )
        print ("Bad Values:", bad_values_low, bad_values_high)
        print ("Bounds to small:", np.greater(diff, epsilon))
        
    return valid

def fixBounds(bounds):
    """
        Fixes bounds that are too close together
        pre-req all(bounds[1] is > bounds[0])
    """
        
    # bounds not too close to each other
    epsilon = 0.1
    bounds = np.array(bounds)
    diff = bounds[1]-bounds[0]
    # print ("bounds: ", bounds)
    # print("diff: ", diff)
    for i in range(diff.shape[0]):
        if (diff[i] < epsilon ):
            bounds[1][i] = bounds[1][i] + epsilon
            bounds[0][i] = bounds[0][i] - epsilon
        if ((not np.isfinite(diff[i]) or (diff[i] != diff[i]))):
            ## Fix inf and nan
            bounds[1][i] = epsilon
            bounds[0][i] = -epsilon
    # print("Bounds fixed: ", bounds)
    return bounds

def checkDataIsValid(data, verbose=False):
        """
            Checks to make sure the data going into the exp buffer is not garbage...
        """
        data = np.array(data)
        if (not np.all(np.isfinite(data))):
            if ( verbose ):
                less_ = np.isfinite(data)
                bad_indecies = np.where(less_ == False)
                print ("Data not finite: ", less_ )
                print ("Bad Value indx: ", bad_indecies)
                bad_values_ = data[bad_indecies]
                print ("Bad Values: ", bad_values_)
            return False
    
        if (np.any(np.less(data, -1000.0))):
            if ( verbose ):
                less_ = np.less(data, -1000.0)
                bad_indecies = np.where(less_ == True)
                print ("Data too negative: ", less_ )
                print ("Bad Value indx: ", bad_indecies)
                bad_values_ = data[bad_indecies]
                print ("Bad Values: ", bad_values_)
            return False
        
        if (np.any(np.greater(data, 1000.0))):
            if ( verbose ):
                less_ = np.greater(data, 1000.0)
                bad_indecies = np.where(less_ == True)
                bad_values_ = data[bad_indecies]
                print ("Data too positive: ", less_ )
                print ("Bad Value indx: ", bad_indecies)
                bad_values_ = data[bad_indecies]
                print ("Bad Values: ", bad_values_)
            return False
        
        return True

def checkValidData(state, action, nextState, reward, verbose=False):
        """
            Checks to make sure the data going into the exp buffer is not garbage...
        """
        state = np.array(state)
        action = np.array(action)
        nextState = np.array(nextState)
        reward = np.array(reward)
        
        
        if (not np.all(np.isfinite(state))):
            if ( verbose ):
                less_ = np.isfinite(state)
                bad_indecies = np.where(less_ == False)
                print ("State not finite: ", less_ )
                print ("Bad Value indx: ", bad_indecies)
                bad_values_ = state[bad_indecies]
                print ("Bad Values: ", bad_values_)
            return False
        
        # print ("action: ", action)
        if (not np.all(np.isfinite(action))):
            if ( verbose ):
                less_ = np.isfinite(action)
                bad_indecies = np.where(less_ == False)
                print ("Action not finite: ", less_ )
                print ("Bad Value indx: ", bad_indecies)
                bad_values_ = action[bad_indecies]
                print ("Bad Values: ", bad_values_)
            return False
        
        if (not np.all(np.isfinite(nextState))):
            if ( verbose ):
                less_ = np.isfinite(nextState)
                bad_indecies = np.where(less_ == False)
                print ("NextState not finite: ", less_)
                print ("Bad Value indx: ", bad_indecies)
                bad_values_ = nextState[bad_indecies]
                print ("Bad Values: ", bad_values_)
            return False
        
        if (not np.all(np.isfinite(reward))):
            if ( verbose ):
                less_ = np.isfinite(reward)
                bad_indecies = np.where(less_ == False)
                bad_values_ = reward[bad_indecies]
                print ("Reward not finite: ", np.isfinite(state) )
                print ("Bad Value indx: ", bad_indecies)
                print ("Bad Values: ", bad_values_)
            return False
        
        if (np.any(np.less(state, -1000.0))):
            if ( verbose ):
                less_ = np.less(state, -1000.0)
                bad_indecies = np.where(less_ == True)
                print ("State too negative: ", less_)
                print ("Bad Value indx: ", bad_indecies)
                bad_values_ = state[bad_indecies]
                print ("Bad Values: ", bad_values_)
            return False
        
        if (np.any(np.less(nextState, -1000.0))):
            if ( verbose ):
                less_ = np.less(nextState, -1000.0)
                bad_indecies = np.where(less_ == True)
                print ("nextState too negative: ", less_ )
                print ("Bad Value indx: ", bad_indecies)
                bad_values_ = nextState[bad_indecies]
                print ("Bad Values: ", bad_values_)
            return False
        
        if (np.any(np.less(action, -1000.0))):
            if ( verbose ):
                less_ = np.less(action, -1000.0)
                bad_indecies = np.where(less_ == True)
                print ("action too negative: ", less_ )
                print ("Bad Value indx: ", bad_indecies)
                bad_values_ = action[bad_indecies]
                print ("Bad Values: ", bad_values_)
            return False
        
        if (np.any(np.greater(state, 1000.0))):
            if ( verbose ):    
                less_ = np.greater(state, 1000.0)
                bad_indecies = np.where(less_ == True)
                bad_values_ = state[bad_indecies]
                print ("State too positive: ", less_)
                print ("Bad Value indx: ", bad_indecies)
                bad_values_ = state[bad_indecies]
                print ("Bad Values: ", bad_values_)
            return False
        
        if (np.any(np.greater(nextState, 1000.0))):
            if ( verbose ):
                less_ = np.greater(nextState, 1000.0)
                bad_indecies = np.where(less_ == True)
                print ("nextState too positive: ", less_ )
                print ("Bad Value indx: ", bad_indecies)
                bad_values_ = nextState[bad_indecies]
                print ("Bad Values: ", bad_values_)
            return False
        
        if (np.any(np.greater(action, 1000.0))):
            if ( verbose ):
                less_ = np.greater(action, 1000.0)
                bad_indecies = np.where(less_ == True)
                print ("action too positive: ", less_ )
                print ("Bad Value indx: ", bad_indecies)
                bad_values_ = action[bad_indecies]
                print ("Bad Values: ", bad_values_)
            return False
        
        return True
    

if __name__ == '__main__':
    import sys
    import matplotlib.pyplot as plt
    # np.set_printoptions(threshold=np.nan)
    
    settingsFileName = sys.argv[1]
    file = open(settingsFileName)
    settings = json.load(file)
    file.close()
    num_samples = 5000
    
    action_bounds = np.array(settings["action_bounds"], dtype=float)
    # action_bounds[0][0] = 0
    reward_bounds = np.array([[-10.1],[0.0]])
    action = np.array((action_bounds[1]+action_bounds[0])/2.0) # right in the middle
    print ("Action bounds: " + str(action_bounds))
    print ("Action: " + str(action))
    print ("Action length: " + str(action.shape[0]))
    print ("Normalized Action: " + str(norm_action(action, action_bounds)) + " same action: " + str(scale_action(norm_action(action, action_bounds), action_bounds)))
    print ("Normalized Action: " + str(norm_action(action+0.5, action_bounds)) + " same action: " + str(action+0.5) + " as " + str(scale_action(norm_action(action+0.5, action_bounds), action_bounds)))
    print ("Normalized Action: " + str(norm_action(action+-0.5, action_bounds)) + " same action: " + str(action+-0.5) + " as "  + str(scale_action(norm_action(action+-0.5, action_bounds), action_bounds)))
    actions_=[]
    actions_list = []
    actions_list2 = []
    std = action_bound_std(action_bounds)
    for i in range(action.shape[0] ):
        actions_.append([])
    print ("Actions Data: ", actions_)
    
    for i in range(num_samples):
        action_ = randomExporation(settings["exploration_rate"], action, action_bounds)
        actions_list.append(action_)
        print (" Exploration action: ", action_)
        for j in range(action.shape[0]):
            act_ = action_[j]
            actions_[j].append(act_)
            
    for i in range(num_samples):
        action_ = randomExporationSTD(settings["exploration_rate"], actions_list[i], std, action_bounds)
        # action_ = randomExporation(settings["exploration_rate"], actions_list[i], action_bounds)
        actions_list2.append(action_)
        print (" Exploration action: ", action_)
            
    # data = actions_
    
    for k in range(len(actions_)):
        plt.hist(actions_[k], bins=20)
        plt.show()
    
    reward=np.array([-9.0])
    """
    print ("Norm Reward: " + str(norm_reward(reward, reward_bounds)) )
    print ("Norm Reward: " + str(norm_reward(reward+-0.5, reward_bounds)))
    
    print ("Valid bounds: " + str(validBounds([[ -6.34096220e-01,   9.42412945e-01,  -2.77025047e+00,
          5.99883344e-01,  -5.19683588e-01,  -5.19683588e-01,
         -3.42888109e-01,  -5.23556180e-01,   7.76794216e-02,
         -9.48665544e-02,  -1.25781827e+00,  -5.54537258e-01,
         -1.47478797e-02,   3.06775891e-01,  -5.49878858e-02,
         -3.10999480e-01,  -5.23225430e-01,   4.41216961e-02,
          6.70018120e-02,  -2.68502903e-01,  -1.07900884e-01,
         -3.31729491e-01,  -8.55080422e-01,  -4.32993609e-01,
         -1.05998050e-01,  -2.68106419e-01,  -1.07713842e-01,
         -2.50738648e-01,  -8.67029229e-01,  -4.42178656e-01,
         -2.98209530e-02,   6.47656790e-01,  -6.87410339e-02,
         -3.04862383e-01,  -5.17820447e-01,  -2.19130177e-02,
          1.87506175e-01,   2.88989499e-01,  -5.06674074e-02,
         -3.06572773e-01,  -5.28246094e-01,   9.33682034e-03,
         -2.12056797e-01,   2.88929341e-01,  -4.98415256e-02,
         -3.06224259e-01,  -5.28127163e-01,   1.84725992e-03,
         -5.74297924e-03,  -7.39800415e-01,  -3.28428126e-01,
         -2.37915139e-01,  -1.49736493e+00,  -8.38406722e-01,
         -1.22256339e-01,  -7.39480049e-01,  -3.29799656e-01,
         -2.03172964e-01,  -1.51351519e+00,  -8.77449749e-01,
          1.85604062e-01,  -1.00539563e-01,  -6.78502488e-02,
         -3.08624715e-01,  -5.32719181e-01,  -2.79023435e-01,
         -2.13983417e-01,  -1.00673412e-01,  -6.55732138e-02,
         -3.12425854e-01,  -5.33302855e-01,  -3.10067463e-01,
         -4.53427219e-02,  -1.00490585e+00,  -3.94977763e-01,
         -2.29839893e-01,  -1.30519213e+00,  -7.17686616e-01,
         -1.29773048e-01,  -1.00561552e+00,  -3.99412635e-01,
         -3.34232676e-01,  -1.31537288e+00,  -7.46435832e-01,
         -4.72069519e-02,  -1.02928355e+00,  -3.19346926e-01,
         -1.22069807e-01,  -4.60775992e-01,  -7.68291495e-01,
         -1.24429322e-01,  -1.02959452e+00,  -3.23858854e-01,
         -1.56321658e-01,  -4.63447258e-01,  -7.87323291e-01],
       [  5.52303145e-01,   1.04555761e+00,   3.02666000e+01,
          1.00002054e+00,   8.06735146e-01,   8.06735146e-01,
          3.51870125e-01,   9.91835992e-02,   1.46733007e+00,
          1.20645504e-01,   1.27079125e+00,   5.69349748e-01,
          1.47486517e-02,   4.01719900e-01,   3.42407443e-03,
          3.17251037e-01,   1.04200792e-01,   1.46485770e+00,
          1.07099820e-01,  -2.24500388e-01,   1.59571481e-01,
          2.57378948e-01,   1.38080353e-01,   1.54077880e+00,
         -6.78278735e-02,  -2.24941047e-01,   1.59509411e-01,
          3.43361385e-01,   1.41715459e-01,   1.53683745e+00,
          2.97725769e-02,   6.67872076e-01,   7.29232433e-02,
          3.09939696e-01,   1.23269819e-01,   1.36720212e+00,
          2.12060324e-01,   3.09593576e-01,   2.78002296e-02,
          3.12247037e-01,   1.11951187e-01,   1.48659232e+00,
         -1.87558614e-01,   3.09643560e-01,   2.65821725e-02,
          3.12895954e-01,   1.12183498e-01,   1.49365569e+00,
          1.25905392e-01,  -6.39025192e-01,   3.38600265e-01,
          1.98073892e-01,   4.55785665e-01,   1.35369655e+00,
          2.88702715e-03,  -6.39567527e-01,   3.43208581e-01,
          2.41450683e-01,   4.51482942e-01,   1.37950851e+00,
          2.13967233e-01,  -8.02279111e-02,   3.96722428e-02,
          3.18095279e-01,   1.06835040e-01,   1.63583511e+00,
         -1.85688622e-01,  -8.01561112e-02,   3.65994168e-02,
          3.14904357e-01,   1.08489792e-01,   1.66426497e+00,
          1.34669408e-01,  -9.01365767e-01,   3.74676050e-01,
          3.23646209e-01,   5.41831850e-01,   1.34261187e+00,
          4.15927916e-02,  -9.00870526e-01,   3.85834112e-01,
          2.30340241e-01,   5.35241615e-01,   1.38430484e+00,
          1.29370966e-01,  -9.27492425e-01,   4.43455684e-01,
          1.60730840e-01,   2.58810565e-01,   1.61350876e+00,
          4.36603207e-02,  -9.27366355e-01,   4.54681722e-01,
          1.24063643e-01,   2.61238771e-01,   1.64705196e+00]])))
    
    
    """
    
    print("std: ", std)
    actions_list = np.array(actions_list)
    actions_list2 = np.array(actions_list2)
    print ("Actions: ", actions_list)
    print ("Actions2: ", actions_list2)
    
    print ("Actions2 mean:", np.mean(actions_list2, axis=0))
    print ("Actions2 std:", np.std(actions_list2, axis=0))
    
    # print ("Actions2 norm mean:", np.mean(norm_action(actions_list2, action_bounds), axis=0))
    # print ("Actions2 norm std:", np.std(norm_action(actions_list2, action_bounds), axis=0))
    print ("Actions2 mean:", np.mean(scale_action(norm_action(actions_list2, action_bounds), action_bounds), axis=0))
    print ("Actions2 std:", np.std(scale_action(norm_action(actions_list2, action_bounds), action_bounds), axis=0))
    
    actions_list_2 = norm_action(actions_list, action_bounds)
    # actions_list_2= []
    # for i in range(num_samples):
    #     actions_list_2.append(norm_action(actions_list2[i], action_bounds))
        
    actions_list2_ = []
    for i in range(num_samples):
        action_ = randomExporationSTD(settings["exploration_rate"], scale_action(actions_list_2[i], action_bounds), std, action_bounds)
        # action_ = randomExporation(settings["exploration_rate"], actions_list[i], action_bounds)
        actions_list2_.append(action_)
        # print (" Exploration action: ", action_)
        
    # print ("Actions2 mean:", np.mean(scale_action(actions_list2_, action_bounds), axis=0))
    # print ("Actions2 std:", np.std(scale_action(actions_list2_, action_bounds), axis=0))
    print ("Actions2 mean:", np.mean(actions_list2_, axis=0))
    print ("Actions2 std:", np.std(actions_list2_, axis=0))
    
    # l = loglikelihood(actions_list2, actions_list, std, actions_list.shape[1])
    # l2 = loglikelihood(actions_list2+0.001, actions_list, std, actions_list.shape[1])
    # print ("Action std: ", np.repeat([std], 50, axis=0))
    # print ("Likelyhood: ", np.exp(l-l2))
    # print ("Likelyhood: ", (l))
    