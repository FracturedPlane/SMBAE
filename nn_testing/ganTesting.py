import numpy as np
# import lasagne
import sys
sys.path.append('../')
import matplotlib.pyplot as plt
import math
import random
import json



def _computeHeight(self, action_):
    """
        action_ y-velocity
    """
    init_v_squared = (action_*action_)
    # seconds_ = 2 * (-self._box.G)
    return (-init_v_squared)/1.0  

def _computeTime(self, velocity_y):
    """
    
    """
    seconds_ = velocity_y/-self._gravity
    return seconds_

def OUNoise(theta, sigma, x_t, dt):
    """
        Ornstein–Uhlenbeck process
    
        d x t = θ ( μ − x t ) d t + σ d W t {\displaystyle dx_{t}=\theta (\mu -x_{t})\,dt+\sigma \,dW_{t}} {\displaystyle dx_{t}=\theta (\mu -x_{t})\,dt+\sigma \,dW_{t}}  
    """
    
    dWt = np.random.normal(0.0,0.3)
    dx_t = theta *(0.0 - x_t)* dt + (sigma * dWt)
    return dx_t
    

def integrate(dt,pos,vel, gravity = -9.8):
    """
        Perform simple Euler integration
        assume G = -9.8
        return pos, new_vel
    """ 
    
    new_pos = pos + ( vel * dt )
    new_vel =  vel + (gravity * dt)
    return (new_pos, new_vel)
    
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
    from algorithm.ForwardDynamics import ForwardDynamics
    from util.ExperienceMemory import ExperienceMemory
    from util.SimulationUtil import createForwardDynamicsModel, createRLAgent
    discrete_actions = np.array(settings['discrete_actions'])
    
    trajectory_length = 20
        
    state_bounds = np.array([[0.0],[20.0]])
    action_bounds = np.array([[-10.0],[10.0]])
    result_state_bounds = np.array([[-50.0]*trajectory_length,[50.0]*trajectory_length])
    reward_bounds = np.array([[0.0],[1.0]])
    experience_length = 500
    num_samples = 10
    batch_size=64
    # states = np.repeat([np.linspace(-5.0, 5.0, experience_length)],2, axis=0)
    velocities = np.linspace(1.0, 15.0, experience_length)
    actions = []
    next_states_ = []
    states_ = []
    dt = 0.1
    for v_ in velocities:
        accel = np.random.normal(0,2.5)
        for samples in range(num_samples):
            traj = []
            pos = 0
            vel_ = v_
            states_.append([vel_])
            # print("accel: ", accel)
            actions.append([accel])
            noise = 0.0
            for t_ in range(trajectory_length):
                (pos, vel_) = integrate(dt, pos, vel_, gravity=accel)
                ### This makes it really hard for the generator
                noise = noise + OUNoise(0.15, 0.3, noise, dt)
                pos = pos + noise
                traj.append(pos)
                
            next_states_.append(traj)
        # print ("traj length: ", len(traj))
            

    print("trag mean", np.mean(next_states_, axis=0))
    print("trag std", np.std(next_states_, axis=0))
    # print states
    # states2 = np.transpose(np.repeat([states], 2, axis=0))
    # print states2
    # model = createRLAgent(settings['agent_name'], state_bounds, discrete_actions, reward_bounds, settings)
    model = createForwardDynamicsModel(settings, state_bounds, action_bounds, None, None, None)
    
    experience = ExperienceMemory(len(state_bounds[0]), len(action_bounds[0]), experience_length*num_samples, continuous_actions=True, settings=settings, result_state_length=trajectory_length)
    experience.setStateBounds(state_bounds)
    experience.setRewardBounds(reward_bounds)
    experience.setActionBounds(action_bounds)
    experience.setSettings(settings)

    model.setStateBounds(state_bounds)
    model.setRewardBounds(reward_bounds)
    model.setActionBounds(action_bounds)
    
    arr = list(range(len(states_)))
    random.shuffle(arr)
    given_actions=[]
    given_states=[]
    for i in range(len(states_)):
        a = actions[arr[i]]
        action_ = np.array([a])
        given_actions.append(action_)
        state_ = np.array([states_[arr[i]]])
        next_state_ = np.array([next_states_[arr[i]]])
        given_states.append(state_)
        # print "Action: " + str([actions[i]])
        experience.insert(state_, action_, next_state_, np.array([1]))
        # print ("Added tuple: ", i)
    
    errors=[]
    for i in range(settings['rounds']):
        # print ("Actions: ", _actions)
        # print ("States: ", _states) 
        # (error, lossActor) = model.train(_states, _actions, _result_states, _rewards)
        for j in range(1):
            _states, _actions, _result_states, _rewards, falls_, advantage, exp_actions__ = experience.get_batch(batch_size)
            error = model.trainCritic(_states, _actions, _result_states, _rewards)
        for j in range(5):
            _states, _actions, _result_states, _rewards, falls_, advantage, exp_actions__ = experience.get_batch(batch_size)
            lossActor = model.trainActor(_states, _actions, _result_states, _rewards)
        errors.append(error)
        if (i % 100 == 0):
            print ("Iteration: ", i)
            print ("discriminator loss: ", error, " generator loss: ", lossActor)
        # print "Error: " + str(error)
    
    
    # states = np.linspace(-5.0, 5.0, experience_length)
    test_index = 400
    states_ = np.array(states_)
    print(states_[test_index])
    
    
    gen_state = model.predict([states_[test_index]], [actions[test_index]])
    _fig, (_bellman_error_ax) = plt.subplots(1, 1, sharey=False, sharex=True)
    for j in range(3):
        test_index = int(states_.shape[0]/5) * j
        print ("test_index: ",  test_index)
        discriminator_value = model.q_value([states_[test_index]], [actions[test_index]], next_states_[test_index])
        _bellman_error, = _bellman_error_ax.plot(range(len(gen_state)), next_states_[test_index], linewidth=3.0, label="True function: " + str(discriminator_value), linestyle='-', marker='o')
        for i in range(5):
            gen_state = model.predict([states_[test_index]], [actions[test_index]])
            discriminator_value = model.q_value([states_[test_index]], [actions[test_index]], gen_state)
            _bellman_error, = _bellman_error_ax.plot(range(len(gen_state)), gen_state, linewidth=2.0, label="Estimated function, " + str(discriminator_value), linestyle='--')
    # Now add the legend with some customizations.
    legend = _bellman_error_ax.legend(loc='lower right', shadow=True, ncol=1, fancybox=True)
    legend.get_frame().set_alpha(0.25)
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
    
    _fig.set_size_inches(11.0, 6.0, forward=True)
    plt.show()
    fileName="gantesting"
    _fig.savefig(fileName+".svg")
    _fig.savefig(fileName+".png")
