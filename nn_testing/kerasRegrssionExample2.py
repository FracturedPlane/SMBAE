'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import numpy as np
import math
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from model.ModelUtil import *
from util.ExperienceMemory import ExperienceMemory
import itertools
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers.core import Dense, Dropout, Activation
import random

np.random.seed(1337)  # for reproducibility

def f(x):
    return (math.cos(x)-0.75)*(math.sin(x)+0.75)

state_bounds = np.array([[0.0],[5.0]])
action_bounds = np.array([[-4.0],[2.0]])
reward_bounds = np.array([[-3.0],[1.0]])
experience_length = 200
batch_size=32
# states = np.repeat(np.linspace(0.0, 5.0, experience_length),2, axis=0)
states = np.linspace(0.0, 5.0, experience_length)
# shuffle = range(experience_length)
# states = states[shuffle]
# random.shuffle(shuffle)
# print ("States: " , states)
old_states = states
# norm_states = np.array(map(norm_state, states, itertools.repeat(state_bounds, len(states))))
# X_test = X_train = norm_states
# states = np.linspace(-5.0,-2.0, experience_length/2)
# states = np.append(states, np.linspace(-1.0, 5.0, experience_length/2))
# print states
# actions = np.array(map(fNoise, states))
actions = np.transpose(np.array([map(f, states)]))
# actions = actions[shuffle]
# norm_actions = np.array(map(norm_action, actions, itertools.repeat(action_bounds, len(actions))))
# y_train = y_test = norm_actions
# settings = {}

experience = ExperienceMemory(len(state_bounds[0]), len(action_bounds[0]), experience_length, continuous_actions=True)
experience.setStateBounds(state_bounds)
experience.setRewardBounds(reward_bounds)
experience.setActionBounds(action_bounds)
for i in range(experience_length):
    action_ = np.array([actions[i]])
    state_ = np.array([states[i]])
    # print "Action: " + str([actions[i]])
    # tmp_action = f(states[i])
    # print ("Ation diff: " , tmp_action, action_)
    experience.insert(state_, action_, state_, np.array([0]))
    # tmp_state_ = scale_state(experience._state_history[experience.samples()-1], state_bounds)
    # tmp_action_ = scale_action(experience._action_history[experience.samples()-1], action_bounds)
    # tmp_action = f(tmp_state_)
    # print ("State diff: " , tmp_state_, states[i])
    # print ("Ation diff2: " , tmp_action, action_)


model = Sequential()
# 2 inputs, 10 neurons in 1 hidden layer, with tanh activation and dropout
model.add(Dense(128, init='uniform', input_shape=(1,))) 
model.add(Activation('relu'))
model.add(Dense(64, init='uniform')) 
model.add(Activation('relu'))
# 1 output, linear activation
model.add(Dense(1, init='uniform'))
model.add(Activation('linear'))

sgd = SGD(lr=0.01, momentum=0.9)
print ("Clipping: ", sgd.decay)
model.compile(loss='mse', optimizer=sgd)

from keras.callbacks import EarlyStopping
# early_stopping = EarlyStopping(monitor='val_loss', patience=2)

errors=[]
for i in range(5000):
    _states, _actions, _result_states, _rewards, fals_ = experience.get_batch(batch_size)
    # scale_states = np.array(map(scale_state, _states, itertools.repeat(state_bounds, len(_states))))
    # tmp_actions = np.transpose(np.array([map(f, scale_states)]))
    # norm_actions = np.array(map(norm_action, tmp_actions, itertools.repeat(action_bounds, len(tmp_actions))))
    # print ("mini batch actions: " , tmp_actions, _actions)
    # print ("y diff: " ,  _actions - norm_actions) 
    # error = model.train(_states, _actions)
    # errors.append(error)
    # print "Error: " + str(error)
    score = model.fit(_states, _actions,
              nb_epoch=1, batch_size=32,
              validation_data=(_states, _actions)
              # callbacks=[early_stopping],
              )

    errors.extend(score.history['loss'])

# print ("Score: " , errors)


states = (np.linspace(-1.0, 6.0, experience_length))
norm_states = np.array(map(norm_state, states, itertools.repeat(state_bounds, len(states))))
# norm_states = ((states  - 2.5)/3.5)
# predicted_actions = np.array(map(model.predict, states))

# x=np.transpose(np.array([states]))

# print ("States: " , x)
norm_predicted_actions = model.predict(norm_states, batch_size=200, verbose=0)
predicted_actions = np.array(map(scale_action, norm_predicted_actions, itertools.repeat(action_bounds, len(norm_predicted_actions))))
# print ("Prediction: ", predicted_actions)

# print "var : " + str(predicted_actions_var)
# print "act : " + str(predicted_actions)


std = 1.0
# _fig, (_bellman_error_ax, _reward_ax, _discount_error_ax) = plt.subplots(1, 1, sharey=False, sharex=True)
_fig, (_bellman_error_ax, _training_error_ax) = plt.subplots(1, 2, sharey=False, sharex=False)
_bellman_error, = _bellman_error_ax.plot(old_states, actions, linewidth=2.0, color='y', label="True function")
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
legend = _bellman_error_ax.legend(loc='lower right', shadow=True)
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

# _fig.show()
# er.show()
plt.show()

# print('Test score:', score[0])
# print('Test accuracy:', score[1])