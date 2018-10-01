import theano
from theano import tensor as T
import numpy as np
import matplotlib.pyplot as plt

def load_data2(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''
    import json
    dataFile = open(dataset)
    s = dataFile.read()
    data = json.loads(s)
    dataFile.close()
    tuples = data["Tuples"]
    states = []
    actions = []
    rewards = []
    nextStates = []
    # print tuples
    for tup in tuples:
        states.append(tup["InitialState:"]["Params"])
        actions.append(tup["Action:"]["ID"])
        rewards.append(tup["Reward:"])
        nextStates.append(tup["ResultState:"]["Params"])
        
    states = numpy.array(states)
    actions = numpy.array(actions)
    reward = numpy.array(rewards)
    nextStates = numpy.array(nextStates)

    """
    fixed_states = numpy.amax(numpy.abs(states), axis=0)
    fixed_states = numpy.divide(states, fixed_states)
    states = fixed_states    
    # fixed_actions = numpy.amax(numpy.abs(actions), axis=0)
    # actions = numpy.divide(actions, fixed_actions)
    fixed_reward = numpy.amax(numpy.abs(rewards), axis=0)
    reward = numpy.divide(reward, fixed_reward)
    fixed_nextStates = numpy.amax(numpy.abs(nextStates), axis=0)
    nextStates = numpy.divide(nextStates, fixed_nextStates)
    """
    # print states
    # print "Max state parameters " + str(fixed_states)
    # print "Actions: " + str(actions[:200])
    # print "Rewards: " + str(reward[:200])

    train_set = (states,actions,nextStates,rewards)
    
    train_set_states, train_set_actions, train_set_result_states, train_set_rewards = shared_dataset(train_set)
    
    rval = (train_set_states, train_set_actions, train_set_result_states, train_set_rewards)
    return rval

dataset = "controllerTuples.json"
states, actions, result_states, rewards = load_data2(dataset)


X = T.scalar()
Y = T.scalar()

def model(X, w):
    return (X * w) + b

w = theano.shared(np.asarray(0., dtype=theano.config.floatX))
b = theano.shared(np.asarray(0., dtype=theano.config.floatX))
y = model(X, w)

cost = T.mean(T.sqr(y - Y))
gradient = T.grad(cost=cost, wrt=w)
gradient_b = T.grad(cost=cost, wrt=b)
updates = [[w, w - gradient * 0.01],
           [b, b - gradient_b * 0.01]]

train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)

for i in range(100):
    for x, y in zip(trX, trY):
        train(x, y)
        
print w.get_value() #something around 2
w_ = w.get_value()

plt.scatter(trX,trY)
plt.plot([trX[0], trX[-1]], [trX[0]*w_, trX[-1]*w_])
plt.axis("tight")

plt.show()

