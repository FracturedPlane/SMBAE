import numpy as np
# import scipy
from scipy.spatial.distance import pdist, squareform
from theano import tensor as T
import theano
from theano import pp
theano.config.mode='FAST_COMPILE'

s = np.array([[ -0.15,  -0.37],
              [-0.1, -0.3],
              [ 0.0, -0.4],
              [-0.7,  0.3]])

s_1_dist = np.sqrt(np.sum((s[0]-s[1])**2))
print ("s_1_dist: ", s_1_dist)

print ("s: ", s )

a_ = np.array([[ 0.1,  0.2],
              [ 0.2,  0.7],
              [-0.3,  0.1],
              [ 0.0, -0.4]])

s_squared = s**2
print ("s^2 ", s_squared)

# s_dist = s_squared - np.transpose(s_squared)
dist = pdist(s, metric='euclidean')
dist = squareform(dist)
print ("Dist: ", dist)
print( "s sum axis 1: ", np.sum(s, axis=1))

# X is an m-by-n matrix (rows are examples, columns are dimensions)
# D is an m-by-m symmetric matrix of pairwise Euclidean distances
a = np.sum(s**2, axis=1)
D = np.sqrt((((a + np.reshape(a, (1,4)).T) - (2*np.dot(s, s.T)))))
print("a: ", a)
print( "Dist2: ", D)
print("newaxis: ", a[np.newaxis])
print("newaxis: ", a[np.newaxis].shape)

avgD = np.mean(D, axis=1)
print ("Avg distance: ", avgD)

State = T.matrix("State")
State.tag.test_value = np.random.rand(s.shape[0], s.shape[1])
Action = T.matrix("Action")
Action.tag.test_value = np.random.rand(a_.shape[0], a_.shape[1])

state_sum = T.mean(T.pow(State,2), axis=1)
Distance = ((state_sum + T.reshape(state_sum, (1,-1)).T) - (2*T.dot(State, State.T)))
action_sum = T.mean(T.pow(Action,2), axis=1)
Distance_action = ((((action_sum + T.reshape(action_sum, (1,-1)).T) - (2*T.dot(Action, Action.T)))))
weighted_dist = theano.tensor.elemwise.Elemwise(theano.scalar.mul)(Distance, Distance_action)
wieghted_mean_dist = -T.mean(T.mean(weighted_dist, axis=1))

gs = T.grad(wieghted_mean_dist, [State, Action])
# pp(gs)
get_mean_dist = theano.function([State, Action], weighted_dist)

get_state_weighted_dist = theano.function([State, Action], wieghted_mean_dist)
get_state_weighted_dist_grad = theano.function([State, Action], gs)

a_dist_ = get_mean_dist(s, a_)
print ("Theano a_dist: ", a_dist_)

w_dist_ = get_state_weighted_dist(s, a_)
print ("Theano w_dist: ", w_dist_)

w_dist_grad = get_state_weighted_dist_grad(s, a_)
print ("Theano w_dist_grad: ", w_dist_grad)

