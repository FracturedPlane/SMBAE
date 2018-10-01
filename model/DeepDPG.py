import theano
from theano import tensor as T
import numpy as np
import lasagne
import sys
sys.path.append('../')
from model.ModelUtil import *
from model.AgentInterface import AgentInterface


# For debugging
# theano.config.mode='FAST_COMPILE'
from collections import OrderedDict

def rmsprop(loss_or_grads, params, learning_rate=1.0, rho=0.9, epsilon=1e-6):
    """RMSProp updates
    Scale learning rates by dividing with the moving average of the root mean
    squared (RMS) gradients. See [1]_ for further description.
    Parameters
    ----------
    loss_or_grads : symbolic expression or list of expressions
        A scalar loss expression, or a list of gradient expressions
    params : list of shared variables
        The variables to generate update expressions for
    learning_rate : float or symbolic scalar
        The learning rate controlling the size of update steps
    rho : float or symbolic scalar
        Gradient moving average decay factor
    epsilon : float or symbolic scalar
        Small value added for numerical stability
    Returns
    -------
    OrderedDict
        A dictionary mapping each parameter to its update expression
    Notes
    -----
    `rho` should be between 0 and 1. A value of `rho` close to 1 will decay the
    moving average slowly and a value close to 0 will decay the moving average
    fast.
    Using the step size :math:`\\eta` and a decay factor :math:`\\rho` the
    learning rate :math:`\\eta_t` is calculated as:
    .. math::
       r_t &= \\rho r_{t-1} + (1-\\rho)*g^2\\\\
       \\eta_t &= \\frac{\\eta}{\\sqrt{r_t + \\epsilon}}
    References
    ----------
    .. [1] Tieleman, T. and Hinton, G. (2012):
           Neural Networks for Machine Learning, Lecture 6.5 - rmsprop.
           Coursera. http://www.youtube.com/watch?v=O3sxAc4hxZU (formula @5:20)
    """
    clip = 2.0
    grads = lasagne.updates.get_or_compute_grads(loss_or_grads, params)
    # grads = theano.gradient.grad_clip(grads, -clip, clip) 
    grads_ = []
    for grad in grads:
        grads_.append(theano.gradient.grad_clip(grad, -clip, clip) )
    grads = grads_
    
    print ("Grad Update: " + str(grads[0]) )
    
    updates = OrderedDict()

    # Using theano constant to prevent upcasting of float32
    one = T.constant(1)

    for param, grad in zip(params, grads):
        value = param.get_value(borrow=True)
        accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                             broadcastable=param.broadcastable)
        accu_new = rho * accu + (one - rho) * grad ** 2
        updates[accu] = accu_new
        updates[param] = param - (learning_rate * grad /
                                  T.sqrt(accu_new + epsilon))

    return updates

class DeepDPG(AgentInterface):
    
    def __init__(self, n_in, n_out, state_bounds, action_bounds, reward_bound, settings_):
        """
            In order to get this to work we need to be careful not to update the actor parameters
            when updating the critic. This can be an issue when the Concatenating networks together.
            The first first network becomes a part of the second. However you can still access the first
            network by itself but an updates on the second network will effect the first network.
            Care needs to be taken to make sure only the parameters of the second network are updated.
        """
        
        super(DeepDPG,self).__init__(n_in, n_out, state_bounds, action_bounds, reward_bound, settings_)

        batch_size=self.getSettings()['batch_size']
        # data types for model
        State = T.dmatrix("State")
        State.tag.test_value = np.random.rand(batch_size,self._state_length)
        ResultState = T.dmatrix("ResultState")
        ResultState.tag.test_value = np.random.rand(batch_size,self._state_length)
        Reward = T.col("Reward")
        Reward.tag.test_value = np.random.rand(batch_size,1)
        Action = T.dmatrix("Action")
        Action.tag.test_value = np.random.rand(batch_size, self._action_length)
        # create a small convolutional neural network
        inputLayerActA = lasagne.layers.InputLayer((None, self._state_length), State)
        
        l_hid1ActA = lasagne.layers.DenseLayer(
                inputLayerActA, num_units=128,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        
        l_hid2ActA = lasagne.layers.DenseLayer(
                l_hid1ActA, num_units=64,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        
        l_hid3ActA = lasagne.layers.DenseLayer(
                l_hid2ActA, num_units=32,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
    
        self._l_outActA = lasagne.layers.DenseLayer(
                l_hid3ActA, num_units=self._action_length,
                nonlinearity=lasagne.nonlinearities.linear)
        
        inputLayerA = lasagne.layers.InputLayer((None, self._state_length), State)

        concatLayer = lasagne.layers.ConcatLayer([inputLayerA, self._l_outActA])
        
        l_hid1A = lasagne.layers.DenseLayer(
                concatLayer, num_units=128,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        
        l_hid2A = lasagne.layers.DenseLayer(
                l_hid1A, num_units=64,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        
        l_hid3A = lasagne.layers.DenseLayer(
                l_hid2A, num_units=32,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
    
        self._l_outA = lasagne.layers.DenseLayer(
                l_hid3A, num_units=1,
                nonlinearity=lasagne.nonlinearities.linear)
        # self._b_o = init_b_weights((self._action_length,))

        # self.updateTargetModel()
        inputLayerActB = lasagne.layers.InputLayer((None, self._state_length), State)
        
        l_hid1ActB = lasagne.layers.DenseLayer(
                inputLayerActB, num_units=128,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        
        l_hid2ActB = lasagne.layers.DenseLayer(
                l_hid1ActB, num_units=64,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
    
        l_hid3ActB = lasagne.layers.DenseLayer(
                l_hid2ActB, num_units=32,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        
        self._l_outActB = lasagne.layers.DenseLayer(
                l_hid3ActB, num_units=self._action_length,
                nonlinearity=lasagne.nonlinearities.linear)

        inputLayerB = lasagne.layers.InputLayer((None, self._state_length), State)
        concatLayerB = lasagne.layers.ConcatLayer([inputLayerB, self._l_outActB])
        
        l_hid1B = lasagne.layers.DenseLayer(
                concatLayerB, num_units=128,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        
        l_hid2B = lasagne.layers.DenseLayer(
                l_hid1B, num_units=64,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
        
        l_hid3B = lasagne.layers.DenseLayer(
                l_hid2B, num_units=32,
                nonlinearity=lasagne.nonlinearities.leaky_rectify)
    
        self._l_outB = lasagne.layers.DenseLayer(
                l_hid3B, num_units=1,
                nonlinearity=lasagne.nonlinearities.linear)
            
        # print ("Initial W " + str(self._w_o.get_value()) )
        
        self._learning_rate = self.getSettings()['learning_rate']
        self._discount_factor= self.getSettings()['discount_factor']
        self._rho = self.getSettings()['rho']
        self._rms_epsilon = self.getSettings()['rms_epsilon']
        
        self._weight_update_steps=self.getSettings()['steps_until_target_network_update']
        self._updates=0
        self._decay_weight=self.getSettings()['regularization_weight']
        
        self._states_shared = theano.shared(
            np.zeros((batch_size, self._state_length),
                     dtype=theano.config.floatX))

        self._next_states_shared = theano.shared(
            np.zeros((batch_size, self._state_length),
                     dtype=theano.config.floatX))

        self._rewards_shared = theano.shared(
            np.zeros((batch_size, 1), dtype=theano.config.floatX),
            broadcastable=(False, True))

        self._actions_shared = theano.shared(
            np.zeros((batch_size, self._action_length), dtype=theano.config.floatX),
            )
        
        self._q_valsActA = lasagne.layers.get_output(self._l_outActA, State)
        self._q_valsActB = lasagne.layers.get_output(self._l_outActB, ResultState)
        self._q_valsActB2 = lasagne.layers.get_output(self._l_outActB, State)
        inputs_ = {
            State: self._states_shared,
            Action: self._q_valsActA,
        }
        self._q_valsA = lasagne.layers.get_output(self._l_outA, inputs_)
        inputs_ = {
            ResultState: self._next_states_shared,
            Action: self._q_valsActB,
        }
        self._q_valsB = lasagne.layers.get_output(self._l_outB, inputs_)
        
        
        self._q_func = self._q_valsA
        self._q_funcAct = self._q_valsActA
        self._q_funcB = self._q_valsB
        self._q_funcActB = self._q_valsActB
        
        # self._q_funcAct = theano.function(inputs=[State], outputs=self._q_valsActA, allow_input_downcast=True)
        
        self._target = (Reward + self._discount_factor * self._q_valsB)
        self._diff = self._target - self._q_valsA
        self._loss = 0.5 * self._diff ** 2 + (self._decay_weight * lasagne.regularization.regularize_network_params(
                self._l_outA, lasagne.regularization.l2))
        self._loss = T.mean(self._loss)
    
        # assert len(lasagne.layers.helper.get_all_params(self._l_outA)) == 16
        # Need to remove the action layers from these params
        self._params = lasagne.layers.helper.get_all_params(self._l_outA)[-len(lasagne.layers.helper.get_all_params(self._l_outActA)):] 
        print ("******Number of Layers is: " + str(len(lasagne.layers.helper.get_all_params(self._l_outA))))
        print ("******Number of Action Layers is: " + str(len(lasagne.layers.helper.get_all_params(self._l_outActA))))
        self._actionParams = lasagne.layers.helper.get_all_params(self._l_outActA)
        self._givens_ = {
            State: self._states_shared,
            # ResultState: self._next_states_shared,
            Reward: self._rewards_shared,
            # Action: self._actions_shared,
        }
        self._actGivens = {
            State: self._states_shared,
            # ResultState: self._next_states_shared,
            # Reward: self._rewards_shared,
            # Action: self._actions_shared,
        }
        
        # SGD update
        #updates_ = rmsprop(loss, params, self._learning_rate, self._rho,
        #                                    self._rms_epsilon)
        # TD update
        # minimize Value function error
        self._updates_ = rmsprop(T.mean(self._q_func) + (self._decay_weight * lasagne.regularization.regularize_network_params(
        self._l_outA, lasagne.regularization.l2)), self._params, 
                    self._learning_rate * -T.mean(self._diff), self._rho, self._rms_epsilon)
        
        
        # actDiff1 = (Action - self._q_valsActB) #TODO is this correct?
        # actDiff = (actDiff1 - (Action - self._q_valsActA))
        # actDiff = ((Action - self._q_valsActB2)) # Target network does not work well here?
        #self._actDiff = ((Action - self._q_valsActA)) # Target network does not work well here?
        #self._actLoss = 0.5 * self._actDiff ** 2 + (1e-4 * lasagne.regularization.regularize_network_params( self._l_outActA, lasagne.regularization.l2))
        #self._actLoss = T.mean(self._actLoss)
        
        # actionUpdates = rmsprop(actLoss + 
        #    (1e-4 * lasagne.regularization.regularize_network_params(
        #        self._l_outActA, lasagne.regularization.l2)), actionParams, 
        #            self._learning_rate * 0.01 * (-actLoss), self._rho, self._rms_epsilon)
        
        # Maximize wrt q function
        
        # theano.gradient.grad_clip(x, lower_bound, upper_bound) # // TODO
        self._actionUpdates = rmsprop(T.mean(self._q_func) + 
          (self._decay_weight * lasagne.regularization.regularize_network_params(
              self._l_outActA, lasagne.regularization.l2)), self._actionParams, 
                  self._learning_rate * 0.1, self._rho, self._rms_epsilon)
        
        
        
        self._train = theano.function([], [self._loss, self._q_func], updates=self._updates_, givens=self._givens_)
        # self._trainActor = theano.function([], [actLoss, self._q_valsActA], updates=actionUpdates, givens=actGivens)
        self._trainActor = theano.function([], [self._q_func], updates=self._actionUpdates, givens=self._actGivens)
        self._q_val = theano.function([], self._q_valsA,
                                       givens={State: self._states_shared})
        self._q_action = theano.function([], self._q_valsActA,
                                       givens={State: self._states_shared})
        inputs_ = [
                   State, 
                   Reward, 
                   # ResultState
                   ]
        self._bellman_error = theano.function(inputs=inputs_, outputs=self._diff, allow_input_downcast=True)
        # self._diffs = theano.function(input=[State])
        
    def _trainOneActions(self, states, actions, rewards, result_states):
        print ("Training action")
        # lossActor, _ = self._trainActor()
        State = T.dmatrix("State")
        # State.tag.test_value = np.random.rand(batch_size,self._state_length)
        #ResultState = T.dmatrix("ResultState")
        #ResultState.tag.test_value = np.random.rand(batch_size,self._state_length)
        #Reward = T.col("Reward")
        #Reward.tag.test_value = np.random.rand(batch_size,1)
        Action = T.dmatrix("Action")
        #Action.tag.test_value = np.random.rand(batch_size, self._self._action_length)
        
        
        for state, action, reward, result_state in zip(states, actions, rewards, result_states):
            # print (state)
            # print (action)
            self._states_shared.set_value([state])
            self._next_states_shared.set_value([result_state])
            self._actions_shared.set_value([action])
            self._rewards_shared.set_value([reward])
            # print ("Q value for state and action: " + str(self.q_value([state])))
            # all_paramsA = lasagne.layers.helper.get_all_param_values(self._l_outA)
            # print ("Network length: " + str(len(all_paramsA)))
            # print ("weights: " + str(all_paramsA[0]))
            # lossActor, _ = self._trainActor()
            _params = lasagne.layers.helper.get_all_params(self._l_outA)
            # print (_params[0].get_value())
            inputs_ = {
                State: self._states_shared,
                Action: self._q_valsActA,
            }
            self._q_valsA = lasagne.layers.get_output(self._l_outA, inputs_)
            
            
            updates_ = rmsprop(T.mean(self._q_valsA) + (1e-6 * lasagne.regularization.regularize_network_params(
                self._l_outA, lasagne.regularization.l2)), _params, 
                self._learning_rate * -T.mean(self._diff), self._rho, self._rms_epsilon)
            
            ind = 0
            print ("Update: " + str (updates_.items()))
            print ("Updates length: " + str (len(updates_.items()[ind][0].get_value())) )
            print (" Updates: " + str(updates_.items()[ind][0].get_value()))
            
            
    def updateTargetModel(self):
        # print ("Updating target Model")
        """
            Target model updates
        """
        
        all_paramsA = lasagne.layers.helper.get_all_param_values(self._l_outA)
        all_paramsB = lasagne.layers.helper.get_all_param_values(self._l_outB)
        lerp_weight = 0.001
        # vals = lasagne.layers.helper.get_all_param_values(self._l_outActA)
        
        # print ("l_out length: " + str(len(all_paramsA)))
        # print ("l_out length: " + str(all_paramsA[-6:]))
        # print ("l_out[0] length: " + str(all_paramsA[0]))
        # print ("l_out[4] length: " + str(all_paramsA[4]))
        # print ("l_out[5] length: " + str(all_paramsA[5]))
        # print ("l_out[6] length: " + str(all_paramsA[6]))
        # print ("l_out[7] length: " + str(all_paramsA[7]))
        # print ("l_out[11] length: " + str(all_paramsA[11]))
        # print ("param Values")
        all_params = []
        for paramsA, paramsB in zip(all_paramsA, all_paramsB):
            # print ("paramsA: " + str(paramsA))
            # print ("paramsB: " + str(paramsB))
            params = (lerp_weight * paramsA) + ((1.0 - lerp_weight) * paramsB)
            all_params.append(params)
        """
        all_paramsActA = lasagne.layers.helper.get_all_param_values(self._l_outActA)
        all_paramsActB = lasagne.layers.helper.get_all_param_values(self._l_outActB)
        # print ("l_outAct[0] length: " + str(all_paramsActA[0]))
        # print ("l_outAct[4] length: " + str(all_paramsActA[4]))
        # print ("l_outAct[5] length: " + str(all_paramsActA[5]))
        all_paramsAct = []
        for paramsA, paramsB in zip(all_paramsActA, all_paramsActB):
            # print ("paramsA: " + str(paramsA))
            # print ("paramsB: " + str(paramsB))
            params = (lerp_weight * paramsA) + ((1.0 - lerp_weight) * paramsB)
            all_paramsAct.append(params)
            """
        lasagne.layers.helper.set_all_param_values(self._l_outB, all_params)
        # lasagne.layers.helper.set_all_param_values(self._l_outActB, all_paramsAct) 
        
    def getNetworkParameters(self):
        params = []
        params.append(lasagne.layers.helper.get_all_param_values(self._l_outA))
        params.append(lasagne.layers.helper.get_all_param_values(self._l_outActA))
        params.append(lasagne.layers.helper.get_all_param_values(self._l_outB))
        params.append(lasagne.layers.helper.get_all_param_values(self._l_outActB))
        return params
        
    def setNetworkParameters(self, params):
        lasagne.layers.helper.set_all_param_values(self._l_outA, params[0])
        lasagne.layers.helper.set_all_param_values(self._l_outActA, params[1])
        lasagne.layers.helper.set_all_param_values(self._l_outB, params[2])
        lasagne.layers.helper.set_all_param_values(self._l_outActB, params[3])
        
    def trainCritic(self, states, actions, rewards, result_states):
        self._states_shared.set_value(states)
        self._next_states_shared.set_value(result_states)
        self._actions_shared.set_value(actions)
        self._rewards_shared.set_value(rewards)
        
        
        if (( self._updates % self._weight_update_steps) == 0):
            self.updateTargetModel()
        self._updates += 1
        loss, _ = self._train()
        return loss
        
    def trainActor(self, states, actions, rewards, result_states):
        self._states_shared.set_value(states)
        self._next_states_shared.set_value(result_states)
        self._actions_shared.set_value(actions)
        self._rewards_shared.set_value(rewards)
        
        loss = self._trainActor()
        return loss
        
    def train(self, states, actions, rewards, result_states):
        loss = self.trainCritic(states, actions, rewards, result_states)
        lossActor = self.trainActor(states, actions, rewards, result_states)
        return loss
    
    def predict(self, state):
        # states = np.zeros((self._batch_size, self._state_length), dtype=theano.config.floatX)
        # states[0, ...] = state
        state = [norm_state(state, self._state_bounds)]
        self._states_shared.set_value(state)
        # q_valsActA = lasagne.layers.get_output(self._l_outActA, state).eval()
        action_ = scale_action(self._q_action()[0], self._action_bounds)
        # action_ = q_valsActA[0]
        return action_
    def q_value(self, state):
        # states = np.zeros((self._batch_size, self._state_length), dtype=theano.config.floatX)
        # states[0, ...] = state
        state = [norm_state(state, self._state_bounds)]
        self._states_shared.set_value(state)
        return self._q_val()[0]
    def bellman_error(self, state, action, reward, result_state):
        # return self._bellman_error(state, reward, result_state)
        return self._bellman_error(state, reward)
