"""
This tutorial introduces logistic regression using Theano and stochastic
gradient descent.

Logistic regression is a probabilistic, linear classifier. It is parametrized
by a weight matrix :math:`W` and a bias vector :math:`b`. Classification is
done by projecting data points onto a set of hyperplanes, the distance to
which is used to determine a class membership probability.

Mathematically, this can be written as:

.. math::
  P(Y=i|x, W,b) &= softmax_i(W x + b) \\
                &= \frac {e^{W_i x + b_i}} {\sum_j e^{W_j x + b_j}}


The output of the model or prediction is then done by taking the argmax of
the vector whose i'th element is P(Y=i|x).

.. math::

  y_{pred} = argmax_i P(Y=i|x,W,b)


This tutorial presents a stochastic gradient descent optimization method
suitable for large datasets, and a conjugate gradient optimization method
that is suitable for smaller datasets.


References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 4.3.2

"""
__docformat__ = 'restructedtext en'

import cPickle
import gzip
import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T


class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(value=numpy.zeros((n_in, n_out),
                                                 dtype=theano.config.floatX),
                                name='W', borrow=True)
        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(value=numpy.zeros((n_out,),
                                                 dtype=theano.config.floatX),
                               name='b', borrow=True)

        # compute vector of class-membership probabilities in symbolic form
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.q_y_given_x = (T.dot(input, self.W) + self.b)
        
        # compute vector of class-membership probabilities in symbolic form
        self.p_value_given_resultState = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.q_value_given_resultState = (T.dot(input, self.W) + self.b)

        # compute prediction as class whose probability is maximal in
        # symbolic form
        self.y_pred = T.argmax(self.q_y_given_x, axis=1)
        
        self.p_y_pred = T.max(self.q_y_given_x, axis=1)

        # parameters of the model
        # keep track of model input
        self.input = input

        self.params = [self.W, self.b]
        self._discount = 0.8
        
        q_vals_ = theano.function([input], outputs=[(T.dot(input, self.W) + self.b)],
                                  )
        
        self._q_vals = theano.function([input], outputs=[q_vals_],
                        )

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|} \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
                \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
    
    def q_val(self, state):
        self._q_vals()[0]
        
    def q_value(self, state):
        # compile a predictor function
        q_given_state = (T.dot(state, self.W) + self.b)
        return (q_given_state)
    def q_value2(self, state):
        # compile a predictor function
        q_given_state = (T.dot(state, self.W) + self.b)
        return T.max(q_given_state)
    
    def max_q_value(self, state):
        
        # compile a predictor function
        q_given_state = (T.dot(state, self.W) + self.b)
        
        return T.max(q_given_state, axis=1, keepdims=True)
    
    def q_value_action(self, state):
        
        # compile a predictor function
        q_action_given_state = (T.dot(state, self.W) + self.b)
        
        return T.argmax(q_action_given_state)    
        
    def delta(self, state, reward, resultState):
        """
            computes the delta for the given state, reward, resultState
        """
        target = (reward +
          # (T.ones_like(terminals) - terminals) *
          # self._discount * self.p_value_given_resultState[T.arange(action.shape[0]), action])
          (self._discount * self.max_q_value(resultState)) )
        # diff = target - self.q_value_given_resultState[T.arange(action.shape[0]), action]
        # diff = target - self.q_value(state)[T.arange(batch_size), action.reshape((-1,))].reshape((-1, 1))
        diff = target - self.max_q_value(state)
        return diff.reshape((-1,1))
        
        
    def bellman_loss(self, state, reward, resultState):
        """This will compute the bellman error for the model.
           This does not nessicarily optimize the expected value but seems to work well. 
           
           bellman\_error = (reward + (discount*v(s_{t+1}))) - v(s)
        """
        """T.max
        states = T.tensor4('states')
        next_states = T.tensor4('next_states')
        rewards = T.col('rewards')
        actions = T.icol('actions')
        terminals = T.icol('terminals')
        """
        # ar = resultState + action + state
        # print resultState.get_value()
        batch_size=600
        target = (reward +
                  # (T.ones_like(terminals) - terminals) *
                  # self._discount * self.p_value_given_resultState[T.arange(action.shape[0]), action])
                  (self._discount * self.max_q_value(resultState)) )
        # diff = target - self.q_value_given_resultState[T.arange(action.shape[0]), action]
        # diff = target - self.q_value(state)[T.arange(batch_size), action.reshape((-1,))].reshape((-1, 1))
        diff = target - self.max_q_value(state)
        # diff = target - self.q_value(state)[T.arange(action.shape[0]), action]
        loss = 0.5 * diff ** 2
        # loss = diff
        loss = T.mean(loss)
        return loss               
        # print y
        # return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        
    def bellman_error(self, state, reward, resultState):
        target = (reward +
                  # (T.ones_like(terminals) - terminals) *
                  # self._discount * self.p_value_given_resultState[T.arange(action.shape[0]), action])
                  self._discount * self.max_q_value(resultState))
        diff = target - self.max_q_value(state)
        
        return T.mean(T.abs_(diff))

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', target.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(os.path.split(__file__)[0], "..", "data", dataset)
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)

    print '... loading data'

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.
    
    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

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
    valid_set = (states,actions,nextStates,rewards)
    test_set = (states,actions,nextStates,rewards)
    
    train_set_states, train_set_actions, train_set_result_states, train_set_rewards = shared_dataset(train_set)
    valid_set_states, valid_set_actions, valid_set_result_states, valid_set_rewards = shared_dataset(valid_set)
    test_set_states, test_set_actions, test_set_result_states, test_set_rewards = shared_dataset(test_set)
    
    rval = [(train_set_states, train_set_actions, train_set_result_states, train_set_rewards), 
            (valid_set_states, valid_set_actions, valid_set_result_states, valid_set_rewards),
            (test_set_states, test_set_actions, test_set_result_states, test_set_rewards)]
    return rval

def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    states,actions,nextStates,rewards = data_xy
    shared_states = theano.shared(numpy.asarray(states,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    shared_actions = theano.shared(numpy.asarray(actions,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    shared_result_states = theano.shared(numpy.asarray(nextStates,
                                       dtype=theano.config.floatX),
                         borrow=borrow)
    shared_rewards = theano.shared(numpy.asarray(rewards,
                                   dtype=theano.config.floatX),
                     borrow=borrow)
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return shared_states, T.cast(shared_actions, 'int64'), shared_result_states, shared_rewards



def sgd_optimization_mnist(learning_rate=0.03, n_epochs=1000,
                           dataset='mnist.pkl.gz',
                           batch_size=60):
    """
    Demonstrate stochastic gradient descent optimization of a log-linear
    model

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz

    """
    dataset = "controllerTuples.json"
    datasets = load_data2(dataset)

    train_set_states, train_set_actions, train_set_result_states, train_set_rewards = datasets[0]
    valid_set_states, valid_set_actions, valid_set_result_states, valid_set_rewards = datasets[1]
    test_set_states, test_set_actions, test_set_result_states, test_set_rewards = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_states.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_states.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_states.get_value(borrow=True).shape[0] / batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    state = T.matrix('state')  # the data is presented as rasterized images
    action = T.ivector('action')  # the labels are presented as 1D vector of
                           # [int] labels
    resultState = T.matrix('resultState')
    reward = T.dvector('reward')  # the labels are presented as 1D vector of [double]

    # construct the logistic regression class
    # Each MNIST image has size 28*28
    classifier = LogisticRegression(input=state, n_in=9, n_out=9)

    # the cost we minimize during training is the negative log likelihood of
    # the model in symbolic format
    # cost = classifier.negative_log_likelihood(y)
    cost = classifier.bellman_loss(state, reward, resultState)
    q_value = classifier.q_value2(state)

    # compiling a Theano function that computes the mistakes that are made by
    # the model on a minibatch
    givens ={
        state: test_set_states[index * batch_size:(index + 1) * batch_size],
        # action: test_set_actions[index * batch_size:(index + 1) * batch_size],
        resultState: test_set_result_states[index * batch_size:(index + 1) * batch_size],
        reward: test_set_rewards[index * batch_size:(index + 1) * batch_size]
        }
    test_model = theano.function(inputs=[index],
            outputs=classifier.bellman_error(state, reward, resultState),
            givens=givens)
    givens ={
        state: valid_set_states[index * batch_size:(index + 1) * batch_size],
        # action: valid_set_actions[index * batch_size:(index + 1) * batch_size],
        resultState: valid_set_result_states[index * batch_size:(index + 1) * batch_size],
        reward: valid_set_rewards[index * batch_size:(index + 1) * batch_size]
        }
    validate_model = theano.function(inputs=[index],
            outputs=classifier.bellman_error(state, reward, resultState),
            givens=givens)

    # compute the gradient of cost with respect to theta = (W,b)
    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)
    g_q_W = T.grad(cost=q_value, wrt=classifier.W)
    g_q_b = T.grad(cost=q_value, wrt=classifier.b)
    
    delta = classifier.delta(state, reward, resultState)

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs.
    updates = [(classifier.W, (classifier.W - learning_rate * g_W)),
               (classifier.b, (classifier.b - learning_rate * g_b))]
    print "g_W shape: " + str(g_W.shape[0])
    print "g_q_w shape: " + str(g_q_W.shape[0])
    print "delta shape: " + str(delta.shape[0])
    print "learing rate: " + str(learning_rate)
    #updates = [(classifier.W, (classifier.W + learning_rate * delta * g_q_W)),
    #           (classifier.b, (classifier.b + learning_rate * delta * g_q_b))]


    q_vals = classifier._q_vals
    # compiling a Theano function `train_model` that returns the cost, but in
    # the same time updates the parameter of the model based on the rules
    # defined in `updates`
    givens ={
        state: test_set_states[index * batch_size:(index + 1) * batch_size],
        # action: test_set_actions[index * batch_size:(index + 1) * batch_size],
        resultState: test_set_result_states[index * batch_size:(index + 1) * batch_size],
        reward: test_set_rewards[index * batch_size:(index + 1) * batch_size]
        }
    train_model = theano.function(inputs=[index],
            outputs=[cost, q_vals],
            updates=updates,
            givens=givens)
                     
    """
    self._train = theano.function([], [loss, q_vals], updates=updates,
                                      givens=givens)
    self._q_vals = theano.function([], q_vals,
                                       givens={states: self.states_shared})
    """
    #q_vals = theano.function(inputs=[index], outputs=[q_vals],
     #                   givens={state: test_set_states[index * batch_size:(index + 1) * batch_size],})

    ###############
    # TRAIN MODEL #
    ###############
    print '... training the model'
    # early-stopping parameters
    patience = 5000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                                  # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                  # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_params = None

    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            print "minibatch loss: " + str(minibatch_avg_cost)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i)
                                     for i in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    # test it on the test set

                    test_losses = [test_model(i)
                                   for i in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(
                        (
                            '     epoch %i, minibatch %i/%i, test error of'
                            ' best model %f %%'
                        ) %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            test_score * 100.
                        )
                    )

                    # save the best model
                    with open('best_model.pkl', 'w') as f:
                        cPickle.dump(classifier, f)

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(
        (
            'Optimization complete with best validation score of %f %%,'
            'with test performance %f %%'
        )
        % (best_validation_loss * 100., test_score * 100.)
    )
    print 'The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.1fs' % ((end_time - start_time)))
    
def predict():
    """
    An example of how to load a trained model and use it
    to predict labels.
    """

    # load the saved model
    classifier = cPickle.load(open('best_model.pkl'))

    # compile a predictor function
    predict_model = theano.function(
        inputs=[classifier.input],
        outputs=classifier.y_pred)
    
    predict_model_p = theano.function(
        inputs=[classifier.input],
        outputs=classifier.q_y_given_x)

    # We can test it on some examples from test test
    dataset = "controllerTuples.json"
    datasets = load_data2(dataset)
    test_set_states, test_set_actions, test_set_result_states, test_set_rewards = datasets[2]

    test_set_states = test_set_states.get_value()

    predicted_values = predict_model(test_set_states[:20])
    print ("Predicted values for the first 10 examples in test set:")
    print predicted_values
    predicted_p_values = predict_model_p(test_set_states[:20])
    print ("Predicted values for the first 10 examples in test set:")
    print predicted_p_values
    print numpy.sum(predicted_p_values, axis=1)[:20]
    print "W is: " + str(classifier.W.get_value())
    
    
if __name__ == '__main__':
    sgd_optimization_mnist()
