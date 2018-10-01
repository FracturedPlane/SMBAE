import numpy as np
import lasagne
import sys
sys.path.append('../')
sys.path.append("../characterSimAdapter/")
from model.ModelUtil import *
import math
import heapq
import copy
import os
# For debugging
# theano.config.mode='FAST_COMPILE'
from model.Sampler import Sampler
from model.ForwardDynamicsSimulator import ForwardDynamicsSimulator
from actor.ActorInterface import *



class Sample(object):
    
    def __init__(self, val, data):
        self._val = val
        self._data = data
        
    def __cmp__(self, other):
        return cmp(self._val, other._val)
    
    def __eq__(self, other):
        return self._val == other._val

class SequentialMCSampler(Sampler):
    """
        This model using a forward dynamics network to compute the reward directly
    """
    def __init__(self, exp, look_ahead, settings):
        super(SequentialMCSampler,self).__init__(settings)
        self._look_ahead=look_ahead
        self._exp=exp
        self._bad_reward_value=0
        
    def setEnvironment(self, exp):
        self._exp = exp
        self._fd.setEnvironment(exp)
    
    def sampleModel(self, model, forwardDynamics, current_state):
        print ("Starting SMC sampling")
        state__ = self._exp.getSimState()
        _bestSample = self._sampleModel(model, forwardDynamics, current_state, self._look_ahead)
        self._exp.setSimState(state__)
        self._bestSample = _bestSample
        return _bestSample
    
    def _sampleModel(self, model, forwardDynamics, current_state, look_ahead):
        """
            The current state in this case is a special simulation state not the same as the
            input states used for learning. This state can be used to create another simulation environment
            with the same state.
        
        """
        _action_dimension = len(self.getSettings()["action_bounds"][0])
        _action_bounds = np.array(self.getSettings()["action_bounds"])
        # import characterSim
        _bestSample=[[0],[-10000000], [], []]
        self._samples=[]
        current_state_copy = current_state 
        if isinstance(forwardDynamics, ForwardDynamicsSimulator):
            # current_state_copy = characterSim.State(current_state.getID(), current_state.getParams())
            current_state_copy = current_state
        # print ("Suggested Action: " + str(action) + " for state: " + str(current_state_copy))
        _action_params = []
        samples = []
        if self.getSettings()["use_actor_policy_action_suggestion"]:
            variance____=self.getSettings()['variance_scalling']
            variance__=[variance____]
            current_state_copy2 = copy.deepcopy(current_state_copy)
            current_state_copy__ = self._exp.getStateFromSimState(current_state_copy)
            _bestSample[0] = model.predict(np.array([current_state_copy__]))
            ## Get first action
            for i in range(look_ahead):
                if isinstance(forwardDynamics, ForwardDynamicsSimulator):
                    current_state_copy3 = copy.deepcopy(current_state_copy2)
                    """
                    if ( (not (np.all(np.isfinite(current_state_copy3)) and (np.all(np.greater(current_state_copy3, -10000.0))) and (np.all(np.less(current_state_copy3, 10000.0))))) or 
                            self._exp.endOfEpoch()  
                         ): # lots of nan values for some reason...
                        print("Found bad sim state in search", current_state_copy3)
                        print("endOfEpoch(): ", self._exp.endOfEpoch())
                        break
                    """
                    current_state_copy__ = self._exp.getStateFromSimState(current_state_copy)
                    """
                    if ( (not (np.all(np.isfinite(current_state_copy__)) and (np.all(np.greater(current_state_copy__, -10000.0))) and (np.all(np.less(current_state_copy__, 10000.0))))) or 
                            self._exp.endOfEpoch()  
                         ): # lots of nan values for some reason...
                        print("Found bad state in search", current_state_copy__)
                        print("endOfEpoch(): ", self._exp.endOfEpoch())
                        break
                    """
                    # print ("current_state_copy__: ", current_state_copy__)
                    pa = model.predict(np.array([current_state_copy__]))
                    """
                    if ( (not (np.all(np.isfinite(pa)) and (np.all(np.greater(pa, -10000.0))) and (np.all(np.less(pa, 10000.0)))))
                         ): # lots of nan values for some reason...
                        print("Found bad action in search: ", pa)
                        break
                    """
                    if self.getSettings()["use_actor_policy_action_variance_suggestion"]:
                        
                        lSquared =(4.1**2)
                        ## This uses the learned model so in the case the state given must be that used by the model
                        current_state_copy3__ = self._exp.getStateFromSimState(current_state_copy3)
                        variance__ = getModelPredictionUncertanty(model, current_state_copy3__, 
                                                        length=4.1, num_samples=32, settings=self.getSettings())
                        
                        variance__ = list(variance__) * look_ahead # extends the list for the number of states to look ahead
                        # print (var_)
                        if not all(np.isfinite(variance__)): # lots of nan values for some reason...
                            print ("Problem computing variance from model: ", )
                            print ("State: ", current_state_copy3__, " action: ", pa)
                            for fg in range(len(samp_)):
                                print ("Sample ", fg, ": ", samp_[fg], " Predictions: ", predictions_[fg])
                                
                        print ("Predicted Variance: " + str(variance__))
                    else:
                        variance__=[variance____]*(len(pa)*look_ahead)
                else:
                    variance__=[variance____]*len(action)
                    pa = model.predict(current_state_copy2)
                
                action = pa
                _action_params.extend(action)
                # print ("Suggested Action: " + str(action) + " for state: " + str(current_state_copy) + " " + str(current_state_copy.getParams()) + " with variance: " + str(variance__))
                current_state_copy3 = forwardDynamics._predict(state__c=current_state_copy2, action=pa)
                # samples = self.generateSamples(_action_bounds,  num_samples=5)
                # samples = self.generateSamples(bounds,  num_samples=self._settings["num_uniform_action_samples"])
            # num_samples_ = pow(self.getSettings()["num_uniform_action_samples"], _action_dimension)
            num_samples_ = self.getSettings()["num_uniform_action_samples"] * (_action_dimension)
            # print ("Number of initial random samples: ", num_samples_)
            # variance__=[variance____]*(len(_action_params))
            # print ("_action_params: ", _action_params, " variance: ", variance__, " for pid: ", os.getpid())
            samples = self.generateSamplesFromNormal(mean=_action_params, num_samples=num_samples_, variance_=variance__)
        else:
            num_samples_ = self.getSettings()["num_uniform_action_samples"] * (_action_dimension)
            # samples = self.generateSamples(_action_bounds,  num_samples=self.getSettings()["num_uniform_action_samples"], repeate=look_ahead)
            samples = self.generateSamplesUniform(_action_bounds,  num_samples=num_samples_, repeate=look_ahead)
            # print ("Samples: ", samples)
        # print ("Current state sample: " + str(current_state_copy.getParams()))
        """
        if not ( np.all(np.isfinite(current_state_copy)) and (np.all(np.greater(current_state_copy, -10000.0))) and (np.all(np.less(current_state_copy, 10000.0))) ): # lots of nan values for some reason...
            print("Found bad Current State in search")
            return _bestSample
        """
        for sample in samples:
            pa = sample
            # print ("sample: " + str(sample))
            actions_ = chunks(sample, _action_dimension)
            actions=[]
            for chunk in actions_:
                act__ = clampAction(chunk, _action_bounds)
                actions.extend(act__)
            # self.updateSampleWeights()
            actions=list(chunks(actions, _action_dimension))
            
            y=[]
            init_states=[]
            predictions=[]
            if isinstance(forwardDynamics, ForwardDynamicsSimulator):
                current_state_ = copy.deepcopy(current_state_copy)
                # actions = chunks(sample, _action_dimension)
                forwardDynamics.setSimState(current_state_)
                for a in range(len(actions)):
                    
                    if ( ((not np.all(np.isfinite(actions[a])) or (np.any(np.less(actions[a], -10000.0))) or (np.any(np.greater(actions[a], 10000.0)))) or
                            forwardDynamics.endOfEpoch()  ) 
                         ): # lots of nan values for some reason...
                        print("Found bad action in search at: ", a)
                        ## Append bad values for the rest of the actions
                        self._bad_reward_value
                        y.append(self._bad_reward_value)
                        continue
                        # break
                    
                    current_state__ = self._exp.getStateFromSimState(current_state_)
                    init_states.append(current_state__)
                    (prediction, reward__) = forwardDynamics._predict(state__c=current_state_, action=actions[a])
                    # epochEnded = forwardDynamics.endOfEpoch()
                    # print ("Epoch Ended: ", epochEnded, " on action: ", a)
                    prediction_ = self._exp.getStateFromSimState(prediction)
                    
                    if ( ( not np.all(np.isfinite(prediction_))) or (np.any(np.less(prediction_, -10000.0))) or (np.any(np.greater(prediction_, 10000.0))) ): # lots of nan values for some reason...
                        print("Reached bad state in search")
                        # break
                    
                        
                    predictions.append(prediction_)
                    # print ("Current State: ", current_state_.getParams(), " Num: ", current_state_.getID())
                    # print ("Prediction: ", prediction.getParams(), " Num: ", prediction.getID())
                    # print ("Executed Action: ", actions[a])
                    ## This reward function is not going to work anymore
                    y.append(reward__)
                    current_state_ = copy.deepcopy(prediction)
                    # goalDistance(np.array(current_state_.getParams()), )
                    # print ("Y : " + str(y))
                    
            else:
                current_state_=current_state_copy
                # actions = chunks(sample, _action_dimension)
                for a in range(len(actions)):
                    init_states.append(current_state_)
                    prediction = forwardDynamics.predict(state=current_state_, action=actions[a])
                    if ( not (np.all(np.isfinite(prediction)) and (np.all(np.greater(prediction, -10000.0))) and (np.all(np.less(prediction, 10000.0)))) ): # lots of nan values for some reason...
                        print("Reached bad state in search")
                        # break
                    
                    predictions.append(prediction)
                    y.append(reward(current_state_, prediction))
                    current_state_ = prediction
            # print (pa, y, id(y))
            if ( np.all(np.isfinite(y)) and (np.all(np.greater(y, -10000.0))) and (np.all(np.less(y, 10000.0))) ): # lots of nan values for some reason...
                # print ("Good sample:")
                self.pushSample(sample, self.discountedSum(y))
            else : # this is bad, usually means the simulation has exploded...
                print ("Y: ", y, " Sample: ", sample)
                print (" current_state_: ", current_state_copy)
                # self._fd.initEpoch(self._exp)
                # return _bestSample
                
                
            if self.discountedSum(y) > self.discountedSum(_bestSample[1]):
                _bestSample[1] = y
                _bestSample[0] = pa[:_action_dimension]
                _bestSample[2] = init_states
                _bestSample[3] = predictions
            del y
        
        self.updateSampleWeights()
        print ("Starting Importance Sampling: *******")
        # print ("Current state sample: " + str(current_state_copy.getParams()))
        for i in range(self.getSettings()["adaptive_samples"]): # 100 samples from pdf
            # print ("Data probabilities: " + str(self._data[:,1]))
            # print ("Data rewards: " + str(self._data[:,0]))
            sample = self.drawSample()
            actions_ = chunks(sample, _action_dimension)
            actions=[]
            for chunk in actions_:
                act__ = clampAction(chunk, _action_bounds)
                actions.extend(act__)
            self.updateSampleWeights()
            actions=list(chunks(actions, _action_dimension))
            # print ("Action samples: " + str(list(actions)))
            """
            for item in self._samples:
                if all(item[1][0] == sample): # skip item already contained in samples
                    print ("Found duplicate***")
                    continue
            """
            pa = sample
            # print ("sample: " + str(sample))
            y=[]
            init_states=[]
            predictions=[]
            if isinstance(forwardDynamics, ForwardDynamicsSimulator):
                current_state_ = copy.deepcopy(current_state_copy)
                # actions = chunks(sample, _action_dimension)
                forwardDynamics.setSimState(current_state_)
                for a in range(len(actions)):
                    if ( ((not np.all(np.isfinite(actions[a])) or (np.any(np.less(actions[a], -10000.0))) or (np.any(np.greater(actions[a], 10000.0)))) or
                            forwardDynamics.endOfEpoch()  )
                         ): # lots of nan values for some reason...
                        print("Found bad action in search at: ", a)
                        ## Append bad values for the rest of the actions
                        self._bad_reward_value
                        y.append(self._bad_reward_value)
                        continue
                        # break
                    current_state__ = self._exp.getStateFromSimState(current_state_)
                    init_states.append(current_state__)
                    (prediction, reward__) = forwardDynamics._predict(state__c=current_state_, action=actions[a])
                    # epochEnded = forwardDynamics.endOfEpoch()
                    # print ("Epoch Ended: ", epochEnded, " on action: ", a)
                    if ( not (np.all(np.isfinite(prediction)) and (np.all(np.greater(prediction, -10000.0))) and (np.all(np.less(prediction, 10000.0)))) ): # lots of nan values for some reason...
                        print("Reached bad state in search")
                        # break
                    
                    prediction_ = self._exp.getStateFromSimState(prediction)
                    predictions.append(prediction_)
                    # print ("Current State: ", current_state_.getParams(), " Num: ", current_state_.getID())
                    # print ("Prediction: ", prediction.getParams(), " Num: ", prediction.getID())
                    # print ("Executed Action: ", actions[a])
                    ## This reward function is not going to work anymore
                    y.append(reward__)
                    current_state_ = copy.deepcopy(prediction)
                    # goalDistance(np.array(current_state_.getParams()), )
                    # print ("Y : " + str(y))
                    
            else:
                current_state_=current_state_copy
                # actions = chunks(sample, _action_dimension)
                for a in range(len(actions)):
                    init_states.append(current_state_)
                    prediction = forwardDynamics.predict(state=current_state_, action=actions[a])
                    if ( not (np.all(np.isfinite(prediction)) and (np.all(np.greater(prediction, -10000.0))) and (np.all(np.less(prediction, 10000.0)))) ): # lots of nan values for some reason...
                        print("Reached bad state in search")
                        # break
                    predictions.append(prediction)
                    y.append(reward(current_state_, prediction))
                    current_state_ = prediction
                    
            # print (pa, y)
            if ( np.all(np.isfinite(y)) and (np.all(np.greater(y, -10000.0))) and (np.all(np.less(y, 10000.0))) ): # lots of nan values for some reason...
                # print ("Good sample:")
                self.pushSample(sample, self.discountedSum(y))
                if self.discountedSum(y) > self.discountedSum(_bestSample[1]):
                    _bestSample[1] = y
                    _bestSample[0] = pa[:_action_dimension]
                    _bestSample[2] = init_states
                    _bestSample[3] = predictions
            else : # this is bad, usually means the simulation has exploded...
                print ("Y: ", y, " Sample: ", sample)
                print (" current_state_: ", current_state_copy)
                # self._fd.initEpoch(self._exp)
                # return _bestSample
        _bestSample[1] = self.discountedSum(_bestSample[1])
        # print ("Best Sample: ", _bestSample[0], _bestSample[1])
        return _bestSample
    
    def discountedSum(self, rewards, discount_factor=0.7):
        """
            Assumed first reward was earliest
        """
        discounted_sum=0
        for state_num in range(len(rewards)):
            discounted_sum += (math.pow(discount_factor,state_num) * rewards[state_num])
        return discounted_sum
    
    def predict(self, state, evaluation_=False):
        """
            Returns the best action
        """
        ## hacky for now
        if ( not evaluation_ ):
            if isinstance(self._fd, ForwardDynamicsSimulator):
                # print ( "SMC exp: ", self._exp)
                self._fd.initEpoch(self._exp)
                # state = self._exp.getState()
                # state_ = self._exp.getSimState()
                self._exp.setSimState(state)
            # if ( self._exp.endOfEpoch() ):
            #     print ("Given back state where it is already endOfEpoch()")
            #     return self._pol.predict(state)
            
            self.sampleModel(model=self._pol, forwardDynamics=self._fd, current_state=state)
            action = self.getBestSample()
            self._exp.setSimState(state)
            # if isinstance(self._fd, ForwardDynamicsSimulator):
            #     self._fd._sim.setState(state)
            # print ("Best Action SMC: " + str(action))
            action = action[0]
            return action
        else:
            return super(SequentialMCSampler, self).predict(state, evaluation_=evaluation_)
    
    def pushSample(self, action, val):
        # print ("Val: " + str(val))
        # print ("[action,val]: " + str([action,val]))
        
        # print ("Samples: " )
        # print ("\n\n\n")
        samp = Sample(val, action)
        heapq.heappush(self._samples, samp)
        
    
    def updateSampleWeights(self):
        num_samples=self.getSettings()["num_adaptive_samples_to_keep"]
        if num_samples > len(self._samples):
            num_samples = len(self._samples)
        data_ = heapq.nlargest(num_samples, self._samples)
        data = []
        for item in data_:
            data.append([item._data, item._val])
        # data = list(data)
        # print ("Data: " + str(data))
        data = np.array(data)
        # print ("Largest N: " + str(data[:,1]))
        min = np.min(data[:,1], 0)
        max = np.max(data[:,1], 0)
        diff = max-min
        if 0.0 == diff: ## To prevent division by 0
            print ("Diff contains zero: " + str(diff))
            print ("Data, largets N: " + str(data[:,1]))
            ## JUst make everything uniform then...
            weights = np.ones(len(data[:,1]))/float(len(data[:,1]))
        else:    
            data_ = (data[:,1]-min)/(diff)
            # data_ = data[:,1]-min
            sum = np.sum(data_, 0) 
            weights = data_ / sum
        self._data = copy.deepcopy(data)
        # print ("Weights: " + str(weights))
        # print ("Data: " + str(self._data))
        self._data[:,1] = np.array(weights, dtype='float64')
        # Delete old samples
        # self._samples = []
        # print ("Done computing pdf data: " + str(self._data))
        
    
    def drawSample(self):
        samp = np.random.choice(self._data[:,0], p=np.array(self._data[:,1], dtype='float64'))
        # samp = np.random.choice(self._data[:,0])
        # print ("Sample: " + str(samp))
        # print ("Sample type: " + str(samp[0].dtype))
        samples = self.generateSamplesFromNormal(samp, 1, variance_=self.getSettings()['variance_scalling'])
        return samples[0]
    