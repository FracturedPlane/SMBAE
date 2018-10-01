import numpy as np
import lasagne
import sys
sys.path.append('../')
from model.ModelUtil import *
import collections
import heapq 

# For debugging
# theano.config.mode='FAST_COMPILE'
from model.Sampler import Sampler
from model.ForwardDynamicsSimulator import ForwardDynamicsSimulator

class AStarNode(object):
    
    def __init__(self, id, parent_id, cost, heuristic, action, state):
        self._id = id                        # node id
        self._parent = parent_id             # parent node id
        self._g = cost                       # actual cost to reach this node
        self._h = heuristic                  # estimate cost from this node to end node
        self._f = cost + heuristic           # g + h
        self._action = action                # action to get to this state
        self._state = state                # action to get to this state
        
class SuccesorNode(object):
    
    def __init__(self, cost, goal):
        self._cost = cost                       # actual cost to reach this node
        self._target = goal                  # estimate cost from this node to end node

class ForwardPlanner(Sampler):
    """
        This model using a forward dynamics network to compute the reward directly
    """
    def __init__(self, exp, look_ahead):
        super(ForwardPlanner,self).__init__()
        self._look_ahead=look_ahead
        self._exp=exp
        self._next_id = 0
        
    def sampleModel(self, model, forwardDynamics, current_state, anchors):
        print ("Starting Forward Model Sampling")
        # _bestSample = self._sampleModel(model, forwardDynamics, current_state, self._look_ahead)
        self._anchors=anchors
        _bestSample = self.findBestPath(start=current_state, goal=current_state.getID() + self._look_ahead)
        return _bestSample
    
    
    def predict(self, state):
        """
        Returns the best action
        """
        # hacky for now
        if isinstance(self._fd, ForwardDynamicsSimulator):
            self._fd.initEpoch(self._exp)
            state = self._exp.getEnvironment().getState()
        
        self.sampleModel(model=self._pol, forwardDynamics=self._fd, current_state=state, anchors=self._settings["anchors"][0])
        action = self.getBestSample()
        print ("Best Action: " + str(action))
        action = action[0]
        return action
    
    def cost(self, previous, current):
        d = travelDist(previous.getParams(), current.getParams())
        return d 
    
    def getHeuristic(self, start, finish):
        return goalDistance(start, self._anchors, finish)
    
    def getSuccesors(self, parent_node, goal):
        # samples = np.random.multivariate_normal(action, np.diag([0.1]*self._pol._action_length), math.power(self._settings["num_uniform_action_samples"], self._pol._action_length))
        samples = self.generateSamples(self._pol._action_bounds,  num_samples=self._settings["num_uniform_action_samples"])
        nodes=[]
        if parent_node._state.getID() >= goal:
            return nodes
        for samp in samples: # name nodes
            id_ = self._next_id
            self._next_id += 1
            prediction = self._fd._predict(state=parent_node._state, action=samp)
            cost = self.cost(parent_node._state, prediction) + parent_node._g 
            heuristic = self.getHeuristic(prediction, goal)
            node = AStarNode(id_, parent_node._id, cost, heuristic, samp, prediction)
            nodes.append(node)
        return nodes
    
    def findBestPath(self, start, goal):
        openList = [];
        closedList = {};
        self._state_grab_loc = self._exp.getEnvironment().getSimInterface().getController().getLinks()[0].getCenterOfMassPosition() 
        heuristic = self.getHeuristic(start, goal) + 20
        successors = []
    
        # create start node
        startNode = AStarNode(self._next_id, -1, 0, heuristic, [0]*self._pol._action_length, start)
        self._next_id+=1
        print ("Start: " + str(start.getID()) + " Goal: " + str(goal))
        print ("Adding node to heap: " + str((heuristic, startNode)))
        heapq.heappush(openList, (heuristic, startNode) )
        bestHeuristicNode = startNode;
        
        
        while (len(openList) > 0):
            (f, node) = heapq.heappop(openList)
            """
            // if we found the target, wrap up search and leave
            if (node.m_id == target)
            {
                closedList.insert(node);
                m_path = closedList.constructPath(start, target);
                break;
            }
            """
            succesors = self.getSuccesors(node, goal)
            for succ in succesors:
                                
                """
                // check if successor is in the open or closed lists
                // if it is, and the new successor g is better than the old one, remove it from the open or closed list
                
                if (openList.hasNode(successorId))
                {
                    if (openList.search(successorId)->m_g > successorg)
                        openList.remove(successorId);
                    else
                        continue;
                }
                else if (closedList.hasNode(successorId))
                {
                    if (closedList.search(successorId)->m_g > successorg)
                        closedList.remove(successorId);
                    else
                        continue;
                }
                """
                # print ("Adding node to heap: " + str((succ._f, succ)))
                heapq.heappush(openList, (succ._f, succ))
                
            closedList[node._id] = (node);
            if (node._f < bestHeuristicNode._f):
                bestHeuristicNode = node
            if (len(openList) < 1): 
                _path = self.constructPath(start, bestHeuristicNode._id, closedList);
                break;
        
        
        firstActionNode = _path[1]
        self._bestSample[1][0] = firstActionNode._f 
        self._bestSample[0] = firstActionNode._action
        
        print ("Path found: " + str(_path))
        
        return self._bestSample
        
        
    def constructPath(self, start, finish, list):
        path = []
        finish = list[finish]
        while(True):
            path.append(finish);
            print ("Path found: " + str(path))
            if(finish._parent == -1):
                break;
    
            finish = list[finish._parent]
        path.reverse()   
        return path
    