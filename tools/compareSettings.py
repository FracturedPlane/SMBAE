
"""
    This compares the layer sizes, parameter values and activations of networks.

"""

import sys
sys.path.append('../')
import theano
from theano import tensor as T
import numpy as np
import lasagne
import os
import json


def compareSettings(agent1, agent2):
    """
        Prints out information related to the size, shape and parameter values of the networks
    """
    
    for key in agent1.keys():
        # print(" key: ", key)
        if ( key in agent2 ):
            if ( agent1[key] == agent2[key] ):
                pass
            else:
                print("Data differs for key ", key)
                print("agent1 value ", agent1[key])
                print("agent2 value ", agent2[key])
        else:
            print("Agent2 does not contain key ", key)
            print("with value key ", agent1[key])
    
    
if __name__ == '__main__':
    
    file = open(sys.argv[1])
    settings = json.load(file)
    file.close()
    net1 = settings
    
    file = open(sys.argv[2])
    settings = json.load(file)
    file.close()
    net2 = settings
    
    compareSettings(net1, net2)