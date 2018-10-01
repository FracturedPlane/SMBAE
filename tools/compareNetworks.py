
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

from util.SimulationUtil import loadNetwork 

def visualizeNetworkWeights(layer_parameters, folder):
    """
        Turns the parameter values of each layer in the network into an images and saves them
        all to files
    """
    if not os.path.exists(folder):
            os.makedirs(folder)
    for i in range(len(layer_parameters)):
        print ("layer_parameters[i]: ", layer_parameters[i].shape)
        params = np.reshape(layer_parameters[i], (layer_parameters[i].shape[0], -1))
        print ("params.shape: ", params.shape)
        makeImageFromLayerParameters(params, folder + "/layer" + str(i))
    return 0

def makeImageFromLayerParameters(layer_parameters, path):
    """
        converts the layer parameters into an images
        saves the image.
    """
    from matplotlib import pyplot as plt
    plt.imshow(layer_parameters, interpolation='nearest')
    plt.savefig(path+".svg")
    # plt.show()
    return 0

def compareNetworks(agent1, agent2):
    """
        Prints out information related to the size, shape and parameter values of the networks
    """
    print ("blah")
    net1 = agent1.getModel()
    net2 = agent2.getModel()    
    print ("Comparing layer sizes:")
    net_layers = lasagne.layers.get_all_layers(net1.getActorNetwork()) 
    for i in range(len(net_layers)):
        print("Actor1 network layer ", i ," : ", net_layers[i])
        print("Layer size ", i ," : ", net_layers[i].output_shape)
    
    print("Other model")  
    net_layers = lasagne.layers.get_all_layers(net2.getActorNetwork()) 
    for i in range(len(net_layers)):
        print("Actor2 network layer ", i ," : ", net_layers[i])
        print("Layer size ", i ," : ", net_layers[i].output_shape)
        
    ### visualize layer weights
    net_params = lasagne.layers.get_all_param_values(net1.getActorNetwork())
    visualizeNetworkWeights(net_params, folder="viz_params1/")
    net_params = lasagne.layers.get_all_param_values(net2.getActorNetwork())
    visualizeNetworkWeights(net_params, folder="viz_params2/")
    
    
if __name__ == '__main__':
    
    
    net1 = loadNetwork(sys.argv[1])
    net2 = loadNetwork(sys.argv[2])
    compareNetworks(net1, net2)