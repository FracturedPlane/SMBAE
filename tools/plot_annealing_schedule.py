import matplotlib.pyplot as plt
# from matplotlib import mpl
import numpy as np
# import matplotlib.animation as animation
import random
import sys
import json
from NNVisualize import NNVisualize
import math
from model.ModelUtil import anneal_value




if __name__ == '__main__':
    
    file = open(sys.argv[1])
    settings = json.load(file)
    print ("Settings: " + str(json.dumps(settings, indent=4)))
    file.close()
    
    nv = NNVisualize("annealing schedule", settings=settings)
    
    rounds = settings['rounds']
    x = range(rounds)
    ps = []
    
    # anneal_type = 'linear'
    anneal_type = 'log'
    # anneal_type = 'square'
    
    settings['annealing_schedule'] = 'log'
    
    for round_ in range(0,rounds):
        # p = math.fabs(settings['initial_temperature'] / (math.log(round_*round_) - round_) )
        # p = (settings['initial_temperature'] / (math.log(round_))) 
        # p = ((settings['initial_temperature']/math.log(round_))/math.log(rounds))
        if ( 'annealing_schedule' in settings and (settings['annealing_schedule'] != False)):
            p = anneal_value(float(round_/rounds), settings_=settings)
        else:
            p = ((settings['initial_temperature']/math.log(round_+2))) 
        # p = ((rounds - round_)/rounds) ** 2
        p = max(settings['min_epsilon'], min(settings['epsilon'], p)) # Keeps it between 1.0 and 0.2
        if ( settings['load_saved_model'] ):
            p = settings['min_epsilon']
        ps.append(p)
    nv.init()    
    nv.updateLoss(ps, np.zeros(len(ps)))
    
    nv.show()