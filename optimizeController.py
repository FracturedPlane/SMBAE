import json
import sys
import numpy as np
from algorithm.cma import fmin
from algorithm.cma import fcts
import math
sys.path.append("../simbiconAdapter")
import simbiconAdapter

def gcos2(xs):
    sum=0
    for i in range(len(xs)):
        sum = sum + math.sin(float(xs[i])) + xs[i]
    return sum

def onethousand(x, simbicon, z):
    sum = 0
    sum = sum + math.fabs(x[0]-1000)
    print (sum, x)
    print (simbicon.getEvaluationData())
    return sum

def halfBad(x):
    if random.choice([0, 1]) > 0.5:
        return np.NaN
    else:
        # return np.NaN
        return onethousand(x)
    
def runsimbicon(x, simbicon, z):
    simbicon.initEpoch()
    print ("Action: ", x)
    for j in range(20): # Take 5 steps
        simbicon.act(x)
    
    averageSpeed = simbicon.getEvaluationData()[0]
    targetSpeed = 1.5;
    reward = math.fabs(targetSpeed - averageSpeed) # optimal is 0
    print ("AverageS Speed: ", simbicon.getEvaluationData(), " reward ", reward)
    return reward
    
# @profile(precision=5)
# @memprof(plot = True)
def train(settingsFileName):
    
    print ("Blah")
    file = open(settingsFileName, 'r')
    settings = json.load(file)
    print ("Settings: " , str(json.dumps(settings)))
    file.close()  
    
    action_bounds = settings["action_bounds"]
    
    c = simbiconAdapter.Configuration(str(settings['sim_config_file']))
    print ("Num state: ", c._NUMBER_OF_STATES)
    sim = simbiconAdapter.SimbiconWrapper(c)
    sim.init()
    args_ = (sim, 0)
    
    lbound = [0.0, 0.0] # lower bounds
    ubound = [1.0, 1.0] # upper bounds
    bounds = [lbound, ubound]
    print bounds
    options = {'bounds':bounds, 'seed':1234, 'maxfevals':1000} # Setting options for optimization
    result = fmin(runsimbicon, x0=[0.1, 0.1], sigma0=0.15,  args=args_, **options)
    
    
    
if (__name__ == "__main__"):
    
    train(sys.argv[1])