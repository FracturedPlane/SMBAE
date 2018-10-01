"""
    This script will help understand how the distribution changes
    when samples are taken from a Gaussian distribution in different ways.
"""

import numpy as np

if __name__ == '__main__':
    
    samples = 200
    x = []
    for i in range(samples):
        x.append(np.random.normal((i, 1.0, 1)[0]))
        
    print("mean: ", np.mean(x))
    print("std: ", np.std(x))