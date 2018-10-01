
import numpy as np
import math

if __name__ == '__main__':
    
    samples = 100000
    x = np.array(np.random.normal(0,2.5,size=(samples,3)))
    
    
    
    print ("x mean: ", np.mean(x, axis=0))
    print ("x std: ", np.std(x, axis=0))
    x_mean = x[0]
    x_mean2 = x[0]
    x_std = (x[1] - ((x[0]+x[1])/2.0)**2)/2
    x_std2 = x_std
    variance = x_std
    for i in range(1,len(x)):
        x_mean_old = x_mean
        x_mean = x_mean + ((x[i] - x_mean)/i)
    
        if i > 1:
            x_std = (((i-2)*x_std) + ((i-1)*(x_mean_old - x_mean)**2) + ((x[i] - x_mean)**2))
            x_std = (x_std/float(i-1))
            
            x_std2 = x_std2 + (x[i]-x_mean_old)*(x[i]-x_mean)
            
            variance += (x[i-1]-x[i])*(x[i]-x_mean+x[i-1]-x_mean_old)/(i)
        
    
    print ("Running x_mean: ", x_mean)
    print ("Running x_std: ", np.sqrt(x_std))
    print ("Running x_std2: ", np.sqrt(x_std2/samples))
    print ("Running variance: ", (variance))