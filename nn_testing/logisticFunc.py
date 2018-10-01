
import matplotlib as mpl
import matplotlib.pyplot as plt

import theano
import theano.tensor as T
import numpy as np
x = T.dmatrix('x')
s = 1 / (1 + T.exp(-x))
logistic = theano.function([x], s)
_x = [[0, 1, 2, 3, 4, 5], [-1, -2, -3, -4, -5, -6]]
_x = [np.linspace(-6, 6, 100)]
print _x
y = logistic(_x)
print x
print s

plt.plot(_x[0],y[0])
plt.axis("tight")

plt.show()