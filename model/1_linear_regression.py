import theano
from theano import tensor as T
import numpy as np
import matplotlib.pyplot as plt

trX = np.linspace(-1, 1, 101)
trY = 2 * trX + np.random.randn(*trX.shape) * 0.33
trY = trY + 2

X = T.scalar()
Y = T.scalar()

def model(X, w, b):
    return (X * w) + b

w = theano.shared(np.asarray(0., dtype=theano.config.floatX))
b = theano.shared(np.asarray(0., dtype=theano.config.floatX))
y = model(X, w, b)

cost = T.mean(T.sqr(y - Y))
# p_for_x = y - Y
gradient = T.grad(cost=cost, wrt=w)
gradient_b = T.grad(cost=cost, wrt=b)
updates = [[w, w - gradient * 0.01],
           [b, b - gradient_b * 0.01]]

train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
# predict = theano.function(inputs=[X, Y], outputs=p_for_x, updates=updates, allow_input_downcast=True)

for i in range(100):
    for x, y in zip(trX, trY):
        train(x, y)
        
#print predict(trX)
        
print w.get_value() #something around 2
print b.get_value()

w_ = w.get_value()
b_ = b.get_value()

plt.scatter(trX,trY)
plt.plot([trX[0], trX[-1]], [trX[0]*w_+b_, trX[-1]*w_+b_])
plt.axis("tight")

plt.show()

