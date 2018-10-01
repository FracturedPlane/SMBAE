import matplotlib.pyplot as plt
import numpy
import theano
import theano.tensor as T
rng = numpy.random

N = 400
feats = 2
D = (rng.randn(N, feats), rng.randint(size=N, low=0, high=2))
for i in range(N):
    if D[1][i] > 0.5:
        D[0][i][1] += -2.0
    else:
        D[0][i][1] += 2.0
        
training_steps = 10000

# Declare Theano symbolic variables
x = T.matrix("x")
y = T.vector("y")
w = theano.shared(rng.randn(feats), name="w")
b = theano.shared(0., name="b")
print("Initial model:")
print(w.get_value())
print(b.get_value())

# Construct Theano expression graph
p_1 = 1 / (1 + T.exp(-T.dot(x, w) - b))   # Probability that target = 1
prediction = p_1 > 0.5                    # The prediction thresholded
xent = -y * T.log(p_1) - (1-y) * T.log(1-p_1) # Cross-entropy loss function
cost = xent.mean() + 0.01 * (w ** 2).sum()# The cost to minimize
gw, gb = T.grad(cost, [w, b])             # Compute the gradient of the cost
                                          # (we shall return to this in a
                                          # following section of this tutorial)

# Compile
train = theano.function(
          inputs=[x,y],
          outputs=[prediction, xent],
          updates=((w, w - 0.1 * gw), (b, b - 0.1 * gb)))
predict = theano.function(inputs=[x], outputs=prediction)
probb = theano.function(inputs=[x], outputs=p_1)

# Train
for i in range(training_steps):
    pred, err = train(D[0], D[1])

print("Final model:")
print(w.get_value())
print(b.get_value())
print("target values for D:")
print(D[1])
print("prediction on D:")
print(predict(D[0]))
print(probb(D[0]))

# print D[0]

colour = D[1]
plt.scatter(D[0][:,0],D[0][:,1] ,c=colour, s=20.0)

r = T.dmatrix('r')
q = 1 / (1 + T.exp(-r))
logistic = theano.function([r], q)
_x = [numpy.linspace(-4, 4, 100)] #  * w.get_value()[0] 
print _x
y = logistic(_x) * w.get_value()[1] +  b.get_value()

plt.plot(_x[0],y[0])
plt.axis("tight")

plt.show()

