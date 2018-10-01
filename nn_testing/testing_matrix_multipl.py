
import numpy as np



x = np.array([[1, 2, 3],
              [3, 4, 5]
              ])


y = np.array([[1, 2],
              [3, 4],
              [4, 5]
              ])

k = np.array([[1, 2],
              [3, 4]
              ])
print("x: ", x)
print("y: ", y)

z = np.dot(np.dot(x,y),k)

print (z)


x2 = np.array([[1, 2, 3],
              [3, 4, 5],
              [0, 0, 0]
              ])
z = np.dot(np.dot(x2,y),k)
print(z)