import numpy as np

a = np.array( [[1,2,0], [4,5,0], [10,11,0]] )

b = [7,8,1]

c = np.vstack((a,b))

print(c[c[:,1].argsort()])


# or

d = np.insert(a, 2, b, 0) # index 2 along axis 0

print(d)