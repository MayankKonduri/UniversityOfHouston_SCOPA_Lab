import numpy as np
import matplotlib.pyplot as plt


# load samples
filename = "../heatEQ-nn-spl-10000.npz"
npzfile = np.load( filename )

# extract time series data and associated parameters
y = npzfile['y']
theta = npzfile['theta']

print(y.shape)
print(theta.shape)
print(theta)

# get dimensions
nplot = 10

ns = y.shape[0]
n = y.shape[1] - 1
if ns < nplot:
    nplot = ns

# generate list of random samples from state solutions
ids = np.random.permutation(ns)[:nplot]

print('number of samples', ns)
print('sample size', n)

# plot
x = np.linspace(0, 1, n+1 )
fig, ax = plt.subplots(1)
for i in range( nplot ):
    plt.plot(x, y[ids[i]] )

plt.show()
