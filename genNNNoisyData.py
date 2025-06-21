import numpy as np
import sys
import matplotlib.pyplot as plt

from RDPDE import *


alpha_lo = 0.0001
alpha_up = 0.01
ns = 10000 # number of samples
#delta = 0.01 # noise paramter
delta = 0.1 # noise paramter

n = 128
nt = 16


u0 = 1.0 # initial condition

# draw samples uniformly over the hal-open interval [low,high); each
# value in this interval is equally likely to be drawn
bounds = [alpha_lo, alpha_up] # upper and lower bound for paramters
alpha_spl = np.random.uniform( bounds[0], bounds[1], ns )


# define object for heat equation
pde = RDPDE()
pde.do_setup( n, nt )

x = np.linspace(0, 1, n+1 )
state_spl = np.zeros( (ns, n+1) )
# for samples drawn, compute solution
for i in range(ns):
    print('generating sample', i, 'of', ns)
    # "draw" sample
    alpha = alpha_spl[ i ]
    noise = np.random.randn(n+1)

    # solve forward problem
    u = pde.fwd_sol( u0, alpha, 0.05 )

    # store last time point
    state_spl[i] = u[:,-1] + delta*noise

#    plt.plot(state_spl[i])
#    plt.show()


# store files
y, theta = [np.array(state_spl), np.array(alpha_spl)]

# write out
xfile = 'rdiffEQ-nn-spl-' + str(ns) + '-addnoise-delta=' + str(delta) +'.npz'
np.savez( xfile, y=y, theta=theta )
