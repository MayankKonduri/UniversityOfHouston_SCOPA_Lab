import numpy as np
import sys

from RDPDE import *


alpha_lo = 0.0001
alpha_up = 0.01
ns = 10000 # number of samples

n = 128
nt = 16

u0 = 1.0 # initial condition

# draw samples uniformly over the hal-open interval [low,high); each
# value in this interval is equally likely to be drawn
bounds = [alpha_lo, alpha_up] # upper and lower bound for paramters
alpha_spl = np.random.uniform( bounds[0], bounds[1], ns )


# define object for heat equation
# pde = HeatPDE()
pde = RDPDE()
pde.do_setup( n, nt )

x = np.linspace(0, 1, n+1 )
state_spl = np.zeros( (ns, n+1) )
# for samples drawn, compute solution
for i in range(ns):
    print('generating sample', i, 'of', ns)
    # "draw" sample
    alpha = alpha_spl[ i ]

    # solve forward problem
    u = pde.fwd_sol( u0, alpha, 0.05 )


    # store last time point
    state_spl[i] = u[:,-1]

# store files
y, theta = [np.array(state_spl), np.array(alpha_spl)]

# write out
xfile = 'rdiffEQ-nn-spl-' + str(ns) + '.npz'
np.savez( xfile, y=y, theta=theta )
