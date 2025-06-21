import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append("../")

from RDPDE import *

n = 128
nt = 128

# define RDEPDE object
pde = RDPDE()
pde.do_setup( n, nt )
u = pde.fwd_sol( 1.0, 0.001, 0.5 )

x = np.linspace(0, 1, n+1 )
fig, ax = plt.subplots(1)
plt.plot( x, u[:, 0] )
plt.plot( x, u[:,-1] )
plt.show()
