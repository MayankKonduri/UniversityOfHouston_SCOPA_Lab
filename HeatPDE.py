import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# solve heat equation given by
# p_t u - alpha*\lap(u) = 0, with u = u0 at t=0

class HeatPDE:
    def __init__(self):
        self.debug = 0
        self.lap = []
        self.n = []
        self.nt = []


    def enable_debug(self):
        self.debug = 1


    def do_setup( self, n, nt ):
        # setup operators and variables
        h = 1.0 / (n+1)
        e = np.ones( n+1 )
        lap = sp.sparse.spdiags([e,-2.0*e,e], [-1,0,1], n+1, n+1 )

        # neumann boundary conditions
        ld = lap.todense()
        ld[  0,  0] = -2.0
        ld[  0,  1] =  2.0
        ld[  n,  n] = -2.0
        ld[  n,n-1] =  2.0
        ld = ld # / (h*h)

        self.lap = ld
        self.n = n
        self.nt = nt

        return


    def fwd_sol( self, u0, alpha ):

        # define rhs
        rhs = lambda x : alpha*(self.lap @ x)

        # allocate memory for solution
        u = np.zeros( ( self.n+1, self.nt+1 ) )
        ht = 1.0 / (self.nt + 1)

        # set initial guess
        m = int(self.n/2.0)
        u[m,0] = u0


        # execute time integrator
        for k in range(self.nt):
            uk = u[:,k]
            # map to a colum vector
            uk = uk.reshape(self.n+1,1)

            # rk2 time integration
            rhs1 = rhs( uk )
            u_pred = uk + ht*rhs1
            rhs2 = rhs( u_pred )
            uk = uk + (ht/2.0)*(rhs1 + rhs2)

            # map to a row vector
            u[:,k+1] = uk.T

        return u


