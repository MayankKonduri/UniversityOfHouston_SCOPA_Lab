import numpy as np
import math
import scipy as sp
import matplotlib.pyplot as plt

# solve heat equation given by
# p_t u - alpha*\lap(u) + f(u) = 0, with u = u0 at t=0
# where f is the reaction part

class RDPDE:
    def __init__(self):
        self.debug = 0
        self.lap = []
        self.u0 = []
        self.n = []
        self.nt = []


    def enable_debug(self):
        self.debug = 1


    def do_setup( self, n, nt ):
        # setup operators and variables
        h = 1.0 / (n+1)
        e = np.ones( n+1 )
        lap = sp.sparse.spdiags([e,-2.0*e,e], [-1,0,1], n+1, n+1, format="csr" )

        # neumann boundary conditions
        ld = lap.tolil()
        ld[  0,  0] = -2.0
        ld[  0,  1] =  2.0
        ld[  n,  n] = -2.0
        ld[  n,n-1] =  2.0
        ld = ld / (h*h)

        #self.lap = ld.tocsr()
        self.lap = ld.toarray()
        self.n = n
        self.nt = nt

        self.setup_u0( )

        return

    def setup_u0( self ):

        self.u0 = np.zeros( self.n )
        sigma = 0.0005
        # set initial condition
        mu = 1.0/2.0
        h = 1.0 / self.n

        x = np.linspace( 0.0, 1.0, self.n+1 )
        self.u0 = np.exp( -0.5 * (x - mu)**2 / sigma )


    def eval_logreac( self, x ):
        # evaluate logistic reaction model
        return np.multiply( x, (1.0 - x) )


    def eval_expreac( self, x ):
        # evaluate exponential reaction model
        return x


    def fwd_sol( self, u0, alpha, rho ):

        # allocate memory for solution
        u = np.zeros( ( self.n+1, self.nt+1 ) )
        u[:,0] = self.u0

        # u = self.fwd_sol_rk2( u, alpha, rho )
        u = self.fwd_sol_os( u, alpha, rho )

        return u


    def fwd_sol_rk2( self, u, alpha, rho ):

        # define rhs
        f = lambda x : self.eval_logreac( x )
        rhs = lambda x : alpha*(self.lap @ x) + rho*f( x )

        ht = 1.0 / (self.nt + 1)

        # execute RK2 (2nd order runge Kutta)
        # time integrator
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


    def fwd_sol_os( self, u, alpha, rho ):

        #id = sp.sparse.identity( self.n + 1, format="csr")
        id = np.identity( self.n + 1 )  # ✅ dense identity
        ht = 1.0 / (self.nt + 1)

        # setup matrices for crank nicholson
        s = 0.25*ht*alpha
        BL = id - s*self.lap
        BR = id + s*self.lap

        # execute RK2 (2nd order runge Kutta)
        # time integrator
        for k in range(self.nt):
            # asign and map to a column vector
            uk = u[:,k].reshape(self.n+1,1)

            # first crank nicholson step
            b = BR @ uk
            uk = sp.sparse.linalg.spsolve( BL, b )

            # solve reaction equation analytically
            for i in range(self.n+1):
                #print(f"Before: uk[{i}] = {uk[i]}")
                
                # if ( uk[i] > 0.0 ):
                #     uk[i] = 1.0 / (1.0 + (((1.0 - uk[i])/(uk[i] + 1e-8)) * math.exp( -rho*ht )))

                uk_val = uk[i][0] if isinstance(uk[i], np.ndarray) else uk[i]  # handle shape safely

                # Clamp to avoid division by zero or instability
                uk_val = max(min(uk_val, 1.0 - 1e-6), 1e-6)

                # Apply logistic formula safely
                uk[i] = 1.0 / (1.0 + ((1.0 - uk_val) / uk_val) * math.exp(-rho * ht))

                #print(f"After: uk[{i}] = {uk[i]}")

                

            # map to a column vector
            uk = uk.reshape(self.n+1,1)

            # second crank nicholson step
            b = BR @ uk
            uk = sp.sparse.linalg.spsolve( BL, b )


            # map to a row vector
            u[:,k+1] = uk

        return u
