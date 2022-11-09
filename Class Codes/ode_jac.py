import numpy as np

def doubleexp(x,y):
    return np.array([[998.0,1998.0],[-999.0,-1999.0]]),np.zeros(2)

def enrightpryce(x,y):
    dfdx      = np.zeros(3)
    dfdy      = np.zeros((3,3))
    dfdy[0,0] = -0.013-100.0*y[2]
    dfdy[0,1] = 0.0
    dfdy[0,2] = -1000.0*y[0]
    dfdy[1,0] = 0.0
    dfdy[1,1] = -2500.0*y[2]
    dfdy[1,2] = -2500.0*y[1]
    dfdy[2,0] = -0.013-1000.0*y[2]
    dfdy[2,1] = -2500.0*y[2]
    dfdy[2,2] = -1000.0*y[0]-2500.0*y[1] 
    return dfdy,dfdx
