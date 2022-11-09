import numpy as np

#==============================================================
# function sign = lunarlanding(x,y,dx)
#
# Returns sign of y[0], i.e. of the altitude above ground.
# See ode_test.py, specifically ode_init.
#
#--------------------------------------------------------------

def lunarlanding(y):
    return np.sign(y[0])


