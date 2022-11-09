import numpy as np
import scipy.linalg as linalg
#==============================================================
# Explicit ODEs stepper functions
# Containing:
#   euler
#   rk2
#   rk4
#   rk45
#==============================================================
# function y = euler(fRHS,x0,y0,dx,**kwargs)
#
# Advances solution of ODE by one Euler step y = y0+f*dx, where f = y'
#
# input: 
#   fRHS   : function handle. Needs to return a vector of size(y0);
#   x0     : starting x
#   y0     : starting y(x0)
#   dx     : step size
#   kwargs : additional arguments for consistency with implicit steppers
# output:
#   y      : vector of results
#--------------------------------------------------------------

def euler(fRHS,x0,y0,dx,**kwargs):
    y  = y0 + dx*fRHS(x0,y0,dx)
    return y,1

#==============================================================
# function y = rk2(fRHS,x0,y0,dx,**kwargs)
#
# Advances solution of ODE by one RK2 step.
#
# input: 
#   fRHS   : function handle. Needs to return a vector of size(y0);
#   x0     : starting x
#   y0     : starting y(x0)
#   dx     : step size
#   kwargs : additional arguments for consistency with implicit steppers
# output:
#   y      : vector of results
#--------------------------------------------------------------

def rk2(fRHS,x0,y0,dx,**kwargs):
    k1 = dx*fRHS(x0       ,y0       ,dx)
    k2 = dx*fRHS(x0+0.5*dx,y0+0.5*k1,dx)
    y  = y0+k2
    return y,1

#==============================================================
# function y = rk4(fRHS,x0,y0,dx,**kwargs)
#
# Advances solution of ODE by one RK4 step.
#
# input: 
#   fRHS   : function handle. Needs to return a vector of size(y0);
#   x0     : starting x
#   y0     : starting y(x0)
#   dx     : step size
#   kwargs : additional arguments for consistency with implicit steppers
# output:
#   y      : vector of results
#--------------------------------------------------------------

def rk4(fRHS,x0,y0,dx,**kwargs):
    k1 = dx*fRHS(x0       ,y0       ,dx)
    k2 = dx*fRHS(x0+0.5*dx,y0+0.5*k1,dx)
    k3 = dx*fRHS(x0+0.5*dx,y0+0.5*k2,dx)
    k4 = dx*fRHS(x0+    dx,y0+    k3,dx)
    y  = y0+(k1+2.0*(k2+k3)+k4)/6.0
    return y,1

#==============================================================
# function y = rk45single(fRHS,x0,y0,dx)
#
# Advances solution of ODE by one RK45 step.
# See Numerical Recipes 1992, Sec 16.2. 
# Numpy's array operators allow for a much compacter code.
#
# input: 
#   fRHS   : function handle. Needs to return a vector of size(y0);
#   x0     : starting x
#   y0     : starting y(x0)
#   dx     : step size (determined by step_k45)
# output:
#   y      : vector of results
#--------------------------------------------------------------

def rk45single(fRHS,x0,y0,dx):
    a         = np.array([0.0,0.2,0.3,0.6,1.0,0.875]) # weights for x
    b         = np.array([[0.0           , 0.0        , 0.0          , 0.0             , 0.0         ],
                          [0.2           , 0.0        , 0.0          , 0.0             , 0.0         ],
                          [0.075         , 0.225      , 0.0          , 0.0             , 0.0         ],
                          [0.3           , -0.9       , 1.2          , 0.0             , 0.0         ],
                          [-11.0/54.0    , 2.5        , -70.0/27.0   , 35.0/27.0       , 0.0         ],
                          [1631.0/55296.0, 175.0/512.0, 575.0/13824.0, 44275.0/110592.0, 253.0/4096.0]])
    c         = np.array([37.0/378.0,0.0,250.0/621.0,125.0/594.0,0.0,512.0/1771.0])
    dc        = np.array([2825.0/27648.0,0.0,18575.0/48384.0,13525.0/55296.0,277.0/14336.0,0.25])
    dc        = c-dc
    n         = y0.size
    dy        = np.zeros(n)        # updates (arguments in f(x,y))
    dydx      = np.zeros((6,n))    # derivatives (k1,k2,k3,k4,k5,k6)
    yout      = y0                 # result
    yerr      = np.zeros(n)        # error
    dydx[0,:] = dx*fRHS(x0,y0,dx)  # first guess
    for i in range(1,6):           # outer loop over k_i 
        dy[:]     = 0.0
        for j in range(i):         # inner loop over y as argument to fRHS(x,y)
            dy = dy + b[i,j]*dydx[j,:]
        dydx[i,:] = dx*fRHS(x0+a[i]*dx,y0+dy,a[i]*dx)
    for i in range(0,6):           # add up the k_i times their weighting factors
        yout = yout + c [i]*dydx[i,:]
        yerr = yerr + dc[i]*dydx[i,:]   

    return yout,yerr

#==============================================================
# function y = rk5(fRHS,x0,y0,dx,**kwargs)
#
# Advances solution of ODE by one RK5 step.
# See Numerical Recipes 1992, Sec 16.2. 
# Numpy's array operators allow for a much compacter code.
# Discards the error value, for direct use as 5th-order stepper function.
#
# input: 
#   fRHS   : function handle. Needs to return a vector of size(y0);
#   x0     : starting x
#   y0     : starting y(x0)
#   dx     : step size (determined by step_k45)
#   kwargs : additional arguments for consistency with implicit steppers
# output:
#   y      : vector of results
#--------------------------------------------------------------

def rk5(fRHS,x0,y0,dx,**kwargs):
    yout,err = rk45single(fRHS,x0,y0,dx)
    return yout,1

#==============================================================
# function y,it = rk45(fRHS,x0,y0,dx,**kwargs)
#
# Advances solution of ODE by one RK45.
# See Numerical Recipes 1992, Sec. 16.2
#
# input: 
#   fRHS   : function handle. Needs to return a vector of size(y0);
#   x0     : starting x
#   y0     : starting y(x0)
#   dx     : step size (determined by ode_fixed)
#   kwargs : additional arguments for consistency with implicit steppers
#            eps: tolerance at which to accept solution. Default eps = 1e-10
# output:
#   y      : vector of results
#   it     : number of iterations used
#--------------------------------------------------------------

def rk45(fRHS,x0,y0,dx,**kwargs):
    eps     = 1e-10                                         # tolerance at which to accept solution
    for key in kwargs:                                      # if keyword eps is set, use this instead of default
        if (key=='eps'):
            eps = kwargs[key]
    pshrink = -0.25
    pgrow   = -0.2
    safe    = 0.9
    errcon  = 1.89e-4
    maxit   = 100000                                        # this should depend on problem
    xt      = 0.0                                           # temporary independent variable: will count up to dx.
    x1      = x0                                            # keeps track of absolute x position
    it      = 0                                             # iteration counter, as safeguard
    n       = y0.size
    dydx    = fRHS(x0,y0,dx)
    y1      = y0
    y2      = y0
    dxtry   = dx                                            # starting guess for step size    
    dxtmp   = dx
    idone   = 0
    while ((xt < dx) and (it < maxit)):
        yscal = np.abs(y1) + np.abs(dxtry*dydx)             # error scaling on last timestep, see NR92, sec 16.2
        idone = 0                                           # reset idone
        while (not idone):                                  # figure out an acceptable stepsize
            y2,yerr = rk45single(fRHS,x1,y1,dxtry)          # keep recalculating y2 using current trial step
            errmax  = np.max(np.abs(yerr/yscal)/eps)
            if (errmax > 1.0):                              # stepsize too large - reduce
                dxtmp = dxtry*safe*np.power(errmax,pshrink)
                dxtry = np.max(np.array([dxtmp,0.1*dxtry])) # warning! This is only for dxtry > 0.
                xnew  = x1 + dxtry
                if (xnew == xt):
                    raise Exception('[step_rk45]: xnew == xt. dxtry = %13.5e' % (dxtry))
            else:                                           # stepsize ok - we're done with the trial loop
                idone = 1
        y1 = y2                                             # update so that integration is advanced at next iteration.
        it = it+1
        if (errmax > errcon):                               # if the error is larger than safety, reduce growth rate
            dxnext = safe*dxtry*np.power(errmax,pgrow)
        else:                                               # if error less than safety, increase by factor of 5.
            dxnext = 5.0*dxtry
        xt    = xt + dxtry
        x1    = x1 + dxtry
        dxtry = np.min(np.array([dx-xt,dxnext]))            # guess next timestep - make sure it's flush with dx.
    return y2,it



