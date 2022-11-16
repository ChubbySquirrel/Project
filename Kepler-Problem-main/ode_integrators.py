import numpy as np
#==============================================================
# Fixed stepsize integrator.
# Containing:
#   ode_fixed
#   ode_bvp
#==============================================================
# function [x,y] = ode_ivp(fRHS,fORD,fBVP,x0,y0,x1,nstep)
#
# Solving a system of 
# ordinary differential equations using fixed
# step size.
#
# input:
#   nstep  : number of steps
#   fRHS   : function handle. Needs to return a vector of size(y0).
#   fORD   : function handle. integrator order (step for single update). 
#   fBVP   : unused function handle. For consistency in calls by ode_test.
#   x0     : starting x.
#   y0     : starting y (this is a (nvar,1) vector).
#   x1     : end x.
#
# output:
#   x      : positions of steps (we'll need this for
#            consistency with adaptive step size integrators later)
#   y      : (nvar,nstep+1) maxtrix of resulting y's
#---------------------------------------------------------------

def ode_ivp(fRHS,fORD,fBVP,x0,y0,x1,nstep,**kwargs):

    if (((fRHS.__name__ == 'keplerdirect_symp1') or (fRHS.__name__ == 'keplerdirect_symp2')) and (not (fORD.__name__ == 'euler'))):
        raise Exception('[ode_ivp]: symplectic integrators require fORD = step_euler.')

    stage = 0
    for key in kwargs:          # stage 0: mass lose; stage 1: inter-looper
        if (key=='stage'):
            stage = kwargs[key]

    if (stage==0):
        nvar    = y0.size                      # number of ODEs
        x       = np.linspace(x0,x1,nstep+1)   # generates equal-distant support points
        y       = np.zeros((nvar,nstep+1))     # result array
        y[:,0]  = y0                           # set initial condition
        dx      = x[1]-x[0]                    # step size
        it      = np.zeros(nstep+1)

        print("dx="+str(dx))
        print("Total step="+str(nstep))
        if dx>=1.0:
            print("Dx too large. Proceed?")
            input()
        pass
        #print("[ode_ivp]: k=%5i x=%13.5e y=%13.5e %13.5e %13.5e %13.5e" % (0,x[0],y[0,0],y[1,0],y[2,0],y[3,0]))
        for k in range(1,nstep+1):
            if k%((int)(nstep/100))==0:
                print("", end="\r")
                print("Integrating: "+str((int)(k/nstep*100))+"%", end="")
            t = k*dx
            y[:,k],it[k] = fORD(fRHS,x[k-1],y[:,k-1],dx,t=t,**kwargs)
            #print("[ode_ivp]: k=%5i x=%13.5e y=%13.5e %13.5e %13.5e %13.5e R=%13.5e" % (k,x[k],y[0,k],y[1,k],y[2,k],y[3,k],np.sqrt(y[0,k]*y[0,k]+y[1,k]*y[1,k])))
        pass
        print("")
        return x,y,it
    if (stage==1):
        nvar    = y0.size                      # number of ODEs
        x       = np.linspace(x0,x1,nstep+1)   # generates equal-distant support points
        y       = np.zeros((nvar,nstep+1))     # result array
        y[:,0]  = y0                           # set initial condition
        dx      = x[1]-x[0]                    # step size
        it      = np.zeros(nstep+1)

        print("dx="+str(dx))
        print("Total step="+str(nstep))
        if dx>=1.0:
            print("Dx too large. Proceed?")
            input()
        pass
        #print("[ode_ivp]: k=%5i x=%13.5e y=%13.5e %13.5e %13.5e %13.5e" % (0,x[0],y[0,0],y[1,0],y[2,0],y[3,0]))
        for k in range(1,nstep+1):
            if k%((int)(nstep/100))==0:
                print("", end="\r")
                print("Integrating: "+str((int)(k/nstep*100))+"%", end="")
            t = k*dx
            y[:,k],it[k] = fORD(fRHS,x[k-1],y[:,k-1],dx,t=t,**kwargs)
            #print("[ode_ivp]: k=%5i x=%13.5e y=%13.5e %13.5e %13.5e %13.5e R=%13.5e" % (k,x[k],y[0,k],y[1,k],y[2,k],y[3,k],np.sqrt(y[0,k]*y[0,k]+y[1,k]*y[1,k])))
        pass
        print("")
        return x,y,it
