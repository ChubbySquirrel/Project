#==============================================================
# Test bank for ODE integrators. 
# Containing the functions
#
#   ode_init        : initializes ODE problem, setting functions and initial values
#   ode_check       : performs tests on results (plots, sanity checks)
#   main            : calls the rest. Calling sequence: kepler.py euler 3 
#
#==============================================================
# required libraries
import argparse	                 # allows us to deal with arguments to main()
from argparse import RawTextHelpFormatter
import numpy as np               # numerical routines (arrays, math functions etc)
import math
import matplotlib.pyplot as plt  # plotting commands
import globalvar                 # interface for global variables

import ode_integrators as odeint # contains the drivers for ODE integration
import ode_step as step          # the stepper functions
import ode_dydx as dydx          # contains the RHS functions for selected problems.
import ode_bvp  as bvp           # contains the boundary value functions for selected problems.
import ode_jac  as jac           # contains definitions of Jacobians for various problems
import gc

#==============================================================
# functions
#==============================================================
# function mass,eps,rap,vorb,torb = get_planetdata(which)
#
# Returns planetary orbit data
#
# input:
#   which: integer array with elements between 1 and 8, with 1: Mercury...8: Neptune
# output:
#   mass: planet mass in kg
#   eps : eccentricity
#   rap : aphelion distance (in km)
#   vorb: aphelion velocity (in km/s)
#   torb: orbital period (in years)
#---------------------------------------------------------------

def get_planetdata(which):
    nplanets             = len(which)
    mass                 = np.array([1.989e30,3.3011e23,4.8675e24,5.972e24,6.41e23,1.89819e27,5.6834e26,8.6813e25,1.02413e26])
    eps                  = np.array([0.0,0.205,0.0067,0.0167,0.0934,0.0489,0.0565,0.0457,0.0113])
    rap                  = np.array([0.0,6.9816e10,1.0894e11,1.52139e11,2.49232432e11,8.1662e11,1.5145e12,3.00362e12,4.54567e12])
    vorb                 = np.array([0.0,3.87e4,3.479e4,2.929e4,2.197e4,1.244e4,9.09e3,6.49e3,5.37e3])
    yrorb                = np.array([0.0,0.241,0.615,1.0,1.881,1.1857e1,2.9424e1,8.3749e1,1.6373e2])
    rmass                = np.zeros(nplanets+1)
    reps                 = np.zeros(nplanets+1)
    rrap                 = np.zeros(nplanets+1)
    rvorb                = np.zeros(nplanets+1)
    ryrorb               = np.zeros(nplanets+1)
    rmass [0]            = mass [0]
    rmass [1:nplanets+1] = mass [which]
    reps  [1:nplanets+1] = eps  [which]
    rrap  [1:nplanets+1] = rap  [which]
    rvorb [1:nplanets+1] = vorb [which]
    ryrorb[1:nplanets+1] = yrorb[which]
    return rmass,reps,rrap,rvorb,ryrorb

def get_interdata(which):
    # halley, moon, sun, Nibiru, nothing
    nplanets             = 1
    mass                 = np.array([2.2e14, 7.35e22, 1.989e30, 1.31e22*5000, 1e-20])
    eps                  = np.array([0, 0, 0, 0, 0])
    # initial setup: 2e13:11au; 3e13:25au; 5e13:75au;
    # after setup:   2e13:17au; 3e13:40au; 5e13:130au;
    rap                  = np.array([5e13, 5e13, 5e13, 3e13, 1e20])  # distance
    vorb                 = np.array([8e2, 8e2, 8e2, 8e2, 1e20])  # initial push
    yrorb                = np.array([1.6373e2, 1.6373e2, 1.6373e2, 1.6373e2, 1.6373e2])
    radius               = np.array([5.5e3, 1.7347e3, 6.957e8, 2e6, 1e1])
    rmass                = np.zeros(nplanets)
    reps                 = np.zeros(nplanets)
    rrap                 = np.zeros(nplanets)
    rvorb                = np.zeros(nplanets)
    ryrorb               = np.zeros(nplanets)
    rmass [0:nplanets] = mass [which]
    reps  [0:nplanets] = eps  [which]
    rrap  [0:nplanets] = rap  [which]
    rvorb [0:nplanets] = vorb [which]
    ryrorb[0:nplanets] = yrorb[which]
    globalvar.set_inter(radius[which])
    return rmass,reps,rrap,rvorb,ryrorb

#==============================================================
# function fRHS,x0,y0,x1 = ode_init(iprob,stepper)
#
# Initializes derivative function, parameters, and initial conditions
# for ODE integration.
#
# input: 
#   iprob:   problem parameter
#   stepper: euler
#            rk2
#            rk4
#            rk45
# output:
#   fINT   : function handle for integrator (problem) type: initial or boundary value problem (ode_ivp or ode_bvp)
#   fORD   : function handle for integrator order (euler, rk2, rk4, rk45). 
#            Note: symplectic integrators require euler.
#   fRHS   : function handle for RHS of ODE. Needs to return vector dydx of same size as y0.
#   fBVP   : function handle for boundary values in case fINT == ode_bvp.
#   fJAC   : functino handle for Jacobian required for implicit integration. Default = None
#   x0     : starting x
#   y0     : starting y(x0)
#   x1     : end x
#--------------------------------------------------------------

def ode_init(stepper,planet,usesymp):

    fBVP = None # default: IVP, but see below.
    fJAC = None # default: explicit integrators (don't need Jacobian)
    eps  = None  # default: fixed stepsize integrators don't need eps
    if (usesymp==1):
        stepper = 'euler'
        fRHS    = dydx.keplerdirect_symp1
    elif (usesymp==2):
        stepper = 'euler'
        fRHS    = dydx.keplerdirect_symp2
    else:
        fRHS    = dydx.keplerdirect
    if (stepper == 'euler'):
        fORD = step.euler
    elif (stepper == 'rk2'):
        fORD = step.rk2
    elif (stepper == 'rk4'):
        fORD = step.rk4
    elif (stepper == 'rk45'):
        fORD = step.rk45
    elif (stepper == 'rk5'):
        fORD = step.rk5
    elif (stepper == 'eulersi'):
        fORD = step.eulersi
    elif (stepper == 'kr4si'):
        fORD = step.kr4si
    elif (stepper == 'rb34si'):
        fORD = step.rb34si
    else:
        raise Exception('[ode_init]: invalid stepper value: %s' % (stepper))

    print('[ode_init]: initializing Kepler problem')
    # We set the initial positions, assuming orbit starts at aphel.
    # Units are different here. We set G=1, L=1AU, t=1yr. This results
    # a set scale for the mass, as below.
    AU      = 1.495979e11               # AU in meters
    year    = 3.6e3*3.65e2*2.4e1        # year in seconds
    if (planet != "all"):
        mass,eps,r_aphel,v_orb,yr_orb = get_planetdata(np.array([int(planet)]))
    elif (planet == "all"):
        mass,eps,r_aphel,v_orb,yr_orb = get_planetdata(np.array([5,6,7,8]))
        # mass,eps,r_aphel,v_orb,yr_orb = get_planetdata(np.array([1,2,3,4,5,6,7,8]))
    else:
        raise Exception('[ode_init]: invalid planet: %' % (planet))

    gnewton = 6.67408e-11
    uLeng   = AU
    uTime   = year
    uVelo   = uLeng/uTime
    uAcce   = uVelo/uTime
    uMass   = uAcce*uLeng*uLeng/gnewton # ???
    masscu  = mass/uMass
    rapcu   = r_aphel/uLeng
    velcu   = v_orb/uVelo
    # Set initial conditions. All objects are aligned along x-axis, with planets to positive x, sun to negative x.
    rapcu[0]= -np.sum(masscu*rapcu)/masscu[0]
    velcu[0]= -np.sum(masscu*velcu)/masscu[0]

    nstepyr = stpyr                          # number of steps per year
    nyears  = int(np.ceil(np.max(yr_orb)))
    x0      = 0.0                          # starting at t=0
    #x1      = nyears*year/uTime            # end time in years
    x1      = total_years
    nstep   = nyears*nstepyr               # thus, each year is resolved by nstepyr integration steps
    nbodies = mass.size                    # number of objects
    y0      = np.zeros(4*nbodies)
    par     = np.zeros(nbodies+1)          # number of parameters
    par[0]  = 1.0
    for k in range(nbodies):               # fill initial condition array and parameter array
        y0[2*k]             = rapcu[k]
        y0[2*k+1]           = 0.0
        y0[2*(nbodies+k)]   = 0.0
        y0[2*(nbodies+k)+1] = velcu[k]
        par[k+1]            = masscu[k]
    fINT    = odeint.ode_ivp
    eps     = 1e-8


    globalvar.set_odepar(par)
    return fINT,fORD,fRHS,fBVP,fJAC,x0,y0,x1,nstep,eps

#==============================================================
# function ode_check(x,y)
#
# Performs problem-dependent tests on results.
#
# input: 
#   iinteg   : integrator type
#   x    :  independent variable
#   y    :  integration result
#   it   :  number of substeps used. Only meaningful for RK45 (iinteg = 4). 
#--------------------------------------------------------------

def ode_check(x,y,it):
    # limit datapoints so that they are plottable
    max_datapoint = 10**5
    if y.shape[1]>max_datapoint:
        factor = y.shape[1]/max_datapoint
        x = x[::int(np.ceil(factor))]
        y = y[:, ::int(np.ceil(factor))]
        it = it[::int(np.ceil(factor))]


    # for the direct Kepler problem, we check for energy and angular momentum conservation,
    # and for the center-of-mass position and velocity
    color   = ['black','green','cyan','blue','red','black','black','black','black']
    n       = x.size
    par     = globalvar.get_odepar()
    npar    = par.size
    nbodies = par.size-1
    gnewton = par[0]
    masses  = par[1:npar]
    Egrav   = np.zeros(n)
    indx    = 2*np.arange(nbodies)
    indy    = 2*np.arange(nbodies)+1
    indvx   = 2*np.arange(nbodies)+2*nbodies
    indvy   = 2*np.arange(nbodies)+2*nbodies+1
    E       = np.zeros(n) # total energy
    Lphi    = np.zeros(n) # angular momentum
    R       = np.sqrt(np.power(y[indx[0],:]-y[indx[1],:],2)+np.power(y[indy[0],:]-y[indy[1],:],2))
    Rs      = np.zeros(n) # center of mass position
    vs      = np.zeros(n) # center of mass velocity

    print("Ode_check running:")
    for k in range(n):
        if k%(int(n/100))==0:
            print("", end="\r")
            print("Part 1: "+str(int(k/n*100))+"%", end="")

        E[k]    = 0.5*np.sum(masses*(np.power(y[indvx,k],2)+np.power(y[indvy,k],2)))
        Lphi[k] = np.sum(masses*(y[indx,k]*y[indvy,k]-y[indy,k]*y[indvx,k]))
        Rsx     = np.sum(masses*y[indx,k])/np.sum(masses)
        Rsy     = np.sum(masses*y[indy,k])/np.sum(masses)
        vsx     = np.sum(masses*y[indvx,k])/np.sum(masses)
        vsy     = np.sum(masses*y[indvy,k])/np.sum(masses)
        Rs[k]   = np.sqrt(Rsx*Rsx+Rsy*Rsy)
        vs[k]   = np.sqrt(vsx*vsx+vsy*vsy)
    print("")
    for j in range(nbodies):
        print("", end="\r")
        print("Part 2: "+str(j)+" out of "+str(nbodies), end="")

        for i in range(j): # preventing double summation. Still O(N^2) though.
            dx    = y[indx[j],:]-y[indx[i],:]
            dy    = y[indy[j],:]-y[indy[i],:]
            Rt    = np.sqrt(dx*dx+dy*dy)
            Egrav = Egrav - gnewton*masses[i]*masses[j]/Rt
    print("")
    E       = E + Egrav
    E       = E/E[0]
    Lphi    = Lphi/Lphi[0]
    #for k in range(n): print('k=%7i t=%13.5e E/E0=%20.12e L/L0=%20.12e Rs=%10.2e vs=%10.2e' % (k,x[k],E[k],Lphi[k],Rs[k],vs[k]))
    Eplot   = E-1.0
    Lplot   = Lphi-1.0

    # now plot everything
    # (1) the orbits
    xmin    = np.min(y[indx,:])
    xmax    = np.max(y[indx,:])
    ymin    = np.min(y[indy,:])
    ymax    = np.max(y[indy,:])

    #plt.figure(num=1,figsize=(8,8),dpi=100,facecolor='white')
    fig1,ax1 = plt.subplots(1,1)

    #if (1.05*xmin<-100 or 1.05*xmax>100): ax1.set_xlim(-100, 100)
    #else: ax1.set_xlim(1.05*xmin,1.05*xmax)

    #if (1.05*ymin<-100 or 1.05*ymax>100): ax1.set_ylim(-100,100)
    #else: ax1.set_ylim(1.05*ymin,1.05*ymax)

    up = 10
    for k in range(nbodies):
        print("", end="\r")
        print("Part 3: "+str(k)+" out of "+str(nbodies), end="")
        ax1.plot(y[indx[k],:],y[indy[k],:],color=color[k],linewidth=1.0,linestyle='-')
        up = np.max([up, np.max([y[indx[k],:], y[indy[k],:]])])

    up = np.min([500.0, up])+5
    ax1.set_xlim([-up, up])
    ax1.set_ylim([-up, up])

    ax1.set_aspect('equal')
    ax1.set_xlabel('x [AU]')
    ax1.set_ylabel('y [AU]')
    # (2) the checks (total energy and angular momentum)
    #plt.figure(num=2,figsize=(8,8),dpi=100,facecolor='white')
    fig2,(ax2,ax3) = plt.subplots(2,1)
    ax2.plot(x,Eplot,linestyle='-',color='black',linewidth=1.0)
    ax2.set_xlabel('t [yr]')
    ax2.set_ylabel('$\Delta$E/E')
    #plt.legend()
    #ax3.subplot(212)
    ax3.plot(x,Lplot,linestyle='-',color='black',linewidth=1.0)
    ax3.set_xlabel('t [yr]')
    ax3.set_ylabel('$\Delta$L/L')

    # print the distance over time
    fig3,(ax31) = plt.subplots(1,1)
    ss = y.shape
    up = 0.0
    for k in range(nbodies):
        print("", end="\r")
        print("Orbital distance: "+str(k)+" out of "+str(nbodies), end="\n")

        orbital_distance = np.zeros(ss[1])
        for j in range(ss[1]):
            if j%(int(ss[1]/100))==0:
                print("", end="\r")
                print("Progress: "+str(int(j/ss[1]*100))+"%", end="")

            orbital_distance[j] = (y[indx[k],j]**2+y[indy[k],j]**2)**0.5
        ax31.plot(x, orbital_distance, linestyle='-',color=color[k],linewidth=1.0)
        up = np.max([up, np.max(orbital_distance)])

    ax31.set_ylim(-1.0, np.min([300.0, up])+3)

    plt.tight_layout()
    plt.show()


#==============================================================

# Init the inter-looper, with index being the selector
def inter_init(index):
    fORD = step.euler
    fRHS    = dydx.keplerdirect_symp1
    fBVP = None
    fJAC = None

    print('\n[Post inter-looper]: initializing inter-looper problem')
    # We set the initial positions, assuming orbit starts at aphel.
    # Units are different here. We set G=1, L=1AU, t=1yr. This results
    # a set scale for the mass, as below.
    AU      = 1.495979e11               # AU in meters
    year    = 3.6e3*3.65e2*2.4e1        # year in seconds
    mass,eps,r_aphel,v_orb,yr_orb = get_interdata(index)

    gnewton = 6.67408e-11
    uLeng   = AU
    uTime   = year
    uVelo   = uLeng/uTime
    uAcce   = uVelo/uTime
    uMass   = uAcce*uLeng*uLeng/gnewton # ???
    masscu  = mass/uMass
    rapcu   = r_aphel/uLeng
    velcu   = v_orb/uVelo
    # Set initial conditions. All objects are aligned along x-axis, with planets to positive x, sun to negative x.
    rapcu[0]= -np.sum(masscu*rapcu)/masscu[0]
    velcu[0]= -np.sum(masscu*velcu)/masscu[0]

    nstepyr = 10**3                          # number of steps per year
    nyears  = int(np.ceil(np.max(yr_orb)))
    x0      = 0.0                          # starting at t=0
    #x1      = nyears*year/uTime            # end time in years
    x1      = 10**4
    nstep   = nyears*nstepyr               # thus, each year is resolved by nstepyr integration steps
    nbodies = mass.size                    # number of objects
    y0      = np.zeros(4*nbodies)
    par     = np.zeros(nbodies)          # number of parameters
    for k in range(nbodies):               # fill initial condition array and parameter array
        y0[2*k]             = rapcu[k]
        y0[2*k+1]           = 0.0
        y0[2*(nbodies+k)]   = 0.0
        y0[2*(nbodies+k)+1] = velcu[k]
        par[k]            = masscu[k]
    fINT    = odeint.ode_ivp
    eps     = 1e-8

    oldpar = globalvar.get_odepar()
    oldpar[1] = dydx.remining_mass(mlf, total_years)
    oldpar = np.append(oldpar, par)

    globalvar.set_odepar(oldpar)
    return fINT,fORD,fRHS,fBVP,fJAC,x0,y0,x1,nstep,eps


#==============================================================
# main
# 
def main():

    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument("stepper",type=str,default='euler',
                        help="stepping function:\n"
                             "  explicit updates:\n"
                             "    euler   : Euler step\n"
                             "    rk2     : Runge-Kutta 2nd order\n"
                             "    rk4     : Runge-Kutta 4th order\n"
                             "    rk5     : Runge-Kutta 5th order\n"
                             "    rk45    : RK-Fehlberg\n"
                             "  semi-implicit updates:\n"
                             "    eulersi : Euler step\n"
                             "    kr4si   : Kaps-Rentrop 4th order\n"
                             "    rb34si  : Rosenbrock 4th order\n")
    parser.add_argument("planet",type=str,default='3',
                        help="planet:\n"
                             "  all       : all planets\n"
                             "  [1,...8]  : single planet\n")
    parser.add_argument("--symp",type=int,default=0,
                        help="use symplectic integrator\n")
    parser.add_argument("--mlf",type=int,default=0,
                        help="what mass lose function to use\n")


    args    = parser.parse_args()
    stepper = args.stepper
    planet  = args.planet
    usesymp = args.symp
    global mlf
    mlf     = args.mlf  # mass lose equation selection

    fINT,fORD,fRHS,fBVP,fJAC,x0,y0,x1,nstep,eps = ode_init(stepper,planet,usesymp)
    x,y,it                                      = fINT(fRHS,fORD,fBVP,x0,y0,x1,nstep,fJAC=fJAC,eps=eps,mlf=mlf,stage=0)

    ode_check(x,y,it)

    # inter-looper after the process
    end_y = y[:, -1]
    del y
    gc.collect()
    y = end_y
    fINTI,fORDI,fRHSI,fBVPI,fJACI,x0I,y0I,x1I,nstepI,epsI = inter_init(interlooper)
    neY = np.zeros(int(y.size+y0I.size))
    for k in range(int(y.size/2+1)):
        if k==int(y.size/2):
            neY[k] = y0I[0]
            neY[k+1] = y0I[1]
            neY[int(neY.size/2+k)] = y0I[2]
            neY[int(neY.size/2+k+1)] = y0I[3]
            break
        neY[k] = y[k]
        neY[int(neY.size/2+k)] = y[int(y.size/2+k)]


    xI, yI, itI = fINT(fRHS,fORD,fBVP,x0I,neY,x1I,nstepI,fJAC=fJACI,eps=epsI,mlf=0,stage=1)
    ode_check(xI, yI, itI)

#==============================================================

stpyr = 10**5
total_years = 10**7
interlooper = 3
main()


