import numpy as np
import globalvar

#==============================================================
# function dydx = keplerdirect(x,y,dx)
#
# Calculates ODE RHS for Kepler problem via direct summation.
#
# input: 
#   x,y    : position and values for RHS
#            Assumes y to have the shape (x, y, v_x, v_y), 
#            with the cartesian positions (x,y) and their
#            velocities (v_x,v_y).
# global:
#   G      : grav constant
# output:
#   dydx   : vector of results as in y'=f(x,y)
#--------------------------------------------------------------

def keplerdirect(x,y,dx):
    nbodies     = y.size//4 # per body, we have four variables
    par         = globalvar.get_odepar()
    npar        = par.size
    gnewton     = par[0]
    masses      = par[1:npar]
    dydx        = np.zeros(4*nbodies)
    indx        = 2*np.arange(nbodies)
    indy        = 2*np.arange(nbodies)+1
    indvx       = 2*np.arange(nbodies)+2*nbodies
    indvy       = 2*np.arange(nbodies)+2*nbodies+1
    dydx[indx]  = y[indvx] # x'=v
    dydx[indy]  = y[indvy]
    for k in range(nbodies):
        gravx = 0.0
        gravy = 0.0
        for j in range(nbodies):
            if (k != j):
                dx    = y[indx[k]]-y[indx[j]]
                dy    = y[indy[k]]-y[indy[j]]
                R3    = np.power(dx*dx+dy*dy,1.5)
                gravx = gravx - gnewton*masses[j]*dx/R3
                gravy = gravy - gnewton*masses[j]*dy/R3
        dydx[indvx[k]] = gravx
        dydx[indvy[k]] = gravy
    return dydx

#==============================================================
# function dydx = keplerdirect_symp1(x,y,dx)
#
# Calculates partial derivative of Hamiltonian with respect
# to q,p for direct Kepler problem. Provides full update
# for one single symplectic Euler step.
#
# input: 
#   x,y    : position and values for RHS
#            Assumes y to have the shape (x, y, v_x, v_y), 
#            with the cartesian positions (x,y) and their
#            velocities (v_x,v_y).
# global:
#   G(m1*m2): grav constant times masses (in any units) (par[0])
# output:
#   dydx    : vector of results as in y'=f(x,y)
#--------------------------------------------------------------

def keplerdirect_symp1(x,y,dx,**kwargs):
    nbodies     = y.size//4 # per body, we have four variables
    par         = globalvar.get_odepar()
    npar        = par.size
    gnewton     = par[0]
    masses      = par[1:npar].copy()
    #=========================================================================
    t = 0
    mlf         = 0
    for key in kwargs:          # mass loss equation selector
        if (key=='mlf'):
            mlf = kwargs[key]
        if (key=='t'):
            t = kwargs[key]
    if (mlf!=0): masses[0] = remining_mass(mlf, t)
    #=========================================================================
    
    

    indx    = 2*np.arange(nbodies)
    indy    = 2*np.arange(nbodies)+1
    orbital_distance = np.zeros(nbodies)
    for k in range(nbodies):
        orbital_distance[k] = (y[indx[k]]**2+y[indy[k]]**2)**0.5

    dydx        = np.zeros(4*nbodies)
    pHpq        = np.zeros(2*nbodies)
    pHpp        = np.zeros(2*nbodies)
    indx        = 2*np.arange(nbodies)
    indy        = 2*np.arange(nbodies)+1
    indvx       = 2*np.arange(nbodies)+2*nbodies
    indvy       = 2*np.arange(nbodies)+2*nbodies+1
    px          = y[indvx]*masses # this is more cumbersome than necessary,
    py          = y[indvy]*masses # but for consistency with derivation.
    qx          = y[indx]
    qy          = y[indy]
    
    
    #=========================================================================
    #change in momentum from solar mass:
    r0 = 6.957e8 #radius of sun
    vesc = 618e3 #escape velocity
    
    
    
    
    
    #=========================================================================
    for i in range(nbodies):
        for j in range(nbodies):
            if (i!=j):
                ddx           = qx[i]-qx[j]
                ddy           = qy[i]-qy[j]
                R3            = np.power(ddx*ddx+ddy*ddy,1.5)
                pHpq[indx[i]] = pHpq[indx[i]] + masses[j]*ddx/R3
                pHpq[indy[i]] = pHpq[indy[i]] + masses[j]*ddy/R3

    pHpq[indx] = pHpq[indx] * gnewton * masses
    pHpq[indy] = pHpq[indy] * gnewton * masses
    pHpp[indx] = (px-dx*pHpq[indx])/masses
    pHpp[indy] = (py-dx*pHpq[indy])/masses
    dydx[indx] = pHpp[indx]
    dydx[indy] = pHpp[indy]
    dydx[indvx]= -pHpq[indx]/masses
    dydx[indvy]= -pHpq[indy]/masses

    pass

    return dydx


#==============================================================
# function abs_mass_of_sun = remining_mass(i, t)
#
# Calculates the remaining mass of the sun at t time into the
# integration process.
#
# input:
#   i       : the particular mass lose equation to use
#   t       : the absolute time into the integration process
#
# output:
#   mass    : the sun's remaining mass at time t based on
#             mass lose equation i
#--------------------------------------------------------------
def remining_mass(i, t):
    par         = globalvar.get_odepar()
    mass     = par[1].copy()
    time_scale = 10**6

    #print(t)
    if i==1:
        if t<time_scale:
            mass = mass * (1 - t * 0.566 / time_scale)
        else:
            mass = mass * (1 - 0.566)
    elif i==2:
        if t<time_scale:
            mass = mass * (1 - t * 0.459 / time_scale)
        else:
            mass = mass * (1 - 0.459)
    elif i==3:
        if t<time_scale:
            mass = mass * (1 - t * 0.424 / time_scale)
        else:
            mass = mass * (1 - 0.424)
    elif i==4:

        if float(t)/6.36 < 10**9:
            for k in range(t):
                pass
            mass = ((-8.14*10**-14)*(1+0.03*(t/10**9)**2)*mass)
        elif float(t)/7.09 < 10**9:
            mass = 2*(-8.14*10**-14)*(1+0.03*(t/10**9)**2)*mass
        else: pass

    #print(mass)
    return mass
