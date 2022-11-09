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

def keplerdirect_symp1(x,y,dx):
    nbodies     = y.size//4 # per body, we have four variables
    par         = globalvar.get_odepar()
    npar        = par.size
    gnewton     = par[0]
    masses      = par[1:npar]
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
    for i in range(nbodies):
        for j in range(0,i):
            ddx           = qx[i]-qx[j]
            ddy           = qy[i]-qy[j]
            R3            = np.power(ddx*ddx+ddy*ddy,1.5)
            pHpq[indx[i]] = pHpq[indx[i]] + masses[j]*ddx/R3
            pHpq[indy[i]] = pHpq[indy[i]] + masses[j]*ddy/R3
        for j in range(i+1,nbodies):
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

    return dydx

#==============================================================
# function dydx = keplerdirect_symp2(x,y,dx)
#
# Calculates partial derivative of Hamiltonian with respect
# to q,p for direct Kepler problem. Provides full update
# for one single symplectic RK2 (Stoermer-Verlet) step.
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

def keplerdirect_symp2(x,y,dx):
    nbodies     = y.size//4 # per body, we have four variables
    par         = globalvar.get_odepar()
    npar        = par.size
    gnewton     = par[0]
    masses      = par[1:npar]
    dydx        = np.zeros(4*nbodies)
    pHpq        = np.zeros(2*nbodies)
    pHpq2       = np.zeros(2*nbodies)
    pHpp        = np.zeros(2*nbodies)
    indx        = 2*np.arange(nbodies)
    indy        = 2*np.arange(nbodies)+1
    indvx       = 2*np.arange(nbodies)+2*nbodies
    indvy       = 2*np.arange(nbodies)+2*nbodies+1
    px          = y[indvx]*masses # this is more cumbersome than necessary,
    py          = y[indvy]*masses # but for consistency with derivation.
    qx          = y[indx]
    qy          = y[indy]
    # first step: constructing p(n+1/2)
    for i in range(nbodies):
        for j in range(0,i):
            ddx           = qx[i]-qx[j]
            ddy           = qy[i]-qy[j]
            R3            = np.power(ddx*ddx+ddy*ddy,1.5)
            pHpq[indx[i]] = pHpq[indx[i]] + masses[j]*ddx/R3
            pHpq[indy[i]] = pHpq[indy[i]] + masses[j]*ddy/R3
        for j in range(i+1,nbodies):
            ddx           = qx[i]-qx[j]
            ddy           = qy[i]-qy[j]
            R3            = np.power(ddx*ddx+ddy*ddy,1.5)
            pHpq[indx[i]] = pHpq[indx[i]] + masses[j]*ddx/R3
            pHpq[indy[i]] = pHpq[indy[i]] + masses[j]*ddy/R3
    pHpq[indx] = pHpq[indx] * gnewton * masses
    pHpq[indy] = pHpq[indy] * gnewton * masses
    px2        = px-0.5*dx*pHpq[indx]
    py2        = py-0.5*dx*pHpq[indy]
    pHpp[indx] = px2/masses
    pHpp[indy] = py2/masses
    qx2        = qx+dx*pHpp[indx]
    qy2        = qy+dx*pHpp[indy]
    for i in range(nbodies):
        for j in range(0,i):
            ddx           = qx2[i]-qx2[j]
            ddy           = qy2[i]-qy2[j]
            R3            = np.power(ddx*ddx+ddy*ddy,1.5)
            pHpq2[indx[i]] = pHpq2[indx[i]] + masses[j]*ddx/R3
            pHpq2[indy[i]] = pHpq2[indy[i]] + masses[j]*ddy/R3
        for j in range(i+1,nbodies):
            ddx           = qx2[i]-qx2[j]
            ddy           = qy2[i]-qy2[j]
            R3            = np.power(ddx*ddx+ddy*ddy,1.5)
            pHpq2[indx[i]] = pHpq2[indx[i]] + masses[j]*ddx/R3
            pHpq2[indy[i]] = pHpq2[indy[i]] + masses[j]*ddy/R3
    pHpq2[indx] = pHpq2[indx] * gnewton * masses
    pHpq2[indy] = pHpq2[indy] * gnewton * masses
    dydx[indx]  = pHpp[indx]
    dydx[indy]  = pHpp[indy]
    dydx[indvx] = -0.5*(pHpq[indx]+pHpq2[indx])/masses
    dydx[indvy] = -0.5*(pHpq[indy]+pHpq2[indy])/masses

    return dydx

