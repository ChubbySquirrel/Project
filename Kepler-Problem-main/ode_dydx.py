import numpy as np
import globalvar

sun_current_mass = 0

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
    stage = 0
    for key in kwargs:          # mass loss equation selector
        if (key=='mlf'):
            mlf = kwargs[key]
        if (key=='t'):
            t = kwargs[key]
        if (key=='inter'):
            stage = kwargs[key]
    if (mlf!=0): masses[0] = remining_mass(mlf, t)
    #=========================================================================
    
    delM = remining_mass(mlf,t)-remining_mass(mlf, t-1)

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
    r0 = 6.957e8 #radius of sun (SI)
    vesc = 618e3 #escape velocity (SI)
    
    #for SI unit conversions--------------------------------------------------
    AU      = 1.495979e11               # AU in meters
    year    = 3.6e3*3.65e2*2.4e1        # Seconds in a year
    gnewton1 = 6.67408e-11
    uLeng   = AU
    uTime   = year
    uVelo   = uLeng/uTime
    uAcce   = uVelo/uTime
    uMass   = uAcce*uLeng*uLeng/gnewton1
    
    
    #-------------------------------------------------------------------------
    #only considering planets past Mars (not in), first space reserved for sun (momentum not affected):
    planet_radius = np.array([0,7.1492e7,6.0268e7,2.4622e7,2.5362e7])
    if (stage==1):
        inter_radius = globalvar.get_inter()
        planet_radius = np.append(planet_radius, inter_radius)
    
    #-------------------------------------------------------------------------
    #SI conversions:
    massesSI = masses*uMass
    #-------------------------------------------------------------------------
    #Particle collisions:
    delM *= -uMass
    
    #mass concentration by collision time:
    Mcol = np.zeros((nbodies))
    for i in range(1,nbodies):
        Mcol[i] = delM/(4*np.pi*(AU*orbital_distance[i])**2)*np.pi*(planet_radius[i])**2
    #mass movement:
    vf = np.zeros((nbodies))
    for i in range (1,nbodies):
        vf[i] = np.sqrt(2*gnewton1*massesSI[0]*(1/(orbital_distance[i]*AU)-1/r0) + vesc**2)
    #------------------------------------------------------------------------
    #converting back to adjusted units:
    Mcol *= 1/uMass
    vf *= 1/uVelo
    delp = Mcol*vf
    delpx = delp
    delpy = delp
    for i in range(1,nbodies):
        delpx[i] *= y[indx[i]]/(orbital_distance[i])
        delpy[i] *= y[indy[i]]/(orbital_distance[i])
    px += delpx
    py += delpy
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
#               1-3: linear portion with respect to M_sol0
#               4: the other equation that
#
#   t       : the absolute time into the integration process
#
# output:
#   mass    : the sun's remaining mass at time t based on
#             mass lose equation i
#--------------------------------------------------------------
def remining_mass(i, t):
    par         = globalvar.get_odepar()
    og_mass     = par[1]
    if (i<4): time_scale = 10**6
    else: time_scale = 6.36*10**9

    if (i==1):
        if (t<time_scale):
            new_mass = og_mass*(1-t*(0.566)/(time_scale))
            return new_mass
        else:
            new_mass = og_mass*(1-0.566)
            return new_mass
    elif (i==2):
        if (t<time_scale):
            new_mass = og_mass*(1-t*(0.459)/(time_scale))
            return new_mass
        else:
            new_mass = og_mass*(1-0.459)
            return new_mass
    elif (i==3):
        if (t<time_scale):
            new_mass = og_mass*(1-t*(0.424)/(time_scale))
            return new_mass
        else:
            new_mass = og_mass*(1-0.424)
            return new_mass
    elif (i==4):
        pass
        #new_mass =
        #return new_mass

    return og_mass
