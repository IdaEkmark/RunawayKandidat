#!/usr/bin/env python3
#
# This example shows how to set up a simple CODE-like runaway
# scenario in DREAM. The simulation uses a constant temperature,
# density and electric field, and generates a runaway current
# through the electric field acceleration, demonstrating Dreicer generation.
#
# Run as
#
#   $ ./basic.py
#   $ ../../build/iface/dreami dream_settings.h5
#
# ###################################################################

#Test för att göra en tids-och radieberoende strömprofil med parametrar passande de i de Vries (tror jag)
#Jag  har ändrat n, T, magnetfält, minor radius, major radius och tMax
#Jag har använt en konstant strömprofil
#Jag har valt att I_p ska variera i tiden precis som i de Vries
#Det har inte fungerat så bra då I_p inte blir som jag angett (trots att E blir bra), samt ökar när jag ökar Np
#Inte heller n_re verkar öka vilket är tråkigt
#f_hot ser inte helt orimlig ut iallafall

#Släng bort f_hot, gör ren fluid-modell
#ds.eqsys.setavalanche se zoom-meddelande
#U/2pi r_stor
#E=V_loop/(2pi (r_0+a/2))

import numpy as np
import sys
from scipy import integrate
import matplotlib.pyplot as plt
from generate_current_profile_radius_time2 import generate_current_profile_fun

sys.path.append('../../py/')

from DREAM.DREAMSettings import DREAMSettings
from DREAM.DREAMOutput import DREAMOutput
from DREAM import runiface
import DREAM.Settings.Equations.DistributionFunction as DistFunc
import DREAM.Settings.Equations.IonSpecies as Ions
import DREAM.Settings.Equations.RunawayElectrons as Runaways
import DREAM.Settings.Solver as Solver
import DREAM.Settings.CollisionHandler as Collisions
import DREAM.Settings.TransportSettings as Transport

#Nu blir det intressant
e=1.60217662e-19
c=299792458
I_re=5e5
A_c=3
n_re0=I_re/(e*c*A_c)#j_re/(e*c)

#Taget från examples/runaway/basic.py
ds = DREAMSettings()
#ds.collisions.collfreq_type = Collisions.COLLFREQ_TYPE_COMPLETELY_SCREENED
ds.collisions.collfreq_type = Collisions.COLLFREQ_TYPE_PARTIALLY_SCREENED

# Grid parameters
pMax = 1    # maximum momentum in units of m_e*c
Np   = 600  # number of momentum grid points
Nxi  = 40   # number of pitch grid points
tMax = 10   # simulation time in seconds
Nt   = 2000   # number of time steps
Nr   = 2    # number of radius points

# Physical parameters
E = 6       # Electric field strength (V/m)
n = 3e18+n_re0    # Electron density (m^-3)
Tt = np.linspace(0,tMax,Nt)
T = 40+760/(Tt+1)**2      # Temperature (eV)
B = 2.4     # Magnetic field
a = np.sqrt(A_c/np.pi)     # Minor radius
r_o = 3     # Major radius

# Set E_field
ds.eqsys.E_field.setPrescribedData(E)

# Set temperature
ds.eqsys.T_cold.setPrescribedData(T, times = Tt)

# Set ions
ds.eqsys.n_i.addIon(name='D', Z=1, iontype=Ions.IONS_PRESCRIBED_FULLY_IONIZED, n=n)

# Runaway electrons
#ds.eqsys.n_re.setAvalanche(avalanche=Runaways.AVALANCHE_MODE_NEGLECT)
ds.eqsys.n_re.setAvalanche(avalanche=Runaways.AVALANCHE_MODE_FLUID_HESSLOW)
ds.eqsys.n_re.setInitialProfile(density=n_re0)

#ds.eqsys.n_re.setDreicer(Runaways.DREICER_RATE_DISABLED)

# Hot-tail grid settings
#ds.hottailgrid.setNxi(Nxi)
#ds.hottailgrid.setNp(Np)
#ds.hottailgrid.setPmax(pMax)
ds.hottailgrid.setEnabled(False)
#ds.hottailgrid.setBiuniformGrid(psep=0.15,npsep_frac=0.5)

# Set initial hot electron Maxwellian
#ds.eqsys.f_hot.setInitialProfiles(n0=n, T0=T)

# Set boundary condition type at pMax
ds.eqsys.f_hot.setBoundaryCondition(DistFunc.BC_PHI_CONST) # extrapolate flux to boundary
ds.eqsys.f_hot.setBoundaryCondition(DistFunc.BC_F_0) # F=0 outside the boundary
ds.eqsys.f_hot.setSynchrotronMode(DistFunc.SYNCHROTRON_MODE_NEGLECT)

# Disable runaway grid
ds.runawaygrid.setEnabled(False)

# Set up radial grid
ds.radialgrid.setB0(B)
ds.radialgrid.setMinorRadius(a)
#ds.radialgrid.setMajorRadius(r_o)
ds.radialgrid.setWallRadius(a)
ds.radialgrid.setNr(Nr)

# Set solver type
ds.solver.setType(Solver.LINEAR_IMPLICIT) # semi-implicit time stepping
ds.solver.preconditioner.setEnabled(False)

# include otherquantities to save to output
ds.other.include('fluid')

# Set time stepper
ds.timestep.setTmax(tMax)
ds.timestep.setNt(Nt)

ds.output.setTiming(stdout=True, file=True)

r=np.linspace(0,a,num=10)
t=np.linspace(0,tMax,num=Nt)
rr, tt = np.meshgrid(r, t)
j=1-(rr/a)**2
I_p=0.9*np.exp(-1/(10*(t+1.697976979769798)))*1e6

#plt.plot(t,I_p)
#plt.xlim(0, tMax)
#plt.ylim(0, 1e6)
#plt.show()

ds_ny, do_ny = generate_current_profile_fun(I_p,j,r,t,ds)

do_ny.eqsys.E_field.plot(r=[0,-1])
#plt.xlim(0, tMax)
#plt.ylim(0, 0.35)
plt.show()
do_ny.eqsys.I_p.plot()
plt.show()
#do_ny.eqsys.f_hot.plot(t=[0,5,10,25,50,-1],r=[0])
#plt.show()

#j_tot = do_ny.eqsys.j_tot[:]
#print(str(j_tot))

m_e=9.10938356e-31
T_J=T*1.602176565e-19 #Temperatur i joule
#v_th=np.sqrt(T_J/m_e)

#xi=np.abs(j_tot/(e*n*v_th))


#tid=np.linspace(0,tMax,num=Nt+1)

#plt.plot(tid,xi[:,0])
#plt.show()

do_ny.eqsys.n_tot.plot(r=[0])
do_ny.eqsys.n_re.plot(r=[0])
#plt.xlim(0.5, tMax)
#plt.ylim(0, 0.35)
plt.show()
