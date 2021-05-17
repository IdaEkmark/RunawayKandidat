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

# Test för att reproducera de Vries med parametrar passande de i de Vries (tror jag)

# Jag  har beräknat E(t) på samma sätt som de Vries gör (från I_tot och V_loop) samt satt
# T_cold och n_cold till en funktion lik den data de Vries använder. Stämmer det att
# T_cold är termala temperaturen och n_cold är termala elektron-densiteten?

# Jag får inte samma I_p som de Vries I_tot utan den ökar mer och mer (efter 10s är den
# ca 3.5e5 ist för 9e5), och I_re beräknat som de Vries gör är alltid större än I_p

# Skenande-elektron-densiteten ökar också mer och mer och blir större än total-elektron-
# densiteten vilket inte är super - det vore bra med något som skulle kunna hindra
# n_re > n_tot

# Allt tar mycket längre tid än för de Vries

import numpy as np
import sys
from scipy import integrate
import matplotlib.pyplot as plt
from generate_current_profile import generate_current_profile_fun

sys.path.append('../../py/')

from DREAM.DREAMSettings import DREAMSettings
from DREAM.DREAMOutput import DREAMOutput
from DREAM import runiface
import DREAM.Settings.Equations.DistributionFunction as DistFunc
import DREAM.Settings.Equations.IonSpecies as Ions
import DREAM.Settings.Equations.RunawayElectrons as Runaways
import DREAM.Settings.Solver as Solver
import DREAM.Settings.CollisionHandler as Collisions
#import DREAM.Settings.Equations.ElectricField as ElectricField
import DREAM.Settings.TransportSettings as Transport

#### Physical parameters ####

# Universal constants
e   = 1.60217662e-19  # Elementary charge [C]
c   = 299792458       # Speed of light [m/s]
m_e = 9.10938356e-31  # Electron mass [kg]

# Assumed constants in de Vries
A_c   = 3     # Plasma cross section [m^2]
L_tor = 5e-6  # Inductance [H]
r_0   = 3     # Major radius [m]
a     = 0.8   # Minor radius [m]
B     = 2.4   # Magnetic field [T]
A0    = 0     # Advection [???]
D0    = 0.03  # Diffusion [m^2/s ?]

# Time and radial parameters
tMax = 10     # Simulation time [s]
Nt   = 10000  # Number of time steps
Nr   = 1      # Number of radial steps

# Experimental data from de Vries (approximated)
t      = np.linspace(0,tMax,num=Nt+1)        # Time vector for time depending data
I_p    = 0.9*np.exp(-1/(10*(t+0.8490)))*1e6  # Time dependent plasma current [A]
dI_p   = np.gradient(I_p,t)                  # dI_p/dt, time derivative of the plasma current [A/s]
V_loop = 0.08+0.77/(t+1)**2.5                # Time dependent external torodial loop voltage [V]
n_e    = (0.285*np.exp(-t**2)+0.015)*1e19    # Time dependent density of thermal electrons [m^-3]
T_e    = (0.040+0.760/(t+1)**2)*1e3          # Time dependent thermal temperature [eV]

# Initial values in de Vries, changeable
I_re  = 5.5e5           # Initial runaway current [A]
n_re0 = I_re/(e*c*A_c)  # Initial density of runaway electrons [m^-3]

# Electric field calculated as in de Vries article
V_res=V_loop-L_tor*dI_p  # Time dependent resistive voltage [V] (V_res=I_tot*R_tot)
E = V_res/(2*np.pi*r_0)  # Time dependent electric field strength [V/m]

# Total electron density
n = n_e[0]+n_re0    # Electron density [m^-3]


#### DREAM settings ####

ds = DREAMSettings()
#ds.collisions.collfreq_type = Collisions.COLLFREQ_TYPE_COMPLETELY_SCREENED
ds.collisions.collfreq_type = Collisions.COLLFREQ_TYPE_PARTIALLY_SCREENED

# Set E_field
ds.eqsys.E_field.setPrescribedData(E, times=t)

# Set thermal temperature and thermal electron density
ds.eqsys.T_cold.setPrescribedData(T_e, times=t)
ds.eqsys.n_cold.setPrescribedData(density=n_e, times=t)

# Set ions
ds.eqsys.n_i.addIon(name='D', Z=1, iontype=Ions.IONS_PRESCRIBED_FULLY_IONIZED, n=n)

# Runaway electrons
#ds.eqsys.n_re.setAvalanche(avalanche=Runaways.AVALANCHE_MODE_NEGLECT)
ds.eqsys.n_re.setAvalanche(avalanche=Runaways.AVALANCHE_MODE_FLUID_HESSLOW)
ds.eqsys.n_re.setInitialProfile(density=n_re0)

# Runaway loss
t = np.linspace(tMax/2, tMax/2, 1)
r = np.linspace(0, a, Nr+1)

A  = A0 * np.ones((1, Nr+1))  # istället för 'np.array([...])'
D  = D0 * np.ones((1, Nr+1))  # istället för 'np.array([...])'

ds.eqsys.n_re.transport.prescribeAdvection(ar=A, t=t, r=r)
ds.eqsys.n_re.transport.prescribeDiffusion(drr=D, t=t, r=r)
ds.eqsys.n_re.transport.setBoundaryCondition(Transport.BC_F_0)

ds.eqsys.n_re.setDreicer(Runaways.DREICER_RATE_DISABLED)

# Hot-tail grid settings
ds.hottailgrid.setEnabled(False)

# Disable runaway grid
ds.runawaygrid.setEnabled(False)

# Set up radial grid
ds.radialgrid.setB0(B)
ds.radialgrid.setMinorRadius(a)
ds.radialgrid.setMajorRadius(r_0)
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

# Save and run DREAM settings
ds.save('settings_de_Vries_test_2.h5')
do = runiface(ds, 'output_de_Vries_test_2.h5', quiet=False)


#### Manage result ####

# Plot E_field
do.eqsys.E_field.plot(r=[0,-1])
plt.show()

# Plot I_p and I_re
n_re = do.eqsys.n_re[:]
I_re = n_re*e*c*A_c
tid=np.linspace(0,tMax,num=len(I_re))
plt.plot(tid,I_re)
do.eqsys.I_p.plot()
plt.show()

# Plot n_tot and n_re
#do.eqsys.n_tot.plot(r=[0,-1])
#do.eqsys.n_re.plot(r=[0,-1])
#plt.show()


# Plot streaming parameter according to de Vries
#I_p = do.eqsys.I_p[:]
#T_J=T_e*1.602176565e-19 #Temperatur i joule
#v_th=np.sqrt(T_J/m_e)
#j_tot=I_p/A_c
#xi=np.abs(j_tot/(e*n*v_th))
#tid=np.linspace(0,tMax,num=len(xi))
#plt.plot(tid,xi)
#plt.show()


###################################################################################################
# Code for Troubleshooting
###################################################################################################

#dI_p_an    = 0.9*np.exp(-1/(10*(t+0.8490)))*(1/(10*(t+0.8490)**2))#*1e6

#plt.plot(t,I_p)
#plt.plot(t,dI_p)
#plt.xlim(-1.5, 8.5)
#plt.ylim(0, 1)
#plt.show()
#plt.plot(t,V_loop)
#plt.plot(t,n_e)
#plt.plot(t,T_e)
#plt.xlim(-1.5, 8.5)
#plt.ylim(0, 1)
#plt.show()
#print(str(I_p))

#E_c=7.66e-22*n_e

#plt.plot(t,E)
#plt.plot(t,E_c)
#plt.show()

#lnLambda=np.linspace(11,16,num=6) #-16
#for ln in lnLambda:
#    dI_re = 3*1e18*np.sqrt(np.pi / 21) * 1 / ln * (E - E_c) - 1 / 18
#    plt.plot(t, dI_re)
#    plt.show()

