
import numpy as np
import sys
from scipy import integrate
import matplotlib.pyplot as plt


sys.path.append('../../py/')

from DREAM.DREAMSettings import DREAMSettings
from DREAM import runiface
import DREAM.Settings.Equations.IonSpecies as Ions
import DREAM.Settings.Equations.RunawayElectrons as Runaways
import DREAM.Settings.Solver as Solver
import DREAM.Settings.CollisionHandler as Collisions
import DREAM.Settings.TransportSettings as Transport
import DREAM.Settings.Equations.ColdElectrons as ColdElectrons
import DREAM.Settings.Equations.ColdElectronTemperature as ColdElectronTemperature
import DREAM.Settings.Equations.ElectricField as ElectricField
import DREAM.Settings.Equations.IonSpecies as Ions

#### Physical parameters ####

# Universal constants
e   = 1.60217662e-19  # Elementary charge [C]
c   = 299792458       # Speed of light [m/s]
m_e = 9.10938356e-31  # Electron mass [kg]

# Assumed constants in de Vries
#A_c    = 3                   # Plasma cross section [m^2]
#L_tor  = 5e-6                # Inductance [H]
r_0    = 6.2                 # Major radius [m]
a      = 2                   # Minor radius [m]
#tau_RE = 18                  # Confinement time [s]
#A0     = 0                   # Advection [???]
B      = 5.3                 # Magnetic field [T]

# Time and radial parameters
tMax_c = 1e-3                           # Simulation time [s]
Nt_c   = 10000                         # Number of time steps
tMax = 1e-3                           # Simulation time [s]
Nt   = 10000                         # Number of time steps
Nr   = 1                             # Number of radial steps
t    = np.linspace(0,tMax,num=Nt+1)  # Time vector for time depending data


#Ions
Z_D = 1
Z_B = 4
a_D = 0.9  # Proportion of ions that are deuterium
a_B = 1 - a_D  # Proportion of ions that are beryllium
n_tot = 1.01e20  # Total ion density
n_D = a_D * n_tot  # Deuterium density
n_B = a_B * n_tot

V_loop_wall = 1
E_initial = 0 #V/m
T_initial = 50 #eV

########################################################################################################################
                                                    #SIMULATION ONE#
########################################################################################################################

ds_c = DREAMSettings()
ds_c.collisions.collfreq_type = Collisions.COLLFREQ_TYPE_PARTIALLY_SCREENED #Är detta rätt?

ds_c.eqsys.E_field.setType(ElectricField.TYPE_SELFCONSISTENT)
ds_c.eqsys.E_field.setInitialProfile(efield=E_initial)
ds_c.eqsys.E_field.setBoundaryCondition(ElectricField.BC_TYPE_PRESCRIBED, V_loop_wall=V_loop_wall, times=t)

ds_c.eqsys.T_cold.setPrescribedData(T_initial, times=t)
ds_c.eqsys.n_cold.setType(ColdElectrons.TYPE_SELFCONSISTENT)

ds_c.eqsys.n_i.setIonization(Ions.IONIZATION_MODE_FLUID)
ds_c.eqsys.n_i.addIon(name='D', Z=Z_D, iontype=Ions.IONS_DYNAMIC_FULLY_IONIZED, n=n_D)
ds_c.eqsys.n_i.addIon(name='B', Z=Z_B, iontype=Ions.IONS_DYNAMIC_FULLY_IONIZED, n=n_B)

ds_c.eqsys.n_re.setAvalanche(avalanche=Runaways.AVALANCHE_MODE_FLUID_HESSLOW)

ds_c.eqsys.n_re.setDreicer(Runaways.DREICER_RATE_NEURAL_NETWORK)

ds_c.hottailgrid.setEnabled(False)
ds_c.runawaygrid.setEnabled(False)

ds_c.radialgrid.setB0(B)#, times=t)
ds_c.radialgrid.setMinorRadius(a)
ds_c.radialgrid.setMajorRadius(r_0)
ds_c.radialgrid.setWallRadius(a)
ds_c.radialgrid.setNr(Nr)

ds_c.solver.setType(Solver.NONLINEAR)

ds_c.timestep.setTmax(tMax_c)
ds_c.timestep.setNt(Nt_c)

ds_c.other.include('fluid')
ds_c.save('settings_SELFCONSISTENT.h5')
do_c = runiface(ds_c, 'output_SELFCONSISTENT.h5', quiet=False)
#ds_c.eqsys.n_re.transport.prescribeAdvection(ar=A, t=t_D, r=r_D)
#ds_c.eqsys.n_re.transport.prescribeDiffusion(drr=D, t=t_D, r=r_D)
#ds_c.eqsys.n_re.transport.setBoundaryCondition(Transport.BC_F_0)

########################################################################################################################
                                                    #SIMULATION TWO#
########################################################################################################################

#ds.eqsys.n_i.addIon(name='D', Z=Z_D, iontype=Ions.IONS_DYNAMIC_FULLY_IONIZED, n=n_D)
#ds.eqsys.n_i.addIon(name='B', Z=Z_B, iontype=Ions.IONS_DYNAMIC_FULLY_IONIZED, n=n_B)

ds = DREAMSettings(ds_c)

ds.fromOutput('output_SELFCONSISTENT.h5')#, ignore=['E_field','T_cold','n_cold','n_re'])
ds.save('SELFCONSISTENT_SETTINGS2.h5')

ds.eqsys.T_cold.setType(ColdElectronTemperature.TYPE_SELFCONSISTENT)
ds.eqsys.T_cold.setInitialProfile(T_initial)
ds.eqsys.n_cold.setType(ColdElectrons.TYPE_SELFCONSISTENT)

ds_c.timestep.setTmax(tMax)
ds_c.timestep.setNt(Nt)

do = runiface(ds, 'output_SELFCONSISTENT2.h5', quiet=False)

########################################################################################################################
                                                    #PLOTS#
########################################################################################################################
do_c.eqsys.E_field.plot()
do.eqsys.E_field.plot()
E=do_c.eqsys.E_field[:]
C=np.linspace(E[-1],E[-1],num=len(E))
t=np.linspace(0,tMax,num=len(E))
plt.plot(t,C)

plt.show()
