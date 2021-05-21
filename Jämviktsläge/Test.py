import numpy as np
import sys
#from scipy import integrate
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

sys.path.append('../../py/')

from DREAM.DREAMSettings import DREAMSettings
from DREAM.DREAMOutput import DREAMOutput
from DREAM import runiface
import DREAM.Settings.Equations.DistributionFunction as DistFunc
import DREAM.Settings.Equations.IonSpecies as Ions
import DREAM.Settings.Equations.RunawayElectrons as Runaways
import DREAM.Settings.Solver as Solver
import DREAM.Settings.CollisionHandler as Collisions

def generate_current_profile_fun(I_p,current_profile,radie,DREAM_settings_object):
    #Skapar DREAM-settings och output för att få conductivity
    ds_conductivity = DREAMSettings(DREAM_settings_object, chain=False)
    ds_conductivity.eqsys.E_field.setPrescribedData(0)
    ds_conductivity.timestep.setNt(1)
    ds_conductivity.save('settings_conductivity_run.h5')
    do_conductivity = runiface(ds_conductivity, 'output_conductivity_run.h5', quiet=False)
    # Simulerar med de DREAMsettings som vi genererat och skapar ett output-objekt för att finna konduktivitet

    #Bestämmer E-fält som krävs för att få önskad strömprofil
    conductivity = do_conductivity.other.fluid.conductivity[-1,:]
    VpVol = do_conductivity.grid.VpVol[:]
    dr = do_conductivity.grid.dr
    r = do_conductivity.grid.r
    #r_f = do_conductivity.grid.r_f
    if len(radie) < 4:
        inter_cub_current_profile = interp1d(radie, current_profile, kind='linear')
    else:
        inter_cub_current_profile = interp1d(radie, current_profile, kind='cubic')
    j_prof_inter = inter_cub_current_profile(r)
    current_profile_integral = np.sum(j_prof_inter*VpVol*dr)
    j_0 = 2 * np.pi * I_p / current_profile_integral
    E_r = j_0 * j_prof_inter / conductivity
    r_0=3
    V_loop = E_r*r_0*2*np.pi

    #Skapar DREAM-settings och output för att få önskad strömprofil
    ds_fun = DREAMSettings(DREAM_settings_object,chain=False)
    ds_fun.eqsys.E_field.setPrescribedData(E_r, radius=r)
    ds_fun.save('settings_generate_current_profile_fun.h5')
    do_fun = runiface(ds_fun, 'output_generate_current_profile_fun.h5', quiet=False)

    return ds_fun, do_fun, V_loop
'''
e   = 1.60217662e-19  # Elementary charge [C]
c   = 299792458       # Speed of light [m/s]
m_e = 9.10938356e-31  # Electron mass [kg]

# Assumed constants in de Vries
A_c    = 3                   # Plasma cross section [m^2]
r_0    = 3#6.2                 # Major radius [m]
a      = np.sqrt(A_c/np.pi) #2                  # Minor radius [m]
#tau_RE = 18                  # Confinement time [s]
#A0     = 0                   # Advection [???]
B      = 2.4#5.3                # Magnetic field [T]

# Time and radial parameters
tMax_c = 0.5                           # Simulation time [s]
Nt_c   = 50000                         # Number of time steps
tMax = 5                           # Simulation time [s]
Nt   = 1000                       # Number of time steps
Nr   = 1                             # Number of radial steps
t    = np.linspace(0,tMax,num=Nt+1)  # Time vector for time depending data


#Ions
Z_D = 1
Z_B = 4
a_D = 0.99  # Proportion of ions that are deuterium
a_B = 1 - a_D  # Proportion of ions that are beryllium
n_tot = 5e18  # Total ion density
n_D = a_D * n_tot  # Deuterium density
n_B = a_B * n_tot


T_initial = 1500 #eV
#T_c_list_max = []
Ip_wish = 3.2e6          #Önskat Ip
# Välj Ebase så att Ip = önskat värde (kanske 15 MA för ITER) vid T=Tbase
current_profile = np.linspace(1,1,2)
radie = np.linspace(0,a,2)
ds_conductivity = DREAMSettings()
ds_conductivity.collisions.collfreq_type = Collisions.COLLFREQ_TYPE_PARTIALLY_SCREENED
#ds_conductivity.eqsys.n_cold.setType(ColdElectrons.TYPE_SELFCONSISTENT)
ds_conductivity.eqsys.n_i.setIonization(Ions.IONIZATION_MODE_FLUID)
ds_conductivity.eqsys.n_i.addIon(name='D', Z=1, iontype=Ions.IONS_PRESCRIBED_FULLY_IONIZED, n=n_D)
ds_conductivity.eqsys.n_i.addIon(name='B', Z=4, iontype=Ions.IONS_PRESCRIBED_FULLY_IONIZED, n=n_B)
ds_conductivity.eqsys.n_re.setAvalanche(avalanche=Runaways.AVALANCHE_MODE_FLUID_HESSLOW)
ds_conductivity.eqsys.n_re.setDreicer(Runaways.DREICER_RATE_CONNOR_HASTIE)
ds_conductivity.hottailgrid.setEnabled(False)
ds_conductivity.runawaygrid.setEnabled(False)
ds_conductivity.radialgrid.setB0(B)  # , times=t)
ds_conductivity.radialgrid.setMinorRadius(a)
ds_conductivity.radialgrid.setMajorRadius(r_0)
ds_conductivity.radialgrid.setWallRadius(a)
ds_conductivity.radialgrid.setNr(Nr)
ds_conductivity.solver.setType(Solver.LINEAR_IMPLICIT)
ds_conductivity.timestep.setTmax(tMax)
ds_conductivity.timestep.setNt(Nt)
ds_conductivity.other.include('fluid')
ds_conductivity.eqsys.E_field.setPrescribedData(0)
ds_conductivity.timestep.setNt(1)
ds_conductivity.eqsys.T_cold.setPrescribedData(T_initial)
ds_fun, do_fun, V_loop = generate_current_profile_fun(Ip_wish, current_profile,radie, ds_conductivity)
do_fun.eqsys.I_p.plot()
plt.show()
'''
