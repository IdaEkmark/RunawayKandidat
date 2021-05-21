import numpy as np
import sys
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
from generate_Teq import generate_Teq
from generate_Teq_Ny import generate_initial_values
from Test import generate_current_profile_fun
#### Physical parameters ####

# Universal constants
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
tMax = 0.01                           # Simulation time [s]
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

Uppl=20
T_initial = 5000 #eV
Ip_wish = np.linspace(1e6, 6e6, Uppl)
#Ip_wish = 3.2e6          #Önskat Ip
T_c_list = np.zeros((len(Ip_wish), Nt+1))
Teq_list = np.zeros((len(Ip_wish), Nt+1))
T_c_list_max = np.zeros((len(Ip_wish), 1))
V_loop_initial = np.zeros((len(Ip_wish), 1))
V_loop_list = np.zeros((len(Ip_wish), 1))
index_loop = 0
for Ip in Ip_wish:
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
    ds_fun, do_fun, V_loop = generate_current_profile_fun(Ip, current_profile,radie, ds_conductivity)
    V_loop_initial[index_loop]=V_loop
    #do_fun.eqsys.I_p.plot()
    #plt.show()
    #plt.plot(V_loop,'o')
    #plt.show()
    #do_fun.eqsys.E_field.plot()
    #plt.show()
    '''
    #                                    (I_p,current_profile,radie,T_initial, tMax, Nt, n_D, n_B, B, a, r_0, Nr):
    do,V_loop,E = generate_initial_values(Ip_wish, current_profile, radie, T_initial, tMax, Nt, n_D, n_B, B, a, r_0, Nr, t)
    do.eqsys.I_p.plot()
    plt.show()
    Uppl = 2 #Upplösning, hur många V_loop_wall som används
    V_loop_wall = np.linspace(0.001, 0.05, Uppl)
    T_c_list = np.zeros((len(V_loop_wall), Nt+1))
    Teq_list = np.zeros((len(V_loop_wall), Nt+1))
    T_c_list_max = np.zeros((len(V_loop_wall), 1))
    #T_c_list_max = []
    '''
    ########################################################################################################################
                                                        #SIMULATION ONE#
    ########################################################################################################################
    E_initial = do_fun.eqsys.E_field[-1]

    do, Teq, Eeq = generate_Teq(Ip, current_profile, radie, T_initial, tMax, Nt, n_D, n_B, B, a, r_0, Nr, E_initial, V_loop, t)
    T_cold = do.eqsys.T_cold
    V_loop_list[index_loop]=do.eqsys.E_field[-1]*2*np.pi*r_0
    i = 0
    for x in T_cold:
        T_c_list[index_loop, i] = x
        i = i+1
    q = 0
    for k in Teq:
        Teq_list[index_loop, q] = k
        q = q+1
    max = np.amax(T_cold)
    for max in max:
        T_c_list_max[index_loop] = max
        #T_c_list_max.append(np.amax(T_cold))
    #plt.ylabel('T_c')
    #do.eqsys.T_cold.plot()
    #do.eqsys.E_field.plot()
    #plt.show()
    index_loop=index_loop+1

########################################################################################################################
'''
index_loop = 0
for V_loop_wall_for in V_loop_wall:
    E_initial = V_loop_wall_for/(2*np.pi*r_0) #V/m
    do, Teq, Eeq = generate_current_profile_fun(Ip_wish, current_profile, radie, T_initial, tMax, Nt, n_D, n_B, B, a, r_0, Nr, E_initial, V_loop_wall_for, t)
    T_cold = do.eqsys.T_cold
    i = 0
    for x in T_cold:
        T_c_list[index_loop, i] = x
        i = i+1
    q = 0
    for k in Teq:
        Teq_list[index_loop, q] = k
        q = q+1
    max = np.amax(T_cold)
    for max in max:
        T_c_list_max[index_loop] = max
    #T_c_list_max.append(np.amax(T_cold))
    plt.ylabel('T_c')
    do.eqsys.T_cold.plot()
    do.eqsys.E_field.plot()
    index_loop = index_loop+1
plt.show()
'''
########################################################################################################################
                                                    #PLOTS#
########################################################################################################################
np.savetxt('Sparad_data_sc/I_p ', Ip_wish, delimiter=',')
np.savetxt('Sparad_data_sc/V_loop_initial.txt', V_loop_initial, delimiter=',')
np.savetxt('Sparad_data_sc/V_loop_final.txt', V_loop_list, delimiter=',')
np.savetxt('Sparad_data_sc/T_c_max.txt', T_c_list_max, delimiter=',')
np.savetxt('Sparad_data_sc/T_c.txt', T_c_list, delimiter=',')
np.savetxt('Sparad_data_sc/Teq.txt', Teq_list, delimiter=',')




########################################################################################################################
                                                    #PLOTS#
########################################################################################################################
ax = do.eqsys.I_p.plot()
#ax2 = plt.plot(t, np.linspace(1, 1, Nt + 1) * Ip_wish)  # Want to compare to plasma current at ITER
plt.legend(['I_p', 'ITER I_p'])
plt.xlim(0, tMax)
plt.show()

plt.legend(['T_cold','Teq'])
ax = do.eqsys.T_cold.plot()
ax2 = plt.plot(t, np.linspace(1, 1, Nt + 1) * Teq)  # Want to compare to plasma current at ITER
#plt.xlim(0, tMax)
#plt.ylim(0,6000)
plt.show()

plt.legend(['E_field','Eeq'])
ax = do.eqsys.E_field.plot()
ax2 = plt.plot(t, np.linspace(1, 1, Nt + 1) * Eeq)  # Want to compare to plasma current at ITER
plt.ylim(bottom=0)
plt.show()
print('Tmax-Tmin\n\n\n\n\nKLAR'+str(index_loop)+'\n\n\n\n\n\n\n\n\n\n')
