import numpy as np
import sys
import matplotlib.pyplot as plt


sys.path.append('../../py/')

from DREAM.DREAMSettings import DREAMSettings
from DREAM.DREAMOutput import DREAMOutput
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
from setups import setup1
from setups import setup2
import time
import os

##Make Parent Folder##
month = time.localtime()[1]
day = time.localtime()[2]
hours = time.localtime()[3]
minutes = time.localtime()[4]
mmdd = str(month) + '_' + str(day)
sshh = str(hours) + ':' + str(minutes)
parent_folder = 'aaaaD06SC_' + mmdd
current_directory = os.getcwd()
if os.path.isdir(parent_folder) == False:
    os.mkdir(current_directory + '/' + parent_folder)



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
tMax_c = 0.5                           # Simulation time [s]
Nt_c   = 5000                         # Number of time steps
tMax = 1                           # Simulation time [s]
Nt   = 10000                         # Number of time steps
Nr   = 1                             # Number of radial steps
t    = np.linspace(0,tMax,num=Nt+1)  # Time vector for time depending data


#Ions
Z_D = 1
Z_B = 4
a_D = 0.6  # Proportion of ions that are deuterium
a_B = 1 - a_D  # Proportion of ions that are beryllium
n_tot = 1.01e20  # Total ion density
n_D = a_D * n_tot  # Deuterium density
n_B = a_B * n_tot

V_loop_wall_list = np.linspace(10, 20, 5)
E_initial = 0 #V/m
T_initial = 25 #eV
T_c_list = np.zeros((len(V_loop_wall_list), Nt+1))
T_c_list_max = np.zeros((len(V_loop_wall_list), 1))
#T_c_list_max = []

index_loop = 0
for V_loop_wall in V_loop_wall_list:
    ####################################################################################################################
                                                        #SIMULATION ONE#
    ####################################################################################################################
    E_initial = V_loop_wall/(r_0*2*np.pi)
    ds_c = setup1(E_initial, T_initial, V_loop_wall, t, Z_D, Z_B, n_D, n_B, B, a, r_0, Nr, tMax_c, Nt_c)

    ds_c.save('settings_SELFCONSISTENT1.h5')
    do_c = runiface(ds_c, 'output_SELFCONSISTENT1.h5', quiet=False)

    ####################################################################################################################
                                                        #SIMULATION TWO#
    ####################################################################################################################
    ds = setup2(ds_c, tMax, Nt, E_initial, T_initial, V_loop_wall, t, Z_D, Z_B, n_D, n_B, B, a, r_0, Nr, tMax_c, Nt_c)
    ds.save('settings_SELFCONSISTENT2.h5')
    do = runiface(ds, 'output_SELFCONSISTENT2.h5', quiet=False)

    ####################################################################################################################
                                                        #SAVE VALUES#
    ####################################################################################################################

    T_cold = do.eqsys.T_cold
    i = 0
    for x in T_cold:
        T_c_list[index_loop, i] = x
        i = i+1
    max = np.amax([T_cold])
    for max in max:
        T_c_list_max[index_loop] = max
    #T_c_list_max.append(np.amax(T_cold))
    index_loop = index_loop+1

    ####################################################################################################################
                                             #PLOTS INSIDE FOR-LOOP#
    ####################################################################################################################
                                                                     #Is created in current folder.
    child_folder = 'aD' + str(round(a_D, 1)) + '_Vloop' + str(round(V_loop_wall, 1)) + '_T' + str(round(T_initial, 1)) + '_' + sshh   #Inside parent-folder. Add something to this string if you're repeating a measurement.

    os.mkdir(current_directory + '/' + parent_folder + '/' + child_folder)


    ax = do_c.eqsys.n_i['D'].plot()
    plt.savefig(parent_folder + '/' + child_folder + '/nD_c')
    ax.clear()

    
    ax = do_c.eqsys.n_i['B'].plot()
    plt.savefig(parent_folder + '/' + child_folder + '/nB_c')
    ax.clear()

    
    ax = do.eqsys.I_p.plot()
    ax2 = plt.plot(t, np.linspace(1, 1, Nt + 1) * 15e6)  # Want to compare to plasma current at ITER
    plt.legend(['I_p', 'ITER I_p'])
    plt.savefig(parent_folder + '/' + child_folder + '/Ip')
    ax.clear()
    ax2.clear()

    ax = do.eqsys.n_re.plot()
    plt.savefig(parent_folder + '/' + child_folder + '/nre')
    ax.clear()


    ax = do.eqsys.n_i['D'].plot()
    plt.savefig(parent_folder + '/' + child_folder + '/nD')
    ax.clear()


    ax = do.eqsys.n_i['B'].plot()
    plt.savefig(parent_folder + '/' + child_folder + '/nB')
    ax.clear()

    
    ax = do.eqsys.T_cold.plot()
    plt.savefig(parent_folder + '/' + child_folder + '/T')
    ax.clear()
    
    ax = do_c.eqsys.E_field.plot()
    plt.savefig(parent_folder + '/' + child_folder + '/Einit')
    ax.clear()
    
    ax = do_c.eqsys.n_cold.plot()
    plt.savefig(parent_folder + '/' + child_folder + '/ncoldinit')
    ax.clear()
    
    ax = do.eqsys.n_cold.plot()
    plt.savefig(parent_folder + '/' + child_folder + '/ncold')
    ax.clear()
    
    ax = do.eqsys.E_field.plot()
    plt.savefig(parent_folder + '/' + child_folder + '/E')
    ax.clear()
########################################################################################################################
                                            #PLOTS/DATA OUTSIDE FOR-LOOP#
########################################################################################################################

np.savetxt('Sparad_data_sc/V_loop_wall.txt', V_loop_wall_list, delimiter=',')
np.savetxt('Sparad_data_sc/T_c_max.txt', T_c_list_max, delimiter=',')
np.savetxt('Sparad_data_sc/T_c.txt', T_c_list, delimiter=',')

for vloopindex in [0,1,2,3,4,5,6]:
    T = T_c_list[vloopindex,:]
    plt.plot(t,T)
plt.show()






