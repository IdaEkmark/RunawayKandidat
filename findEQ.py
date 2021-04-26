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
from generate_Teq import generate_current_profile_fun
import time
import os

##Make Parent Folder##
month = time.localtime()[1]
day = time.localtime()[2]
hours = time.localtime()[3]
minutes = time.localtime()[4]
mmdd = str(month) + '_' + str(day)
sshh = str(hours) + ':' + str(minutes)
parent_folder = 'SC_' + mmdd
current_directory = os.getcwd()
if os.path.isdir(parent_folder) == False:
    os.mkdir(current_directory + '/' + parent_folder)

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
tMax = 0.1                           # Simulation time [s]
Nt   = 1000                         # Number of time steps
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

V_loop_wall = np.linspace(1, 15, 2)
E_initial = 0 #V/m
T_initial = 5000 #eV
T_c_list = np.zeros((len(V_loop_wall), Nt+1))
T_c_list_max = np.zeros((len(V_loop_wall), 1))
#T_c_list_max = []
Ip_wish = 3.2e6          #Önskat Ip
# Välj Ebase så att Ip = önskat värde (kanske 15 MA för ITER) vid T=Tbase
current_profile = np.linspace(1,1,2)
radie = np.linspace(0,a,2)

do,Teq,Eeq = generate_current_profile_fun(Ip_wish,current_profile,radie,T_initial, tMax, Nt, n_D, n_B, B, a, r_0, Nr)


########################################################################################################################
                                                    #PLOTS N' DATA#
########################################################################################################################

child_folder = 'aD' + str(round(a_D, 1)) + '_Ipwish' + str(round(Ip_wish, 1)) + '_T' + str(round(T_initial,
                                                                                                            1)) + '_' + sshh  # Inside parent-folder.
os.mkdir(current_directory + '/' + parent_folder + '/' + child_folder)

ax = do.eqsys.I_p.plot()
ax2 = plt.plot(t, np.linspace(1, 1, Nt + 1) * Ip_wish)  # Want to compare to plasma current at ITER
plt.legend(['I_p', 'ITER I_p'])
plt.xlim(0, tMax)
plt.ylim(0,5e6)
plt.savefig(parent_folder + '/' + child_folder + '/Ip')
plt.show()


ax = do.eqsys.T_cold.plot()
ax2 = plt.plot(t, np.linspace(1, 1, Nt + 1) * Teq)  # Want to compare to plasma current at ITER
plt.xlim(0, tMax)
plt.ylim(0,6000)
plt.savefig(parent_folder + '/' + child_folder + '/T')
plt.show()

ax = do.eqsys.E_field.plot()
ax2 = plt.plot(t, np.linspace(1, 1, Nt + 1) * Eeq)  # Want to compare to plasma current at ITER
plt.ylim(bottom=0)
plt.savefig(parent_folder + '/' + child_folder + '/E')
plt.show()
