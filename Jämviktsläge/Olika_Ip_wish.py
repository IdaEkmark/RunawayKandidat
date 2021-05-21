import numpy as np
import sys
import matplotlib.pyplot as plt


sys.path.append('../../py/')

from DREAM.DREAMSettings import DREAMSettings
import DREAM.Settings.Equations.RunawayElectrons as Runaways
import DREAM.Settings.Solver as Solver
import DREAM.Settings.CollisionHandler as Collisions
import DREAM.Settings.Equations.IonSpecies as Ions
from generate_Teq import generate_Teq
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
B      = 2.4#5.3                # Magnetic field [T]

# Time and radial parameters
tMax_c = 0.5                           # Simulation time [s]
Nt_c   = 50000                         # Number of time steps
tMax = 0.01                           # Simulation time [s]
Nt   = 1000                       # Number of time steps
Nr   = 1                             # Number of radial steps
t    = np.linspace(0,tMax,num=Nt+1)  # Time vector for time depending data


#jonladdningar
Z_D = 1
Z_B = 4

a_D = 0.99  # Andel joner som är deuterium
a_B = 1 - a_D  # Andel joner som är beryllium
n_tot = 5e18  # Total jon densitet
n_D = a_D * n_tot  # Deuterium densitet
n_B = a_B * n_tot # Berrylium densitet

Uppl=20 #aAntal olika strömmar Ip_wish
T_initial = 5000 #Initiala temperaturen i eV
Ip_wish = np.linspace(1e6, 6e6, Uppl) #En array med alla önskade strömmar
#Initierar tomma listor som senare fylls med data som ska sparas
T_c_list = np.zeros((len(Ip_wish), Nt+1))
Teq_list = np.zeros((len(Ip_wish), Nt+1))
T_c_list_max = np.zeros((len(Ip_wish), 1))
V_loop_initial = np.zeros((len(Ip_wish), 1))
V_loop_list = np.zeros((len(Ip_wish), 1))

index_loop = 0 #Ett index för att hålla koll på hur många gånger for loopen har körts
for Ip in Ip_wish:
    current_profile = np.linspace(1,1,2) #Sätter en konstant uniform strömprofil i radiellt led
    radie = np.linspace(0,a,2) #anger tokamakens radie som en vektor
    ### Inställningar för att ta fram konduktiviteten
    ds_conductivity = DREAMSettings()
    ds_conductivity.collisions.collfreq_type = Collisions.COLLFREQ_TYPE_PARTIALLY_SCREENED
    ds_conductivity.eqsys.n_i.setIonization(Ions.IONIZATION_MODE_FLUID)
    ds_conductivity.eqsys.n_i.addIon(name='D', Z=1, iontype=Ions.IONS_PRESCRIBED_FULLY_IONIZED, n=n_D)
    ds_conductivity.eqsys.n_i.addIon(name='B', Z=4, iontype=Ions.IONS_PRESCRIBED_FULLY_IONIZED, n=n_B)
    ds_conductivity.eqsys.n_re.setAvalanche(avalanche=Runaways.AVALANCHE_MODE_FLUID_HESSLOW)
    ds_conductivity.eqsys.n_re.setDreicer(Runaways.DREICER_RATE_CONNOR_HASTIE)
    ds_conductivity.hottailgrid.setEnabled(False)
    ds_conductivity.runawaygrid.setEnabled(False)
    ds_conductivity.radialgrid.setB0(B)
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

    ds_fun, do_fun, V_loop = generate_current_profile_fun(Ip, current_profile,radie, ds_conductivity) #Funktion som tar fram inställningar för att uppnå önskad plasma ström vid den initiala temperaturen
    V_loop_initial[index_loop] = V_loop #sparar de initiela värdena på V_loop
    E_initial = do_fun.eqsys.E_field[-1] #Ansätter det initiala elektriska fältet som det från funktionen ovan

    do, Teq, Eeq = generate_Teq(Ip, current_profile, radie, T_initial, tMax, Nt, n_D, n_B, B, a, r_0, Nr, E_initial, V_loop, t) #Funktion för att ta fram jämviktsläget
    T_cold = do.eqsys.T_cold #Sparar jämviktstemperaturen
    V_loop_list[index_loop]=do.eqsys.E_field[-1]*2*np.pi*r_0 #sparar V_loop vid jämvikt
    #Temperaturen vektorn sparas som ett element i en vektor för varje plasmaström
    i = 0
    for x in T_cold:
        T_c_list[index_loop, i] = x
        i = i+1
    q = 0
    #Här sparas jämviktstemperaturen på samma sätt, ska vara samma som T_c_list
    for k in Teq:
        Teq_list[index_loop, q] = k
        q = q+1
    max = np.amax(T_cold)
    for max in max:
        T_c_list_max[index_loop] = max
    index_loop=index_loop+1

#Datan sparas i textfiler
np.savetxt('Sparad_data_sc/I_p ', Ip_wish, delimiter=',')
np.savetxt('Sparad_data_sc/V_loop_initial.txt', V_loop_initial, delimiter=',')
np.savetxt('Sparad_data_sc/V_loop_final.txt', V_loop_list, delimiter=',')
np.savetxt('Sparad_data_sc/T_c_max.txt', T_c_list_max, delimiter=',')
np.savetxt('Sparad_data_sc/T_c.txt', T_c_list, delimiter=',')
np.savetxt('Sparad_data_sc/Teq.txt', Teq_list, delimiter=',')

                                                    #PLOTAR#
########################################################################################################################
#plottar strömmen
ax = do.eqsys.I_p.plot()
plt.legend(['I_p', 'ITER I_p'])
plt.xlim(0, tMax)
plt.show()

#plottar jämviktstemperatuern
plt.legend(['T_cold','Teq'])
ax = do.eqsys.T_cold.plot()
ax2 = plt.plot(t, np.linspace(1, 1, Nt + 1) * Teq)  # Want to compare to plasma current at ITER
plt.show()

#plottar det elektriska fältet vid jämviktsläget
plt.legend(['E_field','Eeq'])
ax = do.eqsys.E_field.plot()
ax2 = plt.plot(t, np.linspace(1, 1, Nt + 1) * Eeq)  # Want to compare to plasma current at ITER
plt.ylim(bottom=0)
plt.show()
print('Tmax-Tmin\n\n\n\n\nKLAR'+str(index_loop)+'\n\n\n\n\n\n\n\n\n\n')
