import numpy as np
import sys
import matplotlib.pyplot as plt
from generate_Teq import generate_current_profile_fun
from setups import setupSelfconsistent

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
A_c    = 3                   # Plasma cross section [m^2]
r_0    = 3#6.2                 # Major radius [m]
a      = np.sqrt(A_c/np.pi) #2                  # Minor radius [m]
B      = 2.4#5.3                # Magnetic field [T]

# Time and radial parameters
tMax_init = 0.1                           # Simulation time [s]
Nt_init   = 1000                         # Number of time steps
tMax=5
Nt=10000
Nr   = 4                             # Number of radial steps

Ip_wish_vec = np.linspace(1e6,10e6,num=10)
t    = np.linspace(0,tMax,num=Nt+1)  # Time vector for time depending data

#Ions
Z_D = 1
Z_B = 4
a_D = 0.99  # Proportion of ions that are deuterium
a_B = 1 - a_D  # Proportion of ions that are beryllium
n_e = 5e18  # Total ion density
n_tot=n_e/(a_D*Z_D+a_B*Z_B)
n_D = a_D * n_tot  # Deuterium density
n_B = a_B * n_tot

T_max_0 = 3400#9200
T_min_0 = 3300#9100 #eV
T_max_a = 2950#7850 #eV
T_min_a = 2850#7750 #eV

radie = np.linspace(0,a,Nr)
current_profile=1-(radie/a)**2
V_loop_vec=[]
faktor_vec=[]
for Ip_wish in Ip_wish_vec:
    Ip_wish_plot = t * 0 + Ip_wish
    do,Teq,Eeq = generate_current_profile_fun(Ip_wish,current_profile,radie,T_max_0,T_min_0,T_max_a,T_min_a, tMax_init, Nt_init, n_D, n_B, B, a, r_0, Nr)
    T_max_0 = Teq[0]*1.25  # eV
    T_min_0 = Teq[0]*1.15  # eV
    T_max_a = Teq[-1]*1.25  # eV
    T_min_a = Teq[-1]*1.15  # eV
    #plt.plot(radie, Teq)
    #plt.show()

    V_loop_wall_initial = 0.6
    V_loop_wall_final = 2 * np.pi * r_0 * Eeq[-1]
    #V_loop_vec.append(V_loop_wall)
    tol = 1e3
    i = 0
    diff = tol + 1
    E_field = V_loop_wall_initial / (2 * np.pi * r_0) * np.linspace(1, 1, 4)

    while diff > tol:
        V_loop_wall = V_loop_wall_final + (V_loop_wall_initial - V_loop_wall_final) * np.exp(-t)
        T_initial = 500#Teq*0.2
        i = i + 1
        ds = setupSelfconsistent(E_field, T_initial, V_loop_wall, t, Z_D, Z_B, n_D, n_B, B, a, r_0, Nr, tMax, Nt_init)
        do = runiface(ds, quiet=False)
        diff = abs(Ip_wish - do.eqsys.I_p[-1])
        # diff_sum = sum(diff ** 2)
        # print('diff_sum=' + str(diff_sum))
        print('Ip_wish='+str(Ip_wish))
        print('Ip=' + str(do.eqsys.I_p[-1]))
        print('diff=' + str(diff))
        #ax = do.eqsys.I_p.plot()
        #ax2 = plt.plot(t, Ip_wish_plot)  # Want to compare to plasma current at ITER
        #plt.legend(['I_p', 'ITER I_p'])
        #plt.xlim(0, tMax_init)
        #plt.ylim(0, Ip_wish*1.25)
        #plt.show()
        faktor = Ip_wish / do.eqsys.I_p[-1]
        V_loop_wall_initial = faktor * V_loop_wall_initial
        E_field = V_loop_wall_initial / (2 * np.pi * r_0) * np.linspace(1, 1, 4)
    
    faktor = E_field[0] / Eeq[0]
    faktor_vec.append(faktor)

    #T_initial=Teq/Teq[0]*200
    ds = setupSelfconsistent(E_field, T_initial, V_loop_wall, t, Z_D, Z_B, n_D, n_B, B, a, r_0, Nr, tMax, Nt)
    do = runiface(ds, quiet=False)

    print('faktor: ' + str(faktor))
    print('iteration: ' + str(i))
    print('V_loop_wall='+ str(V_loop_wall))
    t=np.linspace(0,tMax,Nt)
    ax = do.eqsys.I_p.plot()
    ax2 = plt.plot(t, np.linspace(1, 1, Nt) * Ip_wish)  # Want to compare to plasma current at ITER
    plt.legend(['I_p', 'ITER I_p'])
    #plt.xlim(0, tMax_init)
    plt.ylim(0, Ip_wish*1.25)
    plt.savefig('sc_3maj/figurer/T_initial=200/I_p__I_p_wish='+str(Ip_wish/1e6)+'.png')
    plt.close()
    #plt.show()

    ax = do.eqsys.T_cold.plot()
    plt.title('$T_{cold}$')
    plt.savefig('sc_3maj/figurer/T_initial=200/T_cold__I_p_wish=' +str(Ip_wish/1e6)+ '.png')
    plt.close()
    #plt.show()

    ax = do.eqsys.E_field.plot()
    plt.title('$E_{field}$')
    plt.savefig('sc_3maj/figurer/T_initial=200/E__I_p_wish=' +str(Ip_wish/1e6)+ '.png')
    plt.close()
    #plt.show()

    ax = do.eqsys.n_re.plot()
    plt.title('$n_{re}$')
    plt.savefig('sc_3maj/figurer/T_initial=200/n_re__I_p_wish=' +str(Ip_wish/1e6)+ '.png')
    plt.close()
    #plt.show()
    #'''

'''
faktor_vec=np.array(faktor_vec)
V_loop_vec=np.array(V_loop_vec)
plt.plot(Ip_wish_vec,V_loop_vec)
plt.title('$V_{loop,wall}$ as a function of $I_p$ at equilibrium')
plt.xlabel('$I_p$')
plt.ylabel('$V_{loop,wall}$')
plt.show()
plt.plot(Ip_wish_vec,faktor_vec)
plt.xlabel('$I_p$')
plt.ylabel('faktor')
plt.show()
np.savetxt('sc_2maj/Data/faktor.txt', faktor_vec, delimiter=',')
np.savetxt('sc_2maj/Data/V_loop_wall.txt', V_loop_vec, delimiter=',')
np.savetxt('sc_2maj/Data/I_p_wish.txt', Ip_wish_vec, delimiter=',')
'''
