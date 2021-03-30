# Test för att reproducera de Vries med parametrar passande de i de Vries och E-fält
# bestämt (prescribed) utifrån de Vries V_res/V_loop

# Jag  har beräknat E(t) på samma sätt som de Vries gör (från I_tot och V_loop, eller bara
# från V_loop) samt satt T_cold till en funktion lik den data de Vries använder. Jag har
# vidare bestämt n_tot utifrån n_e i de Vries artikel, där plasmat består av deuterium och
# beryllium.

# Jag får inte samma I_p som de Vries I_tot utan den beter sig lite intressant men I_RE
# går mot I_p när tiden ökar så det är ju trevligt och kanske kan ganska enkelt förklaras
# med tanke på att det är olika ekvationer som används i simuleringarna. I_p ändras beroende
# på vad man tar för I_re0, vilket är ganska rimligt. Min tanke är att vårt resultat blir att
# finna vilket I_re0 som ger en I_re som dominerar de Vries I_p i slutändan.

# Kan vara så att de Vries räknar ut E på ett annat sätt än jag har antagit från det han
# skrev, eftersom han senare skrev att han tar hänsyn till att Coulomb logaritmen ändras
# från 16 till 11 över tidens gång i beräkningen av E

# Jag ändrar också D0 för att finna vilket D0 som ger konstant ström i slutändan

# Jag använder B=2.4 T


import numpy as np
import sys
from scipy import integrate
import matplotlib.pyplot as plt
from DREAM_run import run_DREAM

sys.path.append('../../py/')

from DREAM.DREAMSettings import DREAMSettings
from DREAM import runiface
import DREAM.Settings.Equations.IonSpecies as Ions
import DREAM.Settings.Equations.RunawayElectrons as Runaways
import DREAM.Settings.Solver as Solver
import DREAM.Settings.CollisionHandler as Collisions
import DREAM.Settings.TransportSettings as Transport
import DREAM.Settings.Equations.ColdElectrons as ColdElectrons

#### Physical parameters ####

# Universal constants
e   = 1.60217662e-19  # Elementary charge [C]
c   = 299792458       # Speed of light [m/s]
m_e = 9.10938356e-31  # Electron mass [kg]

# Assumed constants in de Vries
A_c    = 3                   # Plasma cross section [m^2]
L_tor  = 5e-6                # Inductance [H]
r_0    = 3                   # Major radius [m]
a      = np.sqrt(A_c/np.pi)  # Minor radius [m]
tau_RE = 18                  # Confinement time [s]
A0     = 0                   # Advection [???]
B      = 2.4                 # Magnetic field [T]

# Time and radial parameters
tMax = 8.5                           # Simulation time [s]
Nt   = 10000                         # Number of time steps
Nr   = 1                             # Number of radial steps
t    = np.linspace(0,tMax,num=Nt+1)  # Time vector for time depending data

# Experimental data from de Vries (approximated)
I_p    = 0.9*np.exp(-1/(10*(t+0.8490)))*1e6    # Time dependent plasma current [A]
dI_p   = np.gradient(I_p,t)                    # dI_p/dt, time derivative of the plasma current [A/s]
V_loop = 0.08+0.77*np.exp(-t)                  # Time dependent external torodial loop voltage [V]
n_e    = (0.287*np.exp(-0.8*t**2)+0.023)*1e19  # Time dependent density of thermal electrons
                                               # \approx total density of electrons  [m^-3]
T_e    = (0.052+0.748*np.exp(-0.85*t))*1e3     # Time dependent thermal temperature [eV]

# Atomslag
Z_D=1                        # Atomic number of deuterium
Z_B=4                        # Atomic number of beryllium
UpplZeff = 2                 # antal olika Zeff mellan Z_D och Z_B
Zeff = np.linspace(Z_B, Z_D, UpplZeff)
Index_D0_Z = 0
D0_list=[]
for i in Zeff:
    a_D=(Z_B-i)/(Z_B-Z_D)        # Proportion of ions that are deuterium
    a_B=1-a_D                    # Proportion of ions that are beryllium
    n_tot=n_e/(Z_D*a_D+Z_B*a_B)  # Total ion density
    n_D_for=a_D*n_tot            # Deuterium density
    n_B_for=a_B*n_tot            # Beryllium density
    # Making array to define constant densitys on two radial coordinates
    n_D=[]
    n_B=[]
    for elements in n_D_for:
        n_D.append([elements, elements])
    n_D=np.array(n_D)
    for elements in n_B_for:
        n_B.append([elements, elements])
    n_B=np.array(n_B)
    r=np.linspace(0,a,num=Nr+1)

    # Sweep detalis
    legend=['$I_p$']
    D0min=0
    D0max=0.1
    #D0num=101
    D0num = 10
    I_re0min=1e5
    I_re0max=8e5
    #I_re0num=7*10+1
    I_re0num=10
    #### E \sim V_loop ####
    # Initial values in de Vries, changeable
    I_re  = 6e5             # Initial runaway current [A]
    n_re0 = I_re/(e*c*A_c)  # Initial density of runaway electrons [m^-3]

    # Electric field calculated as in de Vries article
    E_loop = V_loop/(2*np.pi*r_0)  # Time dependent electric field strength [V/m]

    # Find D0 for most constant current
    D0_loop=np.linspace(D0min,D0max,num=D0num)
    dos_loop_D0=[]
    for Ds in D0_loop:
        ds_loop_D0, do_loop_D0 =run_DREAM(E_loop,T_e,'D',Z_D,n_D,'B',Z_B,n_B,n_re0,B,a,r_0,A0,Ds,r,Nr,t,tMax,Nt)
        dos_loop_D0.append(do_loop_D0)
    min_I_p_var_loop=1e20
    i_loop_D0=0
    D0_min_loop = -1
    for do in dos_loop_D0:
        I_p_test_const = do.eqsys.I_p[5500:]
        I_p_var=np.var(I_p_test_const)
        if I_p_var < min_I_p_var_loop:
            min_I_p_var_loop = I_p_var
            D0_min_loop = D0_loop[i_loop_D0]
        i_loop_D0=i_loop_D0+1
    D0_loop=D0_min_loop

    # Find I_re0 which gives I_re=I_p in the end
    I_re  = np.linspace(I_re0min, I_re0max,num=I_re0num)
    n_re0 = I_re/(e*c*A_c)  # Initial density of runaway electrons [m^-3]
    dos_loop_Ire0=[]
    for n_re0s in n_re0:
        ds_loop_Ire0, do_loop_Ire0 = run_DREAM(E_loop,T_e,'D',Z_D,n_D,'B',Z_B,n_B,n_re0s,B,a,r_0,A0,D0_loop,r,Nr,t,tMax,Nt)
        dos_loop_Ire0.append(do_loop_Ire0)
    min_diff_loop=1e20
    i_loop_Ire0=0
    for do in dos_loop_Ire0:
        n_re_do = do.eqsys.n_re[5500:].T[0]
        I_re_do = n_re_do * e * c * A_c
        I_p_do=I_p[5500:]
        diff=I_p_do-I_re_do
        diff_sum=sum(diff**2)
        if diff_sum < min_diff_loop:
            min_diff_loop = diff_sum
            I_re0_min_diff_loop = I_re[i_loop_Ire0]
            index_loop_Ire0=i_loop_Ire0
        i_loop_Ire0=i_loop_Ire0+1
    legend.append('$I_{re0,loop}=$'+str(I_re0_min_diff_loop))

#### E \sim V_res ####
# Initial values in de Vries, changeable
#### Manage result ####

    plt.plot(t, I_p)
    n_re_loop = dos_loop_Ire0[index_loop_Ire0].eqsys.n_re[:]
    I_re_loop = n_re_loop * e * c * A_c
    tid = np.linspace(0, tMax, num=len(I_re_loop))
    plt.plot(tid, I_re_loop)
    #n_re_res = dos_res_Ire0[index_res_Ire0].eqsys.n_re[:]
    #I_re_res = n_re_res * e * c * A_c
    #tid = np.linspace(0, tMax, num=len(I_re_res))
    #plt.plot(tid, I_re_res)
    plt.xlim(-1.5, 8.5)
    plt.ylim(bottom=0)
    plt.xlabel('tid [s]')
    plt.ylabel('I [A]')
    #plt.title('Changing $I_{re0}$ for $E\sim V_{loop}$ and $E\sim V_{res}$, $D_{loop,0}=$'+str(D0_loop)+' & $D_{res,0}=$'+str(D0_res))
    plt.title('Changing $I_{re0}$ for $E\sim V_{loop}$, $D_{loop,0}=$'+str(D0_loop))
    plt.legend(legend)
    plt.show()
    D0_list[Index_D0_Z]=D0_loop
    Index_D0_Z=Index_D0_Z+1
    manager = plt.get_current_fig_manager()
    manager.window.maximize()
    plt.savefig('Find_Grafer=' + str(I_re) + '.png')
    #plt.show()
    plt.close()
#Fixa grafen Zeff,D0-list och spara den
D0_list = np.array(D0_list)
plt.plot(Zeff, D0_list)
plt.xlabel('Zeff')
plt.ylabel('D0')
'''
plt.plot(t, I_p)
for do in dos:
    #do.eqsys.I_p.plot()
    #tid = np.linspace(0, tMax, num=len(I_p))
    n_re = do.eqsys.n_re[:]
    I_re = n_re * e * c * A_c
    tid = np.linspace(0, tMax, num=len(I_re))
    plt.plot(tid, I_re)
plt.xlim(-1.5, 8.5)
plt.ylim(bottom=0)
plt.xlabel('tid [s]')
plt.ylabel('I_re [A]')
#plt.title('Changing B')
#plt.title('Changing a_D')
plt.title('Changing I_re0')
plt.legend(legend)
plt.show()

plt.plot(t, I_p)
for do in dos:
    do.eqsys.I_p.plot()
plt.xlim(-1.5, 8.5)
plt.ylim(bottom=0)
plt.xlabel('tid [s]')
plt.ylabel('I_p [A]')
#plt.title('Changing B')
#plt.title('Changing a_D')
plt.title('Changing I_re0')
plt.legend(legend)
plt.show()

for do in dos:
    E_c_eff = do.other.fluid.Eceff[:]
    E_c_free = do.other.fluid.Ecfree[:]
    kappa = E_c_eff / E_c_free
    tid = np.linspace(0, tMax, num=len(kappa))
    plt.plot(tid, kappa)
plt.xlim(-1.5, 8.5)
plt.ylim(bottom=0)
plt.xlabel('tid [s]')
plt.ylabel('Kappa')
#plt.title('Changing B')
#plt.title('Changing a_D')
plt.title('Changing I_re0')
legend.pop(0)
plt.legend(legend)
plt.show()

for do in dos:
    GammaAva = do.other.fluid.GammaAva[:]
    tid = np.linspace(0, tMax, num=len(GammaAva))
    plt.plot(tid, GammaAva)
plt.xlim(-1.5, 8.5)
#plt.ylim(bottom=0)
plt.xlabel('tid [s]')
plt.ylabel('GammaAva')
#plt.title('Changing B')
#plt.title('Changing a_D')
plt.title('Changing I_re0')
plt.legend(legend)
plt.show()


# Kappa
E_c_eff = do.other.fluid.Eceff[:]
E_c_free = do.other.fluid.Ecfree[:]
kappa = E_c_eff/E_c_free
tid=np.linspace(0,tMax,num=len(kappa))
plt.plot(tid,kappa)
#plt.xlim(-1.5, 8.5)
#plt.ylim(0, 40)
plt.show()

##################################################################################################
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

# Plot Coulomb logarithm
#lnLambdaC = do.other.fluid.lnLambdaC[:]
#lnLambdaT = do.other.fluid.lnLambdaT[:]
#tid=np.linspace(0,tMax,num=len(lnLambdaC))
#plt.plot(tid,lnLambdaC)
#plt.show()
#plt.plot(tid,lnLambdaT)
#plt.show()

# Plot E_field
#do.eqsys.E_field.plot(r=[0,-1])
#plt.show()

# Plot n_tot and n_re
#do.eqsys.n_tot.plot(r=[0,-1])
#do.eqsys.n_re.plot(r=[0,-1])
#plt.show()

# Plot streaming parameter according to de Vries
#I_p = do.eqsys.I_p[:]
#T_J=1.602176565e-19*do.eqsys.T_cold[:] #Temperatur i joule
#v_th=np.sqrt(T_J/m_e)
#j_tot=I_p/A_c
#xi=np.abs(j_tot/(e*n*v_th))
#tid=np.linspace(0,tMax,num=len(xi))
#plt.plot(tid,xi)
#plt.show()

# Plot n_tot and n_re
#do.eqsys.n_tot.plot(r=[0,-1])
#do.eqsys.n_re.plot(r=[0,-1])
#plt.show()

# Plot n_tot och n_tot-n_e
#do.eqsys.n_tot.plot(r=[0,-1])
#n_re2=n_e-n_e
#plt.plot(t,n_re2)
#plt.show()
#'''