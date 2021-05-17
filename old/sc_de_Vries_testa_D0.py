
import numpy as np
import sys
from scipy import integrate
import matplotlib.pyplot as plt
import matplotlib
from sc_DREAM_run import run_DREAM


sys.path.append('../../py/')

from DREAM.DREAMSettings import DREAMSettings
from DREAM import runiface
import DREAM.Settings.Equations.IonSpecies as Ions
import DREAM.Settings.Equations.RunawayElectrons as Runaways
import DREAM.Settings.Solver as Solver
import DREAM.Settings.CollisionHandler as Collisions
import DREAM.Settings.TransportSettings as Transport
import DREAM.Settings.Equations.ColdElectrons as ColdElectrons
import DREAM.Settings.Equations.ElectricField as ElectricField

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

# Electric field calculated as in de Vries article
V_res = V_loop-L_tor*dI_p     # Time dependent resistive voltage [V] (V_res=I_tot*R_tot)
E     = V_loop/(2*np.pi*r_0)  # Time dependent electric field strength [V/m] (Can be
                              # E=V_res/(2*np.pi*r_0))
E_init = E[0]

# Atomslag
Z_D=1                        # Atomic number of deuterium
Z_B=4                        # Atomic number of beryllium
a_D=(Z_B-2)/(Z_B-Z_D)        # Proportion of ions that are deuterium
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

I_re_arr  = np.linspace(1e5, 8e5,num=8)
for I_re in I_re_arr:
    n_re0 = I_re / (e * c * A_c)  # Initial density of runaway electrons [m^-3]
    D0=np.linspace(0,0.2,num=21)
    legend=['$I_p$']
    for elem in D0:
        legend.append(elem)
    #### Manage result ####
    dos=[]
    for Ds in D0:
        ds, do = run_DREAM(E_init, V_loop, T_e, 'D', Z_D, n_D, 'B', Z_B, n_B, n_re0, B, a, r_0, A0, Ds, r, Nr, t, tMax,Nt)
        dos.append(do)

    plt.plot(t, I_p)
    for do in dos:
        I_pen = do.eqsys.I_p[:]
        tid = np.linspace(0, tMax, num=len(I_pen))
        plt.plot(tid, I_pen)
    plt.xlim(-2, 8.5)
    plt.ylim(bottom=0)
    plt.xlabel('tid [s]')
    plt.ylabel('I_p [A]')
    plt.title('Changing D0, $I_re0=$'+str(I_re))
    plt.legend(legend,prop={'size': 8})
    manager = plt.get_current_fig_manager()
    manager.window.maximize()
    plt.savefig('Find_D0_25mars/I_p__I_re0='+str(I_re)+'.png')
    #plt.show()
    plt.close()

    plt.plot(t, I_p)
    for do in dos:
        n_re = do.eqsys.n_re[:]
        I_ren = n_re * e * c * A_c
        tid = np.linspace(0, tMax, num=len(I_ren))
        plt.plot(tid, I_ren)
    plt.xlim(-2, 8.5)
    plt.ylim(bottom=0)
    plt.xlabel('tid [s]')
    plt.ylabel('I_RE [A]')
    plt.title('Changing D0, $I_re0=$'+str(I_re))
    plt.legend(legend,prop={'size': 8})
    manager = plt.get_current_fig_manager()
    manager.window.maximize()
    plt.savefig('Find_D0_25mars/I_re__I_re0=' + str(I_re) + '.png')
    #plt.show()
    plt.close()


#    for do in dos:
#        E_c_eff = do.other.fluid.Eceff[:]
#        E_c_free = do.other.fluid.Ecfree[:]
#        kappa = E_c_eff / E_c_free
#        tid = np.linspace(0, tMax, num=len(kappa))
#        plt.plot(tid, kappa)
#    plt.xlim(-1.5, 8.5)
#    plt.ylim(bottom=0)
#    plt.xlabel('tid [s]')
#    plt.ylabel('Kappa')
    # plt.title('Changing B')
    # plt.title('Changing a_D')
#    plt.title('Changing a_D and B, B=' + str(Bs))
#    legend.pop(-1)
#    plt.legend(legend)
#    plt.show()

#'''
'''
for do in dos:
    n_re = do.eqsys.n_re[:]
    I_re = n_re*e*c*A_c
    tid = np.linspace(0, tMax, num=len(I_re))
    plt.plot(tid,I_re)
plt.xlim(-1.5, 8.5)
plt.ylim(bottom=0)
plt.xlabel('tid [s]')
plt.ylabel('I_re [s]')
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

#print('Sökt n_e ' + str(n_e))
#print('Faktisk n_e ' + str(Z_D*n_D+Z_B*n_B))

#for do in dos:
#    do.eqsys.I_p.plot()
#    tid = np.linspace(0, tMax, num=len(I_p))
#plt.xlim(-1.5, 8.5)
#plt.ylim(bottom=0)
#plt.xlabel('tid [s]')
#plt.ylabel('I_p [s]')
#plt.legend(legend)
#plt.show()

#'''
