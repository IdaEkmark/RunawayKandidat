# Code to reproduce the result in de Vries study.
# The code finds the initial runaway current I_re0 and diffusion D0 that results in a
# stable runaway current after 5.1 s, which dominated I_pV for three effective charges
# Z_eff.
# It will plot the runaway current I_re, plasma current I_p, enhancement factor kappa
# and avalanche-koefficient Gamma_ava


import numpy as np
import sys
from scipy import integrate
import matplotlib.pyplot as plt
from de_Vries_DREAM_settings import run_DREAM

#### Physical parameters ####

# Universal constants
e   = 1.60217662e-19  # Elementary charge [C]
c   = 299792458       # Speed of light [m/s]

# Assumed constants in de Vries
A_c    = 3                     # Plasma cross section [m^2]
L_tor  = 5e-6                  # Inductance [H]
R_0    = 3                     # Major radius [m]
r_0    = np.sqrt(A_c/np.pi)  # Minor radius [m]
tau_RE = 18                    # Confinement time [s]
A0     = 0                     # Advection [???]
B      = 2.4                   # Magnetic field [T]

# Time and radial parameters
tMax = 8.5                           # Simulation time [s]
Nt   = 10000                         # Number of time steps
Nr   = 1                             # Number of radial steps
t    = np.linspace(0,tMax,num=Nt+1)  # Time vector for time depending data
r    = np.linspace(0,r_0,num=Nr+1)    # Radial vector for radius depending data

# Experimental data from de Vries (approximated)
IpV    = 0.9*np.exp(-1/(10*(t+0.8490)))*1e6    # Time dependent plasma current [A]
V_loop = 0.08+0.77*np.exp(-t)                  # Time dependent external torodial loop voltage [V] \approx total density of electrons  [m^-3]
n_e    = (0.287*np.exp(-0.8*t**2)+0.023)*1e19  # Time dependent density of thermal electrons \approx total density of electrons  [m^-3]
m_vec  = np.exp(-1/(0.58*(t+1.1421)))           # Multiply this with T_e to recieve a more realistic I_p
T_e    = (0.055+0.748*np.exp(-0.85*t))*1e3      # Time dependent thermal temperature [eV]

# Electric field calculated as in de Vries article
E_loop = V_loop/(2*np.pi*R_0)  # Time dependent electric field strength [V/m]

# Lists for saving the best parameters
D0_bra_list    = []  # List to save the best D0 for each a_D
I_re0_bra_list = []  # List to save the best I_re0 for each a_D
Z_eff_list     = []  # List to save the Z_eff
do_bra_list    = []  # List to save the best DREAM-output-object for each a_D

# Analyzing the runaway current for different proportions of deuterium and beryllium
a_D_for=np.array([8/9,0.95,1])  # Proportion of ions that are deuterium
for a_D in a_D_for:
    Z_D     = 1                                      # Atomic number of deuterium
    Z_B     = 4                                      # Atomic number of beryllium
    a_B     = 1-a_D                                  # Proportion of ions that are beryllium
    n_tot   = n_e/(Z_D*a_D+Z_B*a_B)                  # Total ion density
    n_D_for = a_D*n_tot                              # Deuterium density
    n_B_for = a_B*n_tot                              # Beryllium density
    Z_eff=(a_D*Z_D**2+a_B*Z_B**2)/(a_D*Z_D+a_B*Z_B)  # Effective charge
    Z_eff_list.append(Z_eff)

    n_D = []  # Making array to define constant Deuterium-density on two radial coordinates
    n_B = []  # Making array to define constant Beryllium-density on two radial coordinates
    for elements in n_D_for:
        n_D.append([elements, elements])
    n_D = np.array(n_D)
    for elements in n_B_for:
        n_B.append([elements, elements])
    n_B = np.array(n_B)

    # Sweep detalis
    D0min    = 0.01                    # Lower limit of D0 sweep
    D0max    = 0.025                   # Upper limit of D0 swep
    D0num    = 151                     # Number of D0-steps
    I_re0min = 5e5                     # Lower limit of I_re0 sweep
    I_re0max = 6e5                     # Upper limit of I_re0 sweep
    I_re0num = 101                     # Number of I_re0-steps
    index    = np.linspace(1,1,num=1)  # Number of iterations
    I_re_for = 6e5                     # Initial runaway current [A]
    for i in index:  # Iterating to find best D0 from previous best I_re0 I
        n_re0 = I_re_for/(e*c*A_c)  # Initial density of runaway electrons [m^-3]

        # Find D0 for most constant current
        D0_loop     = np.linspace(D0min,D0max,num=D0num) # D0-sweep-values
        dos_loop_D0 = []                                 # List to save DREAM-output-object
        for Ds in D0_loop:  # Running DREAM for every D0-sweep-value
            ds_loop_D0, do_loop_D0 = run_DREAM(E_loop,T_e,'D',Z_D,n_D,'B',Z_B,n_B,n_re0,B,r_0,R_0,A0,Ds,r,Nr,t,tMax,Nt)
            dos_loop_D0.append(do_loop_D0)  # Saving each DREAM-output-object

        min_I_p_var_loop = 1e20  # Variable for I_p to find minimum value of variance of I_p for each of the D0-sweep-values
        i_loop_D0        = 0     # Index of iteration
        D0_min_loop      = -1    # Variable for D0 to find minimum value of variance of I_p for each of the D0-sweep-values

        for do in dos_loop_D0:  # Going through the DREAM-output-objects to find the one D0 with most constant I_p in the end
            I_p_test_const = do.eqsys.I_p[6000:]     # I_p after 5.1 s
            I_p_var        = np.var(I_p_test_const)  # Variance of I_p after 5.1 s
            if I_p_var < min_I_p_var_loop:           # Find minimum variance of I_p
                min_I_p_var_loop = I_p_var           # Find value of I_p with minimum variance
                D0_min_loop = D0_loop[i_loop_D0]     # Find value of D0 with minimum variance of I_p
            i_loop_D0=i_loop_D0+1
        D0_loop = D0_min_loop  # Save value of D0 with minimum variance of I_p

        # Find I_re0 which gives I_re=IpV in the end
        I_re          = np.linspace(I_re0min, I_re0max,num=I_re0num)  # D0-sweep-values
        n_re0         = I_re/(e*c*A_c)                                # Initial density of runaway electrons [m^-3]
        dos_loop_Ire0 = []                                            # List to save DREAM-output-object

        for n_re0s in n_re0:  # Running DREAM for every I_re0-sweep-value
            ds_loop_Ire0, do_loop_Ire0 =run_DREAM(E_loop,T_e,'D',Z_D,n_D,'B',Z_B,n_B,n_re0s,B,r_0,R_0,A0,D0_loop,r,Nr,t,tMax,Nt)
            dos_loop_Ire0.append(do_loop_Ire0)  # Saving each DREAM-output-object

        min_diff_loop = 1e20  # Variable for difference between I_re and I_pV
        i_loop_Ire0   = 0     # Index of iteration

        for do in dos_loop_Ire0:  # Going through the DREAM-output-objects to find the one I_re0 wich dominates IpV,deVries in the end
            n_re_do  = do.eqsys.n_re[6000:].T[0]         # n_re after 5.1 s
            I_re_do  = n_re_do * e * c * A_c             # I_re after 5.1 s
            IpV_do   = IpV[6000:]                        # I_pV after 5.1 s
            diff     = IpV_do-I_re_do                    # Difference between I_re and I_pV after 5.1 s
            diff_sum = sum(diff**2)                      # Measurement of total difference to find minimum difference
            if diff_sum < min_diff_loop:                 # Find minimum difference between I_pV and I_re
                min_diff_loop       = diff_sum           # Find value of minimum difference
                I_re0_min_diff_loop = I_re[i_loop_Ire0]  # Find value of I_re0 with minimum difference
                index_loop_Ire0     = i_loop_Ire0        # Find index of value of I_re0 with minimum difference
            i_loop_Ire0=i_loop_Ire0+1

        I_re_for = I_re0_min_diff_loop  # I_re for next iteration

    I_re0_bra_list.append(I_re0_min_diff_loop)
    do_bra_list.append(dos_loop_Ire0[index_loop_Ire0])
    D0_bra_list.append(D0_loop)


#### Manage result ####
# Save data
np.savetxt('pr_12maj/Test1/Z_eff.txt', Z_eff_list, delimiter=',')
np.savetxt('pr_12maj/Test1/D0.txt', D0_bra_list, delimiter=',')
np.savetxt('pr_12maj/Test1/I_re0.txt', I_re0_bra_list, delimiter=',')
for do, Z_eff in zip(do_bra_list, Z_eff_list):
    n_re_bra_list = do.eqsys.n_re[:]
    np.savetxt('pr_12maj/Test1/n_re_Z_eff='+str(round(Z_eff,1))+'.txt', n_re_bra_list, delimiter=',')
    I_p_bra_list = do.eqsys.I_p[:]
    np.savetxt('pr_12maj/Test1/I_p_Z_eff=' + str(round(Z_eff, 1)) + '.txt', I_p_bra_list, delimiter=',')
    E_c_eff_bra_list = do.other.fluid.Eceff[:]
    np.savetxt('pr_12maj/Test1/E_c_eff_Z_eff=' + str(round(Z_eff, 1)) + '.txt', E_c_eff_bra_list, delimiter=',')
    E_c_free_bra_list = do.other.fluid.Ecfree[:]
    np.savetxt('pr_12maj/Test1/E_c_free_Z_eff=' + str(round(Z_eff, 1)) + '.txt', E_c_free_bra_list, delimiter=',')
    GammaAva_bra_list = do.other.fluid.GammaAva[:]
    np.savetxt('pr_12maj/Test1/GammaAva_Z_eff=' + str(round(Z_eff, 1)) + '.txt', GammaAva_bra_list,delimiter=',')

# Plot best time-dependant parameters
#legend_a_D = []      # List for a_D to be shown in legend
#for a_D in a_D_for:  # Filling list with a_D-sweep-values
#    legend_a_D.append(a_D)
#legend_a_D[2]='2/3'  # Change 0.666666666667 to 2/3


# Figure 1
legend = ['$I_{p,V}$']  # Total legend with IpV,deVries as firt element
plt.plot(t, IpV)              # Plots IpV as first graph
for Z_eff, D0, I_re0, do in zip(Z_eff_list, D0_bra_list, I_re0_bra_list, do_bra_list):  # For loop to plot best I_re for every a_D with describing legend
    legend.append('$Z_{eff}=$' + str(round(Z_eff,1)) + ', $D_0=$' + str("%.3f" % D0) + ', \n$I_{re0}=$' + str(I_re0))  # Legend describing a_D, D0 and I_re0

    n_re_loop = do.eqsys.n_re[:]                          # n_re for the best I_re0 and D0 value
    I_re_loop = n_re_loop * e * c * A_c                   # I_re for the best I_re0 and D0 value
    tid       = np.linspace(0, tMax, num=len(I_re_loop))  # Equivalent time steps for plot
    plt.plot(tid, I_re_loop)                              # Plot I_re

# Plot details
plt.xlim(-2.5, 8.5)
plt.ylim(0,1e6)
plt.xlabel('t [s]')
plt.ylabel('I [A]')
plt.title('$I_{re}$ - Finding $I_{re0}$ and $D_0$ for different $Z_{eff}$')
plt.legend(legend,loc='upper left')
manager = plt.get_current_fig_manager()
manager.window.maximize()
#plt.savefig('pr_26mars/I_re__different__a_D.png')
plt.show()

# Figure 2
legend=['$I_{p,V}$']  # Total legend with I_pV as first element
plt.plot(t, IpV)            # Plots I_pV as first graph
for Z_eff, D0, I_re0, do in zip(Z_eff_list, D0_bra_list, I_re0_bra_list, do_bra_list):  # For loop to plot best I_re for every a_D with describing legend
    legend.append('$I_p$: ' + '$Z_{eff}=$' + str(round(Z_eff,1)) + ', $D_0=$' + str("%.3f" % D0) + ', \n$I_{re0}=$' + str(I_re0))  # Legend describing a_D, D0 and I_re0
    do.eqsys.I_p.plot()  # Plots DREAM's I_p

    legend.append('$I_{re}$: ' + '$Z_{eff}=$' + str(round(Z_eff,1)) + ', $D_0=$' + str("%.3f" % D0) + ', \n$I_{re0}=$' + str(I_re0))  # Legend describing a_D, D0 and I_re0
    n_re_loop = do.eqsys.n_re[:]                          # n_re for the best I_re0 and D0 value
    I_re_loop = n_re_loop * e * c * A_c                   # I_re for the best I_re0 and D0 value
    tid       = np.linspace(0, tMax, num=len(I_re_loop))  # Equivalent time steps for plot
    plt.plot(tid, I_re_loop)                              # Plot I_re

# Plot details
plt.xlim(-2.5, 8.5)
plt.ylim(bottom=0)
plt.xlabel('t [s]')
plt.ylabel('I [A]')
plt.title('$I_p$ and $I_{re}$ - Finding $I_{re0}$ and $D_0$ for different $Z_{eff}$')
plt.legend(legend,prop={'size': 8},loc='upper left')
manager = plt.get_current_fig_manager()
manager.window.maximize()
#plt.savefig('pr_26mars/I_p__different__a_D.png')
plt.show()

# Figure 3
legend = ['$I_{p,deVries}$']  # Total legend with I_pV as firt element
plt.plot(t, IpV)              # Plots I_pV as first graph
for Z_eff, D0, I_re0, do in zip(Z_eff_list, D0_bra_list, I_re0_bra_list, do_bra_list):  # For loop to plot best I_re for every a_D with describing legend
    legend.append('$Z_{eff}=$' + str(round(Z_eff,1)) + ', $D_0=$' + str("%.3f" % D0) + ', \n$I_{re0}=$' + str(I_re0))  # Legend describing a_D, D0 and I_re0
    do.eqsys.I_p.plot()  # Plots DREAM's I_p

# Plot details
plt.xlim(-2.5, 8.5)
plt.ylim(bottom=0)
plt.xlabel('t [s]')
plt.ylabel('I [A]')
plt.title('$I_p$ - Finding $I_{re0}$ and $D_0$ for different $Z_{eff}$')
plt.legend(legend,loc='upper left')
manager = plt.get_current_fig_manager()
manager.window.maximize()
#plt.savefig('pr_26mars/I_p__different__a_D.png')
plt.show()

# Figure 4
legend = []  # Total legend
for Z_eff, D0, I_re0, do in zip(Z_eff_list, D0_bra_list, I_re0_bra_list, do_bra_list):  # For loop to plot best I_re for every a_D with describing legend
    legend.append('$Z_{eff}=$' + str(round(Z_eff,1)) + ', $D_0=$' + str("%.3f" % D0) + ', \n$I_{re0}=$' + str(I_re0))  # Legend describing a_D, D0 and I_re0
    E_c_eff  = do.other.fluid.Eceff[:]               # kappa*E_c
    E_c_free = do.other.fluid.Ecfree[:]              # E_c
    kappa    = E_c_eff / E_c_free                    # Enhancing factor kappa
    tid      = np.linspace(0, tMax, num=len(kappa))  # Equivalent time steps for plot
    plt.plot(tid, kappa)  # Plots kappa

# Plot details
plt.xlim(-2.5, 8.5)
plt.ylim(bottom=0)
plt.xlabel('t [s]')
plt.ylabel('$\kappa$')
plt.title('$\kappa$ - Finding $I_{re0}$ and $D_0$ for different $Z_{eff}$')
plt.legend(legend,loc='upper left')
manager = plt.get_current_fig_manager()
manager.window.maximize()
#plt.savefig('pr_26mars/kappa__different__a_D.png')
plt.show()

# Figure 5
legend = []  # Total legend
for Z_eff, D0, I_re0, do in zip(Z_eff_list, D0_bra_list, I_re0_bra_list, do_bra_list):  # For loop to plot best I_re for every a_D with describing legend
    legend.append('$Z_{eff}=$' + str(round(Z_eff,1)) + ', $D_0=$' + str("%.3f" % D0) + ', \n$I_{re0}=$' + str(I_re0))  # Legend describing a_D, D0 and I_re0
    do.other.fluid.GammaAva.plot()               # Avalanche growth rate GammaAva

# Plot details
plt.xlim(-2.5, 8.5)
plt.ylim(bottom=0)
plt.xlabel('t [s]')
plt.ylabel('$\Gamma_{ava}$')
plt.title('$\Gamma_{ava}$ - Finding $I_{re0}$ and $D_0$ for different $Z_{eff}$')
plt.legend(legend,loc='upper left')
manager = plt.get_current_fig_manager()
manager.window.maximize()
#plt.savefig('pr_26mars/GammaAva__different__a_D.png')
plt.show()
