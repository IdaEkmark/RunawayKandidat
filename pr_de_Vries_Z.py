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
from pr_DREAM_run import run_DREAM

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
E_loop = V_loop/(2*np.pi*r_0)  # Time dependent electric field strength [V/m]

# Lists for saving the best parameters
D0_bra_list    = []  # List to save the best D0 for each a_D
I_re0_bra_list = []  # List to save the best I_re0 for each a_D
Z_eff_list     = []
do_bra_list    = []  # List to save the best DREAM-output-object for each a_D

# Analyzing the runaway current for different proportions of deuterium and beryllium
a_D_for=np.array([0.5,0.6,2/3,0.70,0.8,0.9,1]) # Proportion of ions that are deuterium
for a_D in a_D_for:
    Z_D     = 1                      # Atomic number of deuterium
    Z_B     = 4                      # Atomic number of beryllium
    a_B     = 1-a_D                  # Proportion of ions that are beryllium
    n_tot   = n_e/(Z_D*a_D+Z_B*a_B)  # Total ion density
    n_D_for = a_D*n_tot              # Deuterium density
    n_B_for = a_B*n_tot              # Beryllium density
    Z_eff=a_D*Z_D+a_B*Z_B
    Z_eff_list.append(Z_eff)

    n_D = []  # Making array to define constant Deuterium-density on two radial coordinates
    n_B = []  # Making array to define constant Beryllium-density on two radial coordinates
    for elements in n_D_for:
        n_D.append([elements, elements])
    n_D = np.array(n_D)
    for elements in n_B_for:
        n_B.append([elements, elements])
    n_B = np.array(n_B)

    r   = np.linspace(0,a,num=Nr+1)  # Radial coordinates for diffusion

    # Sweep detalis
    D0min    = 0                       # Lower limit of D0 sweep
    D0max    = 0.03                    # Upper limit of D0 swep
    D0num    = 31                     # Number of D0-steps
    I_re0min = 5e5                     # Lower limit of I_re0 sweep
    I_re0max = 7e5                   # Upper limit of I_re0 sweep
    I_re0num = 201                 # Number of I_re0-steps
    index    = np.linspace(1,1,num=1)  # Number of iterations
    I_re_for = 6e5                     # Initial runaway current [A]
    for i in index:  # Iterating to find best D0 from previous best I_re0 I
        n_re0 = I_re_for/(e*c*A_c)  # Initial density of runaway electrons [m^-3]

        # Find D0 for most constant current
        D0_loop     = np.linspace(D0min,D0max,num=D0num) # D0-sweep-values
        dos_loop_D0 = []                                 # List to save DREAM-output-object
        for Ds in D0_loop:  # Running DREAM for every D0-sweep-value
            ds_loop_D0, do_loop_D0 = run_DREAM(E_loop,T_e,'D',Z_D,n_D,'B',Z_B,n_B,n_re0,B,a,r_0,A0,Ds,r,Nr,t,tMax,Nt)
            dos_loop_D0.append(do_loop_D0)  # Saving each DREAM-output-object

        min_I_p_var_loop = 1e20  # Variable for I_p to find minimum value of variance of I_p for each of the D0-sweep-values
        i_loop_D0        = 0     # Index of iteration
        D0_min_loop      = -1    # Variable for D0 to find minimum value of variance of I_p for each of the D0-sweep-values

        for do in dos_loop_D0:  # Going through the DREAM-output-objects to find the one D0 with most constant I_p in the end
            I_p_test_const = do.eqsys.I_p[5500:]     # I_p after 4.7 s
            I_p_var        = np.var(I_p_test_const)  # Variance of I_p after 4.7 s
            if I_p_var < min_I_p_var_loop:           # Find minimum variance of I_p
                min_I_p_var_loop = I_p_var           # Find value of I_p with minimum variance
                D0_min_loop = D0_loop[i_loop_D0]     # Find value of D0 with minimum variance of I_p
            i_loop_D0=i_loop_D0+1
        D0_loop = D0_min_loop  # Save value of D0 with minimum variance of I_p

        # Find I_re0 which gives I_re=I_p in the end
        I_re          = np.linspace(I_re0min, I_re0max,num=I_re0num)  # D0-sweep-values
        n_re0         = I_re/(e*c*A_c)                                # Initial density of runaway electrons [m^-3]
        dos_loop_Ire0 = []                                            # List to save DREAM-output-object

        for n_re0s in n_re0:  # Running DREAM for every I_re0-sweep-value
            ds_loop_Ire0, do_loop_Ire0 =run_DREAM(E_loop,T_e,'D',Z_D,n_D,'B',Z_B,n_B,n_re0s,B,a,r_0,A0,D0_loop,r,Nr,t,tMax,Nt)
            dos_loop_Ire0.append(do_loop_Ire0)  # Saving each DREAM-output-object

        min_diff_loop = 1e20  # Variable for difference between I_p and I_p,deVries to find minimum value of variance of I_p for each of the D0-sweep-values
        i_loop_Ire0   = 0     # Index of iteration

        for do in dos_loop_Ire0:  # Going through the DREAM-output-objects to find the one I_re0 wich dominates I_p,deVries in the end
            n_re_do  = do.eqsys.n_re[5500:].T[0]         # n_re after 4.7 s
            I_re_do  = n_re_do * e * c * A_c             # I_re after 4.7 s
            I_p_do   = I_p[5500:]                        # I_p,deVries after 4.7 s
            diff     = I_p_do-I_re_do                    # Difference between I_re and I_p,deVries after 4.7 s
            diff_sum = sum(diff**2)                      # Measurement of total difference to find minimum difference
            if diff_sum < min_diff_loop:                 # Find minimum difference between I_p,deVries and I_re
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
np.savetxt('pr_6apr/time_plots/Z_eff_list.txt', Z_eff_list, delimiter=',')
np.savetxt('pr_6apr/time_plots/D0_bra_list.txt', D0_bra_list, delimiter=',')
np.savetxt('pr_6apr/time_plots/I_re0_bra_list.txt', I_re0_bra_list, delimiter=',')
for do, Z_eff in zip(do_bra_list, Z_eff_list):
    do.save('output_de_Vries_Z='+str(Z_eff)+'.h5')

# Plot best time-dependant parameters
legend_a_D = []      # List for a_D to be shown in legend
for a_D in a_D_for:  # Filling list with a_D-sweep-values
    legend_a_D.append(a_D)
legend_a_D[2]='2/3'  # Change 0.666666666667 to 2/3

# Figure 1
legend = ['$I_{p,deVries}$']  # Total legend with I_p,deVries as firt element
plt.plot(t, I_p)              # Plots I_p,deVries as first graph
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
legend=['$I_{p,deVries}$']  # Total legend with I_p,deVries as firt element
plt.plot(t, I_p)            # Plots I_p,deVries as first graph
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
legend = ['$I_{p,deVries}$']  # Total legend with I_p,deVries as firt element
plt.plot(t, I_p)              # Plots I_p,deVries as first graph
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

#'''