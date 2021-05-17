# Code to reproduce the result in de Vries study.
# The code finds the initial runaway current I_re0 and diffusion D0 that results in a
# stable runaway current after 5.1 s, which dominated I_pV for effective charges
# Z_eff\in[1,2.0].
# It will plot the I_re0(Z_eff), I_re0(D0) and D0(Z_eff)

import numpy as np
#import sys
#from scipy import integrate
import matplotlib.pyplot as plt
from de_Vries_DREAM_settings import run_DREAM

#### Physical parameters ####

# Universal constants
e   = 1.60217662e-19  # Elementary charge [C]
c   = 299792458       # Speed of light [m/s]
m_e = 9.10938356e-31  # Electron mass [kg]

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

# Experimental data from de Vries (approximated)
I_pV   = 0.9*np.exp(-1/(10*(t+0.8490)))*1e6    # Time dependent plasma current [A]
V_loop = 0.08+0.77*np.exp(-t)                  # Time dependent external torodial loop voltage [V]
n_e    = (0.287*np.exp(-0.8*t**2)+0.023)*1e19  # Time dependent density of thermal electrons \approx total density of electrons  [m^-3]
T_e    = (0.052+0.748*np.exp(-0.85*t))*1e3     # Time dependent thermal temperature [eV]

# Electric field calculated as in de Vries article
E_loop = V_loop/(2*np.pi*R_0)  # Time dependent electric field strength [V/m]

# Lists for saving the best parameters
D0_bra_list    = []  # List to save the best D0 for each a_D
I_re0_bra_list = []  # List to save the best I_re0 for each a_D
Z_eff_list     = []  # List to save the Z_eff
do_bra_list    = []  # List to save the best DREAM-output-object for each a_D

# Analyzing the runaway current for different proportions of deuterium and beryllium
a_D_for=np.linspace(0.7,1,num=101)  # Proportion of ions that are deuterium
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

    r   = np.linspace(0,r_0,num=Nr+1)  # Radial coordinates for diffusion

    # Sweep detalis
    D0min    = 0.013                   # Lower limit of D0 sweep
    D0max    = 0.023                   # Upper limit of D0 swep
    D0num    = 101                     # Number of D0-steps
    I_re0min = 5.4e5                   # Lower limit of I_re0 sweep
    I_re0max = 6.1e5                   # Upper limit of I_re0 sweep
    I_re0num = 101                     # Number of I_re0-steps
    index    = np.linspace(1,1,num=1)  # Number of iterations
    I_re_for = 6e5                     # Initial runaway current [A]
    for i in index:  # Iterating to find best D0 from previous best I_re0 I
        n_re0 = I_re_for/(e*c*A_c)  # Initial density of runaway electrons [m^-3]

        # Find D0 for most constant current
        D0_loop     = np.linspace(D0min,D0max,num=D0num)  # D0-sweep-values
        dos_loop_D0 = []                                  # List to save DREAM-output-object
        for Ds in D0_loop:  # Running DREAM for every D0-sweep-value
            ds_loop_D0, do_loop_D0 = run_DREAM(E_loop,T_e,'D',Z_D,n_D,'B',Z_B,n_B,n_re0,B,r_0,R_0,A0,Ds,r,Nr,t,tMax,Nt)
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
        D0_loop = D0_min_loop  # Save value of D0 with minimum variance of I_pV

        # Find I_re0 which gives I_re=I_p in the end
        I_re          = np.linspace(I_re0min, I_re0max,num=I_re0num)  # D0-sweep-values
        n_re0         = I_re/(e*c*A_c)                                # Initial density of runaway electrons [m^-3]
        dos_loop_Ire0 = []                                            # List to save DREAM-output-object

        for n_re0s in n_re0:  # Running DREAM for every I_re0-sweep-value
            ds_loop_Ire0, do_loop_Ire0 =run_DREAM(E_loop,T_e,'D',Z_D,n_D,'B',Z_B,n_B,n_re0s,B,r_0,R_0,A0,D0_loop,r,Nr,t,tMax,Nt)
            dos_loop_Ire0.append(do_loop_Ire0)  # Saving each DREAM-output-object

        min_diff_loop = 1e20  # Variable for difference between I_re and I_pV
        i_loop_Ire0   = 0     # Index of iteration

        for do in dos_loop_Ire0:  # Going through the DREAM-output-objects to find the one I_re0 wich dominates I_pV in the end
            n_re_do  = do.eqsys.n_re[5500:].T[0]         # n_re after 4.7 s
            I_re_do  = n_re_do * e * c * A_c             # I_re after 4.7 s
            I_pV_do   = I_pV[5500:]                        # I_pV after 4.7 s
            diff     = I_pV_do-I_re_do                    # Difference between I_re and I_pV after 4.7 s
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
np.savetxt('pr_12maj/Test2/Z_eff_list.txt', Z_eff_list, delimiter=',')
np.savetxt('pr_12maj/Test2/D0_bra_list.txt', D0_bra_list, delimiter=',')
np.savetxt('pr_12maj/Test2/I_re0_bra_list.txt', I_re0_bra_list, delimiter=',')

# Plot D0 as function of Z_eff and I_re0 as function of D0 and as function of Z_eff
plt.plot(Z_eff_list,D0_bra_list)
plt.xlabel('$Z_{eff}$')
plt.ylabel('$D_0$')
plt.title('$D_0$ as a function of $Z_{eff}$')
manager = plt.get_current_fig_manager()
manager.window.maximize()
#plt.savefig('pr_26mars/I_re__different__a_D.png')
plt.show()

plt.plot(D0_bra_list,I_re0_bra_list)
plt.xlabel('$D_0$')
plt.ylabel('$I_{re0}$')
plt.title('$I_{re0}$ as a function of $D_0$')
manager = plt.get_current_fig_manager()
manager.window.maximize()
#plt.savefig('pr_26mars/I_re__different__a_D.png')
plt.show()

plt.plot(Z_eff_list,I_re0_bra_list)
plt.xlabel('$Z_{eff}$')
plt.ylabel('$I_{re0}$')
plt.title('$I_{re0}$ as a function of $Z_{eff}$')
manager = plt.get_current_fig_manager()
manager.window.maximize()
#plt.savefig('pr_26mars/I_re__different__a_D.png')
plt.show()
