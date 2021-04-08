import numpy as np
import sys
from scipy import integrate
import matplotlib.pyplot as plt

sys.path.append('../../py/')

from DREAM.DREAMOutput import DREAMOutput


# Load data
Z_eff_list=np.loadtxt('pr_8apr/time_plots/Z_eff_list.txt', dtype=float)
D0_bra_list=np.loadtxt('pr_8apr/time_plots/D0_bra_list.txt', dtype=float)
I_re0_bra_list=np.loadtxt('pr_8apr/time_plots/I_re0_bra_list.txt', dtype=float)

do_bra_list=[]

n_re_list=[]
I_p_list=[]
E_c_eff_list=[]
E_c_free_list=[]
GammaAva_list=[]
for Z_eff in Z_eff_list:
    n_re_list.append(np.loadtxt('pr_8apr/time_plots/data/n_re_Z_eff='+str(round(Z_eff,1))+'.txt', dtype=float))
    I_p_list.append(np.loadtxt('pr_8apr/time_plots/data/I_p_Z_eff=' + str(round(Z_eff, 1)) + '.txt', dtype=float))
    E_c_eff_list.append(np.loadtxt('pr_8apr/time_plots/data/E_c_eff_Z_eff=' + str(round(Z_eff, 1)) + '.txt', dtype=float))
    E_c_free_list.append(np.loadtxt('pr_8apr/time_plots/data/E_c_free_Z_eff=' + str(round(Z_eff, 1)) + '.txt', dtype=float))
    GammaAva_list.append(np.loadtxt('pr_8apr/time_plots/data/GammaAva_Z_eff=' + str(round(Z_eff, 1)) + '.txt', dtype=float))

# Plot best time-dependant parameters
legend_a_D = []      # List for a_D to be shown in legend
for a_D in a_D_for:  # Filling list with a_D-sweep-values
    legend_a_D.append(a_D)
legend_a_D[2]='2/3'  # Change 0.666666666667 to 2/3

# Figure 1
legend = ['$I_{p,deVries}$']  # Total legend with I_p,deVries as firt element
plt.plot(t, I_p)              # Plots I_p,deVries as first graph
for Z_eff, D0, I_re0, n_re_loop in zip(Z_eff_list, D0_bra_list, I_re0_bra_list, n_re_list):  # For loop to plot best I_re for every a_D with describing legend
    legend.append('$Z_{eff}=$' + str(round(Z_eff,1)) + ', $D_0=$' + str("%.3f" % D0) + ', \n$I_{re0}=$' + str(I_re0))  # Legend describing a_D, D0 and I_re0

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
for Z_eff, D0, I_re0, n_re_loop, I_pD in zip(Z_eff_list, D0_bra_list, I_re0_bra_list, n_re_list, I_p_list:  # For loop to plot best I_re for every a_D with describing legend
    legend.append('$I_p$: ' + '$Z_{eff}=$' + str(round(Z_eff,1)) + ', $D_0=$' + str("%.3f" % D0) + ', \n$I_{re0}=$' + str(I_re0))  # Legend describing a_D, D0 and I_re0
    tid = np.linspace(0, tMax, num=len(I_pD))  # Equivalent time steps for plot
    plt.plot(tid,I_pD)

    legend.append('$I_{re}$: ' + '$Z_{eff}=$' + str(round(Z_eff,1)) + ', $D_0=$' + str("%.3f" % D0) + ', \n$I_{re0}=$' + str(I_re0))  # Legend describing a_D, D0 and I_re0
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
for Z_eff, D0, I_re0, I_pD in zip(Z_eff_list, D0_bra_list, I_re0_bra_list, I_p_list):  # For loop to plot best I_re for every a_D with describing legend
    legend.append('$Z_{eff}=$' + str(round(Z_eff,1)) + ', $D_0=$' + str("%.3f" % D0) + ', \n$I_{re0}=$' + str(I_re0))  # Legend describing a_D, D0 and I_re0
    tid = np.linspace(0, tMax, num=len(I_pD))  # Equivalent time steps for plot
    plt.plot(tid, I_pD)

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
for Z_eff, D0, I_re0, E_c_eff, E_c_free in zip(Z_eff_list, D0_bra_list, I_re0_bra_list, E_c_eff_list, E_c_free_list):  # For loop to plot best I_re for every a_D with describing legend
    legend.append('$Z_{eff}=$' + str(round(Z_eff,1)) + ', $D_0=$' + str("%.3f" % D0) + ', \n$I_{re0}=$' + str(I_re0))  # Legend describing a_D, D0 and I_re0
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
for Z_eff, D0, I_re0, GammaAva in zip(Z_eff_list, D0_bra_list, I_re0_bra_list, GammaAva_list):  # For loop to plot best I_re for every a_D with describing legend
    legend.append('$Z_{eff}=$' + str(round(Z_eff,1)) + ', $D_0=$' + str("%.3f" % D0) + ', \n$I_{re0}=$' + str(I_re0))  # Legend describing a_D, D0 and I_re0
    tid = np.linspace(0, tMax, num=len(GammaAva))  # Equivalent time steps for plot
    plt.plot(tid,GammaAva)

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
