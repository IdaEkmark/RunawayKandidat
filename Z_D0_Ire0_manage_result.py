import numpy as np
import sys
from scipy import integrate
import matplotlib.pyplot as plt

# Save data
Z_eff_list=np.loadtxt('pr_6apr/Z_D0_Ire0/Z_eff_list.txt', dtype=float)
D0_bra_list=np.loadtxt('pr_6apr/Z_D0_Ire0/D0_bra_list.txt', dtype=float)
I_re0_bra_list=np.loadtxt('pr_6apr/Z_D0_Ire0/I_re0_bra_list.txt', dtype=float)

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