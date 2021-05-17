import numpy as np
import sys
import matplotlib.pyplot as plt
from selfconsistent_radial_functions import generate_current_profile_fun
from selfconsistent_radial_functions import generate_current_profile_fun_T
from selfconsistent_radial_functions import setupSelfconsistentD
from scipy.interpolate import interp1d

sys.path.append('../../py/')

from DREAM import runiface


def V_loop_wall_fun(V_f,V_max,t,t_f,tMax):
    n_f=int(t_f*20)
    index_f = np.linspace(0, n_f - 1, num=n_f)
    V = np.linspace(V_max, V_max, num= n_f + 1)
    tid = np.linspace(0, tMax, num= n_f + 1)
    tid_f = np.linspace(tMax - t_f, tMax, num=n_f)
    for i in index_f:
        V[int(-i - 1)] = V_f
        tid[int(-i - 1)] = tid_f[int(-i - 1)]
    V_s = interp1d(tid, V, kind='cubic')
    V_loop_wall = V_s(t)
    return V_loop_wall

#### Physical parameters ####

# Universal constants
e   = 1.60217662e-19  # Elementary charge [C]
c   = 299792458       # Speed of light [m/s]
m_e = 9.10938356e-31  # Electron mass [kg]

# Assumed constants in de Vries
A_c     = 3                     # Plasma cross section [m^2]
R_0     = 3                     # Major radius [m]
r_0     = np.sqrt(A_c/np.pi)    # Minor radius [m]
B       = 2.4                   # Magnetic field [T]
D       = 1*r_0**2              # Heat diffusion [m^2/s]
T_start = 0.1                   # Start factor for temperature

# Time and radial parameters
tMax_initial    = 0.1                           # Simulation time to find initial values [s]
Nt_eq           = 1000                          # Number of time steps to find initial values
tMax            = 14                            # Simulation time to generate results [s]
Nt              = 10000                         # Time steps to generate results
t               = np.linspace(0,tMax,num=Nt+1)  # Time vector for time depending data
t_f             = 4                             # Duration of final equilibrium stage [s]
Nr              = 25                            # Number of radial steps
radie           = np.linspace(0,r_0,Nr)         # Radial vector for radius depending data
current_profile = 1-(radie/r_0)**2              # Desired current profile

#Ions
Z_D = 1                      # Charge number deuterium
Z_B = 4                      # Charge number beryllium
a_D = 0.99                   # Proportion of ions that are deuterium
a_B = 1 - a_D                # Proportion of ions that are beryllium
n_e = 5e18                   # Total electron density
n_i = n_e/(a_D*Z_D+a_B*Z_B)  # Total ion density
n_D = a_D * n_i              # Deuterium density
n_B = a_B * n_i              # Beryllium density

# Varying parameters
Ip_final_vec = np.linspace(0.5e6,5e6,num=10)  # Final equilibrium current [A]

T_max_0   = 1500  # Upper limit for centre-temperature sweep to find equilibrium [eV]
T_min_0   = 1400  # Lower limit for centre-temperature sweep to find equilibrium [eV]
T_max_r_0 = 1450  # Upper limit for edge-temperature sweep to find equilibrium [eV]
T_min_r_0 = 1350  # Lower limit for edge-temperature sweep to find equilibrium [eV]

Imax = 1e5     # Upper limit for current sweep [A]
Imin = 0       # Lower limit for current sweep [A]
I_eq1 = 45868  # Guess for current of initial equilibrium [A]

tol = 1e-4  # Tolerance

# Vectors for saving data
V_f_vec      = []
V_max_vec    = []
faktor_f_vec = []
faktor_i_vec = []

for Ip_final in Ip_final_vec:
    Ip_final_plot = t * 0 + Ip_final  # For plotting I_pf
    
    do,Teq,Eeq = generate_current_profile_fun(Ip_final,current_profile,radie,T_max_0,T_min_0,T_max_r_0,T_min_r_0, tMax_initial, Nt_eq, n_D, n_B, B, r_0, R_0, Nr, D)

    # Temperature limit for next Ip_final
    T_max_0   = Teq[0]*3 
    T_min_0   = Teq[0]*1  
    T_max_r_0 = Teq[-1]*3
    T_min_r_0 = Teq[-1]*1
    
    T_initial = T_start*Teq  # Initial temperature [eV]
    
    do, T_r, I_r, E_r = generate_current_profile_fun_T(I_eq1,current_profile,radie,T_initial,Imax,Imin, tMax_initial, Nt_eq, n_D, n_B, B, r_0, R_0, Nr, D)

    faktor_f_max = 30                             # Upper limit for voltage factor
    faktor_f_min = 3                              # Lower limit for voltage factor
    faktor_f     = (faktor_f_max+faktor_f_min)/2  # Factor guess

    while abs(faktor_f_max-faktor_f_min) > tol:  # Finding best factor for V_loop,r0,f that gives constant current in the end
        # Wall loop voltage settings
        V_f         = 2 * np.pi * R_0 * faktor_f * Eeq[-1]   # Final voltage V_loop,r0,f guess [V]
        V_loop_min  = 0                                      # Lower limit for V_loop,r0,i [V]
        V_loop_max  = 0.3                                    # Upper limit for V_loop,r0,i [V]
        V_max       = (V_loop_max + V_loop_min) / 2          # Guess for V_loop,r0,i [V]
        V_loop_wall = V_loop_wall_fun(V_f,V_max,t,t_f,tMax)  # V_loop_r0 as a function of time [V]

        while abs(V_loop_max - V_loop_min) > tol:  # Finding best V_loop,r0,i that gives correct I_pf
            ds = setupSelfconsistentD(E_r, T_r, V_loop_wall, t, Z_D, Z_B, n_D, n_B, B, r_0, R_0, Nr, tMax, Nt_eq, D)
            do = runiface(ds, quiet=False)
            if do.eqsys.I_p[-1] >= Ip_final:
                V_loop_max = V_max
            elif do.eqsys.I_p[-1] < Ip_final:
                V_loop_min = V_max
            V_max       = (V_loop_min + V_loop_max) / 2             # Best V_loop,r0,i [V]
            V_loop_wall = V_loop_wall_fun(V_f, V_max, t, t_f, tMax)

        if do.eqsys.I_p[-1] >= do.eqsys.I_p[-int(Nt_eq / tMax*2)]:
            faktor_f_max=faktor_f
        elif do.eqsys.I_p[-1] < do.eqsys.I_p[-int(Nt_eq / tMax*2)]:
            faktor_f_min=faktor_f
        faktor_f = (faktor_f_max + faktor_f_min) / 2

    V_f         = 2 * np.pi * R_0 * faktor_f * Eeq[-1]   # Best V_loop,r0,f [V]
    V_loop_wall = V_loop_wall_fun( V_f, V_max, t, t_f, tMax)

    V_f_vec.append(V_f)
    V_max_vec.append(V_max)
    faktor_f_vec.append(faktor_f)

    # Final simulation
    ds = setupSelfconsistentD(E_r, T_r, V_loop_wall, t, Z_D, Z_B, n_D, n_B, B, r_0, R_0, Nr, tMax, Nt,D)
    do = runiface(ds, quiet=False)

    ## Save data for DREAM-parameters plots and in files
    t=np.linspace(0,tMax,Nt)
    ax = do.eqsys.I_p.plot()
    ax2 = plt.plot(t, np.linspace(1, 1, Nt) * Ip_final)  # Want to compare to plasma current at ITER
    plt.legend(['I_p', 'ITER I_p'])
    #plt.xlim(0, tMax_initial)
    plt.ylim(0, Ip_final*1.25)
    plt.savefig('sc_8maj/High_res/I_p__I_p_wish='+str(Ip_final/1e6)+'.png')
    np.savetxt('sc_8maj/High_res/I_p__I_p_wish=' + str(Ip_final / 1e6) + '.txt', do.eqsys.I_p[:], delimiter=',')
    plt.close()
    #plt.show()

    ax = do.eqsys.T_cold.plot()
    plt.title('$T_{cold}$')
    plt.savefig('sc_8maj/High_res/T_cold__I_p_wish=' +str(Ip_final/1e6)+ '.png')
    np.savetxt('sc_8maj/High_res/T_cold__I_p_wish=' + str(Ip_final / 1e6) + '.txt', do.eqsys.T_cold[:], delimiter=',')
    plt.close()
    #plt.show()

    ax = do.other.fluid.Tcold_ohmic.plot()
    plt.title('$T_{cold,ohmic}$')
    plt.savefig('sc_8maj/High_res/T_ohmic__I_p_wish=' + str(Ip_final / 1e6) + '.png')
    np.savetxt('sc_8maj/High_res/T_ohmic__I_p_wish=' + str(Ip_final / 1e6) + '.txt',do.eqsys.T_cold[:], delimiter=',')
    plt.close()
    #plt.show()

    ax = do.eqsys.E_field.plot()
    plt.title('$E_{field}$')
    plt.savefig('sc_8maj/High_res/E__I_p_wish=' +str(Ip_final/1e6)+ '.png')
    np.savetxt('sc_8maj/High_res/E__I_p_wish=' + str(Ip_final / 1e6) + '.txt', do.eqsys.E_field[:],delimiter=',')
    plt.close()
    #plt.show()

    ax = do.eqsys.n_re.plot()
    plt.title('$n_{re}$')
    plt.savefig('sc_8maj/High_res/n_re__I_p_wish=' +str(Ip_final/1e6)+ '.png')
    np.savetxt('sc_8maj/High_res/n_re__I_p_wish=' + str(Ip_final / 1e6) + '.txt', do.eqsys.n_re[:],delimiter=',')
    plt.close()
    #plt.show()

    ax = do.eqsys.j_ohm.plot()
    plt.title('$j_{ohm}$')
    plt.savefig('sc_8maj/High_res/j_ohm__I_p_wish=' + str(Ip_final / 1e6) + '.png')
    np.savetxt('sc_8maj/High_res/j_ohm__I_p_wish=' + str(Ip_final / 1e6) + '.txt', do.eqsys.j_ohm[:], delimiter=',')
    plt.close()
    #plt.show()

    np.savetxt('sc_8maj/High_res/gammaDreicer__I_p_wish=' + str(Ip_final / 1e6) + '.txt', do.other.fluid.gammaDreicer[:],delimiter=',')
    np.savetxt('sc_8maj/High_res/gammaAva__I_p_wish=' + str(Ip_final / 1e6) + '.txt',do.other.fluid.GammaAva[:] * do.eqsys.n_re[1:, :],delimiter=',')
    np.savetxt('sc_8maj/High_res/Eceff__I_p_wish=' + str(Ip_final / 1e6) + '.txt',
               do.other.fluid.Eceff[:], delimiter=',')
    np.savetxt('sc_8maj/High_res/Ecfree__I_p_wish=' + str(Ip_final / 1e6) + '.txt',
               do.other.fluid.Ecfree[:], delimiter=',')
    np.savetxt('sc_8maj/High_res/Ectot__I_p_wish=' + str(Ip_final / 1e6) + '.txt',
               do.other.fluid.Ectot[:], delimiter=',')
    np.savetxt('sc_8maj/High_res/EDreic__I_p_wish=' + str(Ip_final / 1e6) + '.txt',
               do.other.fluid.EDreic[:], delimiter=',')


## Save data from while-loops as plots and in files
faktor_f_vec=np.array(faktor_f_vec)
V_f_vec=np.array(V_f_vec)
V_max_vec=np.array(V_max_vec)
plt.plot(Ip_final_vec,V_f_vec)
plt.plot(Ip_final_vec,V_max_vec)
legend=['$V_{loop,r_0,final}$','$V_{loop,r_0,initial}$']
plt.legend(legend,loc='upper left')
plt.title('$V_{loop,wall}$ as a function of $I_p$ at equilibrium')
plt.xlabel('$I_p$')
plt.ylabel('$V_{loop,wall}$')
plt.savefig('sc_8maj/High_res/V_loop_wall(I_p).png')
plt.close()
plt.plot(Ip_final_vec,faktor_f_vec)
plt.xlabel('$I_p$')
plt.ylabel('faktor')
plt.savefig('sc_8maj/High_res/faktor_f.png')
plt.close()
np.savetxt('sc_8maj/High_res/V_loop_final.txt', V_f_vec, delimiter=',')
np.savetxt('sc_8maj/High_res/I_p_wish.txt', Ip_final_vec, delimiter=',')
np.savetxt('sc_8maj/High_res/faktor_f.txt', faktor_f_vec, delimiter=',')

