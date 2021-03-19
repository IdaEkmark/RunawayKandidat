import numpy as np
from numpy.linalg import norm
import sys
import matplotlib
import matplotlib.pyplot as plt


sys.path.append('../../py/')

from DREAM import runiface
from DREAM.DREAMSettings import DREAMSettings
from DREAM.DREAMOutput import DREAMOutput
import DREAM.Settings.Equations.DistributionFunction as DistFunc
import DREAM.Settings.Equations.IonSpecies as Ions
import DREAM.Settings.Equations.RunawayElectrons as Runaways
import DREAM.Settings.Solver as Solver
import DREAM.Settings.CollisionHandler as Collisions
import DREAM.Settings.Equations.ElectricField as ElectricField
import DREAM.Settings.Equations.RunawayElectrons as RE
import DREAM.Settings.Equations.HotElectronDistribution as FHot
import DREAM.Settings.Equations.ColdElectronTemperature as T_cold
import DREAM.Settings.TimeStepper as TimeStep
import DREAM.Settings.TransportSettings as Transport
import DREAM.Settings.Equations.ColdElectrons as ColdElectrons
from jProfSim import simProf
from nRE_Partition import n_re_partition_jprof









#######################################################################################################################
                                                #PARAMETERS#
#######################################################################################################################

# Grid parameters
pMax = 1        # maximum momentum in units of m_e*c ORIG 1

tMax = 10        # simulation time in seconds ORIG 1e-3
Nt = 1000       # number of time steps ORIG 20
Nr = 1          # number of radial grid points

# Constants
eq = -1.602e-19                     # Charge of electron [C]
eps = 8.854e-12                     # Vacuum permitivity [F/m]
lnA = 18                            # Coulomb logarithm
me = 9.109e-31                      # Electron mass [kg]
c = 2.99792458e8                    # Speed of light [m/s]
r0=3                                # Major radius
Ac=3                                # Plasma cross section
a = np.sqrt(Ac/np.pi)               # Minor radius [m]
A0 = 0                              # Advection
D0 = ((a**2)/4)/18#0.03                           # Diffusion
print(D0)

print(D0)
Ltor = 5e-6                         # Inductance [H]



# Varying parameters
t = np.linspace(0,tMax,num=Nt+1)            # Time vector
Ip = (0.9*np.exp(-1/(10*(t+0.8490))))*1e6     # Total plasma current
dIpdt = np.gradient(Ip,t)                   # Plasmaströmmens derivata
Vloop = 0.08+0.77*np.exp(-t)                # Pålagd yttre spänning
n_e = (0.287*np.exp(-0.8*t**2)+0.023)*1e19      # Kall densitet
T_e = (0.052+0.748*np.exp(-0.85*t))*1e3            # Kall temperatur
vth = np.sqrt(2 * T_e * -eq / me)           # Thermal velocity

# Initial values
I_re = Ip[0]*0.5#6.5e5                        # Runaway current A
n_re_init=I_re/(abs(eq)*c*Ac)       # Initial runaway density m^-3
re_share = I_re/Ip[0]               # Percentage of runaway current vs total current
print(re_share)
n_tot = n_e[0]+n_re_init

# Parameters calculated from circuit equation
Vind = Ltor*dIpdt           # Inductive voltage
Vres = Vloop - Vind         # Resistive voltage (which is felt by electrons)
E = Vres/(2*np.pi*r0)       # E(dIpdt,Vloop)


## Matrices & vectors to be filled ##
NB = 5
Np = 3
B_v_preset = np.linspace(2.4,10,NB)
E_v = np.zeros((Nt+1,Np))
Ip_v = np.zeros((Nt+1,Np))
I_re_v = np.zeros((Nt+1,Np))
II_re_v = np.zeros((Nt+1,Np))
kappa_v = np.zeros((Nt,Np))
purity_v = np.linspace(0.99,0.6,Np)
E_ceff_v = np.zeros((Nt,Np))
IpDiff_v = np.zeros((Np,NB))



for o in range(0,NB):
    B = B_v_preset[o]               # B-field [T]

    for i in range(0,Np):

        #######################################################################################################################
                                                # BZ-SWEEP PARAMETERS#
        #######################################################################################################################
        ps = 1                       # Starting share of deuterium
        purity = purity_v[i]          # Deuterium share of ion density in plasma
        n_ions = n_tot/(4-3*purity)     # Total density of ions with given purity (only if the ions are D and Be)
                                        # Derived from n_tot = n_Be*4 + n_D, n_ions = n_Be + n_D, and n_D = n_ions*purity
        nD = n_ions * purity            # Deuterium density
        nBe = n_ions - nD               # Beryllium density (n_tot must be consistent)


        #######################################################################################################################
                                                            #SETUP#
        #######################################################################################################################

        ds = DREAMSettings()
        # ds.collisions.collfreq_type = Collisions.COLLFREQ_TYPE_COMPLETELY_SCREENED
        ds.collisions.collfreq_type = Collisions.COLLFREQ_TYPE_PARTIALLY_SCREENED

        # Set up radial grid and parameters
        ds.radialgrid.setB0(B)
        ds.radialgrid.setMinorRadius(a)
        ds.radialgrid.setWallRadius(a)
        ds.radialgrid.setMajorRadius(r0)
        ds.radialgrid.setNr(Nr)

        # Set E, T, n_cold
        ds.eqsys.T_cold.setPrescribedData(T_e, times=t)
        ds.eqsys.n_cold.setType(ColdElectrons.TYPE_SELFCONSISTENT)  # Ensures n_cold = n_free - n_re - n_hot (n_hot=0 when hottailgrid is off)
        #ds.eqsys.n_cold.setPrescribedData(density=n_e, times=t)

        # Set E-field
        ds.eqsys.E_field.setType(ElectricField.TYPE_SELFCONSISTENT)     # Self consistent
        ds.eqsys.E_field.setInitialProfile(efield=E[0])                    # The initial value is given by the de Vries-like computation of E.
        ds.eqsys.E_field.setBoundaryCondition(bctype=1, V_loop_wall=Vloop, times=t)

        # Set ions
        ds.eqsys.n_i.addIon(name='D', Z=1, iontype=Ions.IONS_PRESCRIBED_FULLY_IONIZED, n=nD)
        ds.eqsys.n_i.addIon(name='Be', Z=4, iontype=Ions.IONS_PRESCRIBED_FULLY_IONIZED, n=nBe)    #n_tot = n_cold + n_re
                                                                                                #Vi vill att nB*4 + nD = n_tot

        # Disable runaway grid
        ds.runawaygrid.setEnabled(False)

        # Set solver type
        ds.solver.setType(Solver.LINEAR_IMPLICIT)  # semi-implicit time stepping
        ds.solver.preconditioner.setEnabled(False)

        # include otherquantities to save to output
        ds.other.include('fluid')

        # Set time stepper
        ds.timestep.setTmax(tMax)
        ds.timestep.setNt(Nt)
        ds.output.setTiming(stdout=True, file=True)

        # Extra de Vries settings
        ds.hottailgrid.setEnabled(False)

        #######################################################################################################################
                                                        #ENABLING DIFFUSION#
        #######################################################################################################################
        ds.eqsys.n_re.setAvalanche(avalanche=Runaways.AVALANCHE_MODE_FLUID_HESSLOW)
        ds.eqsys.n_re.setInitialProfile(density=n_re_init)
        # Aktiverar Diffusion
        #t = np.linspace(0,tMax,Nt)
        #r = np.linspace(0,r0,Nr)
        #A = np.array([[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]])
        #D = np.array([[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]])
        #A = np.array([(t,r)])
        #D = np.array([(t,r)])

        ds.eqsys.n_re.transport.prescribeAdvection(ar=A0)#, t=t, r=r)
        ds.eqsys.n_re.transport.prescribeDiffusion(drr=D0)#,t=t, r=r)
        ds.eqsys.n_re.transport.setBoundaryCondition(Transport.BC_F_0)
        #ds.eqsys.E_field.setPrescribedData(efield=E[-1,:], radius=r_out)
        #print(E[-1,:])
        ds.eqsys.n_re.setDreicer(Runaways.DREICER_RATE_DISABLED)
        do3 = runiface(ds,'deVries_Output.h5',quiet=False)

        #######################################################################################################################
                                                            #PLOTS#
        #######################################################################################################################

        # Plots
        plot_n_re_heatmap = False
        plot_n_re = False
        plot_Ip = False


        if plot_n_re_heatmap == True:
            n_re_end = do3.eqsys.n_re[:]  # Picking out the resulting nRE
            plt.figure(1)
            plt.imshow(n_re_end[:], cmap = 'winter', interpolation = 'nearest')
            plt.title( "Runaway Density Heat Map" )
            plt.ylabel('Time [index]')
            plt.xlabel('Radius [index]')
            plt.colorbar()
            plt.show()

        if plot_n_re == True:
            Ip = do3.eqsys.I_p[:]
            plt.figure(2)
            plt.plot(do3.grid.t, do3.eqsys.n_re[:])
            plt.title("Runaway Density")
            plt.ylabel('[1/m^3]')
            plt.xlabel('Time [s]')
            plt.show()

        if plot_Ip == True:
            Ip = do3.eqsys.I_p[:]
            plt.figure(3)
            plt.plot(do3.grid.t, Ip*1e-6,label='I_p',color='green')
            plt.plot(do3.grid.t, do3.eqsys.n_re[:]*c*Ac*-eq*1e-6,label='I_re', color='purple')
            plt.legend()
            plt.title("Plasma Current" )
            plt.ylabel('Ip [MA]')
            plt.xlabel('Time [s]')
            plt.show()


        #######################################################################################################################
                                        #SAVING ONTO VECTORS/MATRICES#
        #######################################################################################################################
        Ip_v[:, i] = (do3.eqsys.I_p[:]).T
        I_re_v[:, i] = (do3.eqsys.n_re[:]*c*Ac*-eq).T
        purity_v[i] = purity
        kappa_v[:, i] = ((do3.other.fluid.Eceff[:]) / (do3.other.fluid.Ecfree[:])).T
        E_v[:, i] = (do3.eqsys.E_field[:]).T
        E_ceff_v[:, i] = (do3.other.fluid.Eceff[:]).T
        II_re_v[:,i] = (abs(do3.eqsys.j_re[:]*(a**2)*np.pi)).T
        print(np.sum((do3.eqsys.I_p[:] - Ip[:])**2))
        IpDiff_v[i,o] = np.sum((do3.eqsys.I_p[:] - Ip[:])**2)

    #######################################################################################################################
                                                    #BZ PLOTS#
    #######################################################################################################################

    plot_Ip_vs_I_re = True
    plot_kappa = True
    plot_kappaE_vs_E = False

    if plot_Ip_vs_I_re == True:
        Colorgradient = np.linspace(1, 0.5, Np)
        for i in range(0,Np):
            plt.figure(1)
            plt.plot(do3.grid.t, Ip_v[:,i].T, color=Colorgradient[i]*np.array([1,0,0]))
            plt.plot(do3.grid.t, I_re_v[:,i].T, color=Colorgradient[i]*np.array([0,1,0]))
            #plt.plot(do3.grid.t, II_re_v[:, i].T, color=Colorgradient[i] * np.array([0, 0, 1]))
            string = "a"
            plt.title("B = " + str(B) + ", and purity [" + str(purity_v[0]) + "," + str(purity_v[-1]) + "]. \n Darker means more impurity.")
            plt.ylabel('Ip [A]')
            plt.xlabel('Time [s]')
            plt.figlegend(('Plasmaström', 'Skenande-elektron-ström'),
    loc='upper right', bbox_to_anchor=(0.9, 0.88))
        x = plt.plot(t,Ip,color=[1,0.8,0])
        plt.figlegend(x,"Prescribed Ip", loc='lower left')
        plt.show()

    if plot_kappa == True:
        Colorgradient = np.linspace(1,0.1,Np)
        for i in range(0,Np):
            plt.figure(2)
            plt.plot(do3.grid.t[1:], kappa_v[:,i],color=Colorgradient[i]*np.array([1,0,0]))
            string = "a"
            plt.title("\kappa, B = " + str(B) + ", and purity [" + str(purity_v[0]) + "," + str(purity_v[-1]) + "]. \n Darker means more impurity (beryllium).")
            plt.ylabel('kappa')
            plt.xlabel('Time [s]')
        plt.show()

    if plot_kappaE_vs_E == True:
        Colorgradient = np.linspace(1,0.2,Np)
        print(kappa_v[:,1])
        for i in range(0,Np):
            plt.figure(3)
            plt.plot(do3.grid.t,abs(E_v[:,i]),color=Colorgradient[i]*np.array([1,0,0]))
            plt.plot(do3.grid.t[1:], abs(kappa_v[:,i]*E_ceff_v[:,i]),color=Colorgradient[i]*np.array([0,0,1]))
            plt.title("kappa*Eceff (blue) vs E (red)")
            plt.ylabel('kappa')
            plt.xlabel('Time [s]')
        plt.show()
