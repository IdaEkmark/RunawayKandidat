# Test för att reproducera de Vries med parametrar passande de i de Vries (tror jag)

# Jag  har beräknat E(t) på samma sätt som de Vries gör (från I_tot och V_loop) samt satt
# T_cold och n_cold till en funktion lik den data de Vries använder. Stämmer det att
# T_cold är termala temperaturen och n_cold är termala elektron-densiteten? Ja
# T_cold är temperaturen för Maxwellianen. Främsta anledning till att evaluera avalanche och collisions etc
# n_cold är antalet fria elektroner i systemet, alla fria elektroner minus runaways

# Jag får inte samma I_p som de Vries I_tot utan den beter sig lite intressant men I_RE
# går mot I_p när tiden ökar så det är ju trevligt och kanske kan ganska enkelt förklaras
# med tanke på att det är olika ekvationer som används i simuleringarna
# Känslig för densitet och temperatur-utvecklingen

# Skenande-elektron-densiteten ökar också mer och mer och blir större än total-elektron-
# densiteten efter 150 s vilket inte är super - det vore bra med något som skulle kunna hindra
# n_re > n_tot, men det spelar egentligen inte jättestor roll då vi bara ska göra simuleringar
# i ca 10 s
# Antalet fria elektroner är konstant så om runaways försvinner uppkommer nya kalla elektroner

# Kan vara så att de Vries räknar ut E på ett annat sätt än jag har antagit från det han
# skrev, eftersom han senare skrev att han tar hänsyn till att Coulomb logaritmen ändras
# från 16 till 11 över tidens gång i beräkningen av E

# Bytte D0=0.03 mot D0=a**2/18\approx 0.05 men det blev bättre att ha D0=0.03 så jag bytte
# tillbaka

# Jag har ändrat Z från 1 till 2 i enighet med de Vries. För att få en någorlunda konstant
# I_p sänkte jag sedan B från 2.4 till 1.15 eftersom det funkade samt vi inte är säkra på vad
# de Vries har använt
#

# Till skillnad från de Vries får vi att I_re=I_p efter några sekunder oavsett om
# I_re(0)=5.5e5 A eller I_re(0)=6.5e5 A (de Vries får att detta endast stämmer för
# I_re(0)=6.5e5 A)

# Kappa parameter i de Vries, hoppar vissa effekter och löser dem med kappa
# DREAM hoppar inte dessa effekter
# Försöka få till den kappa som de Vries använder i vår kod
# Försöka hitta mer rimliga värden på kappa genom att analysera
# kappa = E_c_eff/E_c_free other.fluid, magnetfält och plasmats laddning påverkar
# Väte bas lägger till ytterliggare ett jonslag, Berrylium? Kol? Vi vill inte använda kol.
# Använd berrylium tror vi. De vries pratar om berrylium.
# Passa så att elektrondensiteten konstant genom att inte lägga tillför mycket Berryliym eller väte.

#testa selfconsistent E fält sätt randvillkor
import numpy as np
import sys
from scipy import integrate
import matplotlib.pyplot as plt
from generate_current_profile import generate_current_profile_fun

sys.path.append('../../py/')

from DREAM.DREAMSettings import DREAMSettings
from DREAM import runiface
import DREAM.Settings.Equations.ColdElectrons as ColdElectrons
import DREAM.Settings.Equations.IonSpecies as Ions
import DREAM.Settings.Equations.RunawayElectrons as Runaways
import DREAM.Settings.Solver as Solver
import DREAM.Settings.CollisionHandler as Collisions
import DREAM.Settings.TransportSettings as Transport
import DREAM.Settings.Equations.ElectricField as ElectricField

#### Physical parameters ####

# Universal constants
e   = 1.60217662e-19  # Elementary charge [C]
c   = 299792458       # Speed of light [m/s]
m_e = 9.10938356e-31  # Electron mass [kg]

# Assumed constants in de Vries
A_c   = 3                   # Plasma cross section [m^2]
L_tor = 5e-6                # Inductance [H]
r_0   = 3                   # Major radius [m]
a     = np.sqrt(A_c/np.pi)  # Minor radius [m]
B     = 2.4                 # Magnetic field [T]
tau_RE = 18
          # Diffusion
# Jämförde DREAMs diffusionsterm (lossterm) med de Vries loss-term och fick fram
# att vi borde använda D0=a/2/tau_RE
# Enheter blir skumma men vi sätter in en 1a så att allt blir bra :)

# Time and radial parameters
tMax = 8.5    # Simulation time [s]
Nt   = 10000  # Number of time steps
Nr   = 1      # Number of radial steps

# Experimental data from de Vries (approximated)
t      = np.linspace(0,tMax,num=Nt+1)        # Time vector for time depending data
I_p    = 0.9*np.exp(-1/(10*(t+0.8490)))*1e6  # Time dependent plasma current [A]
dI_p   = np.gradient(I_p,t)                  # dI_p/dt, time derivative of the plasma current [A/s]
V_loop = 0.08+0.77*np.exp(-t)                # Time dependent external torodial loop voltage [V]
n_e    = (0.287*np.exp(-0.8*t**2)+0.023)*1e19    # Time dependent density of thermal electrons [m^-3]
T_e    = (0.052+0.748*np.exp(-0.85*t))*1e3          # Time dependent thermal temperature [eV]

A0    = 0                   # Advection [???]
# Skippa advection helt och hållet
#D0=np.zeros((len(t), len(t)))
D=[]
D0 = (a/2/tau_RE)*np.exp(-t) #Diffusion
for q in D0:
    D.append([q, q])
D=np.array([D])

# Initial values in de Vries, changeable
I_re  = 6.5e5           # Initial runaway current [A]
n_re0 = I_re/(e*c*A_c)  # Initial density of runaway electrons [m^-3]

# Electric field calculated as in de Vries article
V_res = V_loop[0]-L_tor*dI_p[0]    # Time dependent resistive voltage [V] (V_res=I_tot*R_tot)
E_initial=V_res/(2*np.pi*r_0)  # Time dependent electric field strength [V/m]

# Total electron density
n = n_e[0]+n_re0  # Electron density [m^-3]


#### DREAM settings ####

ds = DREAMSettings()
#ds.collisions.collfreq_type = Collisions.COLLFREQ_TYPE_COMPLETELY_SCREENED
ds.collisions.collfreq_type = Collisions.COLLFREQ_TYPE_PARTIALLY_SCREENED

# Set E_field
ds.eqsys.E_field.setType(ElectricField.TYPE_SELFCONSISTENT)
ds.eqsys.E_field.setInitialProfile(efield=E_initial)
ds.eqsys.E_field.setBoundaryCondition(ElectricField.BC_TYPE_PRESCRIBED, V_loop_wall=V_loop, times=t)

# Set thermal temperature and thermal electron density
ds.eqsys.T_cold.setPrescribedData(T_e, times=t)
#byt n_cold till selfconsistent
#ds.eqsys.n_cold.setPrescribedData(density=n_e, times=t)
ds.eqsys.n_cold.setType(ColdElectrons.TYPE_SELFCONSISTENT)
# Set ions
AndelBerrylium = 1/3
Z1 = 1
Z2 = 4
ds.eqsys.n_i.addIon(name='D', Z=Z1, iontype=Ions.IONS_PRESCRIBED_FULLY_IONIZED, n=n*(1-AndelBerrylium)/1, t=t)
ds.eqsys.n_i.addIon(name='B', Z=Z2, iontype=Ions.IONS_PRESCRIBED_FULLY_IONIZED, n=n*AndelBerrylium/Z2)

#n=n säger hur många atomer. 1 atom med laddning 2 ger två elektroner så dela n med 2 om använder Z=2
# Lägg till ännu en rad för Berrylium, byt namn på det.
# Elektrondensitet som vi vill ha är n_e=n_D*Z_D+n_B*Z_B

# Runaway electrons
#ds.eqsys.n_re.setAvalanche(avalanche=Runaways.AVALANCHE_MODE_NEGLECT)
ds.eqsys.n_re.setAvalanche(avalanche=Runaways.AVALANCHE_MODE_FLUID_HESSLOW)
ds.eqsys.n_re.setInitialProfile(density=n_re0)

# Runaway loss
#t = np.linspace(tMax/2, tMax/2, 1)
t      = np.linspace(0,tMax,Nt+1)
r = np.linspace(0, a, 2)
A  = A0 * np.ones((len(t), len(r)))  # istället för 'np.array([...])'
  # istället för 'np.array([...])'

ds.eqsys.n_re.transport.prescribeAdvection(ar=A, t=t, r=r)
ds.eqsys.n_re.transport.prescribeDiffusion(drr=D, t=t, r=r)
ds.eqsys.n_re.transport.setBoundaryCondition(Transport.BC_F_0)

ds.eqsys.n_re.setDreicer(Runaways.DREICER_RATE_DISABLED)

# Hot-tail grid settings
ds.hottailgrid.setEnabled(False)

# Disable runaway grid
ds.runawaygrid.setEnabled(False)

# Set up radial grid
ds.radialgrid.setB0(B)
ds.radialgrid.setMinorRadius(a)
ds.radialgrid.setMajorRadius(r_0)
ds.radialgrid.setWallRadius(a)
ds.radialgrid.setNr(Nr)

# Set solver type
ds.solver.setType(Solver.LINEAR_IMPLICIT) # semi-implicit time stepping

# include otherquantities to save to output
ds.other.include('fluid')


# Set time stepper
ds.timestep.setTmax(tMax)
ds.timestep.setNt(Nt)

ds.output.setTiming(stdout=True, file=True)

# Save and run DREAM settings
ds.save('settings_de_Vries_test_2.h5')
do = runiface(ds, 'output_de_Vries_test_2.h5', quiet=False)
E_field = do.eqsys.E_field.data
Eceff = do.other.fluid.Eceff
Ecfree = do.other.fluid.Ecfree
tid = np.linspace(0,tMax,num=len(Eceff[:,-1]))
kappa = Eceff[:,-1]/Ecfree[:,-1]
plt.plot(tid,kappa)
plt.ylabel('kappa')
plt.xlabel('t [s]')
plt.show()
tid = np.linspace(0,tMax,num=len(E_field))
plt.plot(tid,E_field)
plt.xlabel('tid [s]'); plt.ylabel('E_field')
plt.show()
do.other.fluid.GammaAva.plot()
plt.show()
#plt.plot(Eceff[:,-1])
#plt.show()
#plt.plot(Ecfree[:,-1])
#plt.show()

#### Manage result ####

# Plot E_field
#do.eqsys.E_field.plot(r=[0,-1])
#plt.show()

# Plot I_p and I_re
n_re = do.eqsys.n_re[:]
I_re = n_re*e*c*A_c
tid=np.linspace(0,tMax,num=len(I_re))
plt.plot(tid,I_re)
plt.legend('R')
do.eqsys.I_p.plot()
plt.legend('P')
#plt.xlim(-1.5, 8.5)
#plt.ylim(0, 1e6)
plt.show()
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


###################################################################################################
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

