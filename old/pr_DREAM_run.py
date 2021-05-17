import numpy as np
import sys
from scipy import integrate
import matplotlib.pyplot as plt

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


def run_DREAM(E,T_e,E1,Z_1,n_1,E2,Z_2,n_2,n_re0,B,a,r_0,A0,D0,r,Nr,t,tMax,Nt):
    #### DREAM settings ####

    ds = DREAMSettings()
    ds.collisions.collfreq_type = Collisions.COLLFREQ_TYPE_PARTIALLY_SCREENED

    # Set E_field
    ds.eqsys.E_field.setPrescribedData(E, times=t)

    # Set thermal temperature and thermal electron density
    ds.eqsys.T_cold.setPrescribedData(T_e, times=t)
    ds.eqsys.n_cold.setType(ColdElectrons.TYPE_SELFCONSISTENT)

    # Set ions
    ds.eqsys.n_i.addIon(name=E1, Z=Z_1, iontype=Ions.IONS_PRESCRIBED_FULLY_IONIZED, n=n_1, r=r, t=t)
    ds.eqsys.n_i.addIon(name=E2, Z=Z_2, iontype=Ions.IONS_PRESCRIBED_FULLY_IONIZED, n=n_2, r=r, t=t)


    # Runaway electrons
    #ds.eqsys.n_re.setAvalanche(avalanche=Runaways.AVALANCHE_MODE_NEGLECT)
    ds.eqsys.n_re.setAvalanche(avalanche=Runaways.AVALANCHE_MODE_FLUID_HESSLOW)
    ds.eqsys.n_re.setInitialProfile(density=n_re0)

    # Runaway loss
    t_D = np.linspace(tMax/2, tMax/2, 1)
    r_D = np.linspace(0, a, Nr+1)

    A  = A0 * np.ones((1, Nr+1))  # istället för 'np.array([...])'
    D  = D0 * np.ones((1, Nr+1))  # istället för 'np.array([...])'

    ds.eqsys.n_re.transport.prescribeAdvection(ar=A, t=t_D, r=r_D)
    ds.eqsys.n_re.transport.prescribeDiffusion(drr=D, t=t_D, r=r_D)
    ds.eqsys.n_re.transport.setBoundaryCondition(Transport.BC_F_0)

    ds.eqsys.n_re.setDreicer(Runaways.DREICER_RATE_DISABLED)

    # Hot-tail grid settings
    ds.hottailgrid.setEnabled(False)

    # Disable runaway grid
    ds.runawaygrid.setEnabled(False)

    # Set up radial grid
    ds.radialgrid.setB0(B)#, times=t)
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

    return ds, do
