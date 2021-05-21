import numpy as np
import sys
import matplotlib.pyplot as plt


sys.path.append('../../py/')

from DREAM.DREAMSettings import DREAMSettings
from DREAM.DREAMOutput import DREAMOutput
import DREAM.Settings.Equations.RunawayElectrons as Runaways
import DREAM.Settings.Solver as Solver
import DREAM.Settings.CollisionHandler as Collisions
import DREAM.Settings.Equations.ColdElectrons as ColdElectrons
import DREAM.Settings.Equations.ColdElectronTemperature as ColdElectronTemperature
import DREAM.Settings.Equations.IonSpecies as Ions

#Funktion för att generera inställningar till en körning
def setupEQ(E,T_initial, tMax, Nt, n_D, n_B, B, a, r_0, Nr, V_loop, t):

    ds = DREAMSettings()
    ds.collisions.collfreq_type = Collisions.COLLFREQ_TYPE_PARTIALLY_SCREENED
    #Ansätter elektriska fältet och temperaturen
    ds.eqsys.E_field.setPrescribedData(E)
    ds.eqsys.T_cold.setType(ColdElectronTemperature.TYPE_SELFCONSISTENT)
    ds.eqsys.T_cold.setInitialProfile(T_initial)
    ds.eqsys.n_cold.setType(ColdElectrons.TYPE_SELFCONSISTENT)

    #inställningar för icke skenande elektroner och joner
    ds.eqsys.n_cold.setType(ColdElectrons.TYPE_SELFCONSISTENT)

    ds.eqsys.n_i.setIonization(Ions.IONIZATION_MODE_FLUID)
    ds.eqsys.n_i.addIon(name='D', Z=1, iontype=Ions.IONS_DYNAMIC_FULLY_IONIZED, n=n_D)
    ds.eqsys.n_i.addIon(name='B', Z=4, iontype=Ions.IONS_DYNAMIC_FULLY_IONIZED, n=n_B)

    #vilken typ av avalanche som används
    ds.eqsys.n_re.setAvalanche(avalanche=Runaways.AVALANCHE_MODE_FLUID_HESSLOW)
    ds.eqsys.n_re.setDreicer(Runaways.DREICER_RATE_CONNOR_HASTIE)

    ds.hottailgrid.setEnabled(False)
    ds.runawaygrid.setEnabled(False)
    #Konstanter för den simulerade tokamaken
    ds.radialgrid.setB0(B)
    ds.radialgrid.setMinorRadius(a)
    ds.radialgrid.setMajorRadius(r_0)
    ds.radialgrid.setWallRadius(a)
    ds.radialgrid.setNr(Nr)

    ds.solver.setType(Solver.NONLINEAR)

    ds.timestep.setTmax(tMax)
    ds.timestep.setNt(Nt)

    ds.other.include('fluid')

    return ds


def setupfindEQ(T_initial, tMax, Nt, n_D, n_B, B, a, r_0, Nr):
    ds = DREAMSettings()
    ds.collisions.collfreq_type = Collisions.COLLFREQ_TYPE_PARTIALLY_SCREENED  # Är detta rätt?

    ds.eqsys.n_cold.setType(ColdElectrons.TYPE_SELFCONSISTENT)

    ds.eqsys.T_cold.setPrescribedData(T_initial, times=t)

    ds.eqsys.n_i.setIonization(Ions.IONIZATION_MODE_FLUID)
    ds.eqsys.n_i.addIon(name='D', Z=1, iontype=Ions.IONS_PRESCRIBED_FULLY_IONIZED, n=n_D)
    ds.eqsys.n_i.addIon(name='B', Z=4, iontype=Ions.IONS_PRESCRIBED_FULLY_IONIZED, n=n_B)

    ds.eqsys.n_re.setAvalanche(avalanche=Runaways.AVALANCHE_MODE_FLUID_HESSLOW)

    ds.eqsys.n_re.setDreicer(Runaways.DREICER_RATE_CONNOR_HASTIE)

    ds.hottailgrid.setEnabled(False)
    ds.runawaygrid.setEnabled(False)

    ds.radialgrid.setB0(B)  # , times=t)
    ds.radialgrid.setMinorRadius(a)
    ds.radialgrid.setMajorRadius(r_0)
    ds.radialgrid.setWallRadius(a)
    ds.radialgrid.setNr(Nr)

    ds.solver.setType(Solver.LINEAR_IMPLICIT)

    ds.timestep.setTmax(tMax)
    ds.timestep.setNt(Nt)

    ds.other.include('fluid')

    return ds