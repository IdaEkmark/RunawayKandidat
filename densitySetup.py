import numpy as np
from numpy.linalg import norm
import sys
import matplotlib
import matplotlib.pyplot as plt

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

def densitySetup(purity, n_e, n_re_init, t, r):
    n_ions = (n_e+n_re_init)/(4-3*purity)
    n_D = purity*n_ions
    n_Be = (1-purity)*n_ions
    tMax = len(t)
    nR = len(r)
    n_D_M = np.zeros((tMax, nR))
    n_Be_M = np.zeros((tMax, nR))

    for i_t in range(0,tMax):
        for i_r in range(0,nR):
            n_D_M[i_t,i_r] = n_D[i_t]
            n_Be_M[i_t, i_r] = n_Be[i_t]

    return n_D_M, n_Be_M
