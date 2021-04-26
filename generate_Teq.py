import numpy as np
import sys
#from scipy import integrate
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

sys.path.append('../../py/')

from DREAM.DREAMSettings import DREAMSettings
from DREAM.DREAMOutput import DREAMOutput
from DREAM import runiface
import DREAM.Settings.Equations.DistributionFunction as DistFunc
import DREAM.Settings.Equations.IonSpecies as Ions
import DREAM.Settings.Equations.RunawayElectrons as Runaways
import DREAM.Settings.Solver as Solver
import DREAM.Settings.CollisionHandler as Collisions
from setups import setupEQ as seq
import DREAM.Settings.Equations.ColdElectrons as ColdElectrons

def generate_current_profile_fun(I_p,current_profile,radie,T_initial, tMax, Nt, n_D, n_B, B, a, r_0, Nr):
    Tmin = 0 # min-temperatur
    Tmax = 2*T_initial # max-temperatur
    tol = 1e-2
    while abs(Tmax - Tmin) > tol:
        T = (Tmin + Tmax) / 2

        # Skapar DREAM-settings och output för att få conductivity
        ds_conductivity = DREAMSettings()
        ds_conductivity.collisions.collfreq_type = Collisions.COLLFREQ_TYPE_PARTIALLY_SCREENED
        #ds_conductivity.eqsys.n_cold.setType(ColdElectrons.TYPE_SELFCONSISTENT)
        ds_conductivity.eqsys.n_i.setIonization(Ions.IONIZATION_MODE_FLUID)
        ds_conductivity.eqsys.n_i.addIon(name='D', Z=1, iontype=Ions.IONS_PRESCRIBED_FULLY_IONIZED, n=n_D)
        ds_conductivity.eqsys.n_i.addIon(name='B', Z=4, iontype=Ions.IONS_PRESCRIBED_FULLY_IONIZED, n=n_B)
        ds_conductivity.eqsys.n_re.setAvalanche(avalanche=Runaways.AVALANCHE_MODE_FLUID_HESSLOW)
        ds_conductivity.eqsys.n_re.setDreicer(Runaways.DREICER_RATE_CONNOR_HASTIE)
        ds_conductivity.hottailgrid.setEnabled(False)
        ds_conductivity.runawaygrid.setEnabled(False)
        ds_conductivity.radialgrid.setB0(B)  # , times=t)
        ds_conductivity.radialgrid.setMinorRadius(a)
        ds_conductivity.radialgrid.setMajorRadius(r_0)
        ds_conductivity.radialgrid.setWallRadius(a)
        ds_conductivity.radialgrid.setNr(Nr)
        ds_conductivity.solver.setType(Solver.LINEAR_IMPLICIT)
        ds_conductivity.timestep.setTmax(tMax)
        ds_conductivity.timestep.setNt(Nt)
        ds_conductivity.other.include('fluid')
        ds_conductivity.eqsys.E_field.setPrescribedData(0)
        ds_conductivity.timestep.setNt(1)
        ds_conductivity.eqsys.T_cold.setPrescribedData(T)

        #ds_conductivity.save('settings_conductivity_run.h5')
        do_conductivity = runiface(ds_conductivity, quiet=False)#, 'output_conductivity_run.h5', quiet=False)
        # Simulerar med de DREAMsettings som vi genererat och skapar ett output-objekt för att finna konduktivitet

        # Bestämmer E-fält som krävs för att få önskad strömprofil
        conductivity = do_conductivity.other.fluid.conductivity[-1, :]
        VpVol = do_conductivity.grid.VpVol[:]
        dr = do_conductivity.grid.dr
        r = do_conductivity.grid.r
        # r_f = do_conductivity.grid.r_f
        if len(radie) < 4:
            inter_cub_current_profile = interp1d(radie, current_profile, kind='linear')
        else:
            inter_cub_current_profile = interp1d(radie, current_profile, kind='cubic')
        j_prof_inter = inter_cub_current_profile(r)
        current_profile_integral = np.sum(j_prof_inter * VpVol * dr)
        j_0 = 2 * np.pi * I_p / current_profile_integral
        E = j_0 * j_prof_inter / conductivity

        ds = seq(E, T, tMax, Nt, n_D, n_B, B, a, r_0, Nr)
        do = runiface(ds, quiet=False)
        #do.eqsys.T_cold.plot()
        #plt.show()
        #do.eqsys.I_p.plot()
        #plt.show()
        Tout = do.eqsys.T_cold[:]
        print(str(Tout[-1]))
        if Tout[-1] > T:
            Tmin = T
            Tmax = Tout[-1]

        elif Tout[-1] < T:
            Tmax = T
            Tmin = Tout[-1]
    Teq = (Tmin + Tmax) / 2
    # Skapar DREAM-settings och output för att få conductivity
    ds_conductivity = DREAMSettings()
    ds_conductivity.collisions.collfreq_type = Collisions.COLLFREQ_TYPE_PARTIALLY_SCREENED
    # ds_conductivity.eqsys.n_cold.setType(ColdElectrons.TYPE_SELFCONSISTENT)
    ds_conductivity.eqsys.n_i.setIonization(Ions.IONIZATION_MODE_FLUID)
    ds_conductivity.eqsys.n_i.addIon(name='D', Z=1, iontype=Ions.IONS_PRESCRIBED_FULLY_IONIZED, n=n_D)
    ds_conductivity.eqsys.n_i.addIon(name='B', Z=4, iontype=Ions.IONS_PRESCRIBED_FULLY_IONIZED, n=n_B)
    ds_conductivity.eqsys.n_re.setAvalanche(avalanche=Runaways.AVALANCHE_MODE_FLUID_HESSLOW)
    ds_conductivity.eqsys.n_re.setDreicer(Runaways.DREICER_RATE_CONNOR_HASTIE)
    ds_conductivity.hottailgrid.setEnabled(False)
    ds_conductivity.runawaygrid.setEnabled(False)
    ds_conductivity.radialgrid.setB0(B)  # , times=t)
    ds_conductivity.radialgrid.setMinorRadius(a)
    ds_conductivity.radialgrid.setMajorRadius(r_0)
    ds_conductivity.radialgrid.setWallRadius(a)
    ds_conductivity.radialgrid.setNr(Nr)
    ds_conductivity.solver.setType(Solver.LINEAR_IMPLICIT)
    ds_conductivity.timestep.setTmax(tMax)
    ds_conductivity.timestep.setNt(Nt)
    ds_conductivity.other.include('fluid')
    ds_conductivity.eqsys.E_field.setPrescribedData(0)
    ds_conductivity.timestep.setNt(1)
    ds_conductivity.eqsys.T_cold.setPrescribedData(Teq)

    # ds_conductivity.save('settings_conductivity_run.h5')
    do_conductivity = runiface(ds_conductivity, quiet=False)  # , 'output_conductivity_run.h5', quiet=False)
    # Simulerar med de DREAMsettings som vi genererat och skapar ett output-objekt för att finna konduktivitet

    # Bestämmer E-fält som krävs för att få önskad strömprofil
    conductivity = do_conductivity.other.fluid.conductivity[-1, :]
    VpVol = do_conductivity.grid.VpVol[:]
    dr = do_conductivity.grid.dr
    r = do_conductivity.grid.r
    # r_f = do_conductivity.grid.r_f
    if len(radie) < 4:
        inter_cub_current_profile = interp1d(radie, current_profile, kind='linear')
    else:
        inter_cub_current_profile = interp1d(radie, current_profile, kind='cubic')
    j_prof_inter = inter_cub_current_profile(r)
    current_profile_integral = np.sum(j_prof_inter * VpVol * dr)
    j_0 = 2 * np.pi * I_p / current_profile_integral
    Eeq = j_0 * j_prof_inter / conductivity

    print('$T_{max}$=' + str(Tmax))
    print('$T_{min}=$' + str(Tmin))
    print('$T_{eq}=$' + str(Teq))
    print('$E_{eq}=$' + str(Eeq))
    print('KLAR\n\n\n\nKLAR')

    ds = seq(Eeq, Teq, tMax, Nt, n_D, n_B, B, a, r_0, Nr)
    do = runiface(ds, quiet=False)
    # Jämviktstemperaturen är T

    return do,Teq,Eeq