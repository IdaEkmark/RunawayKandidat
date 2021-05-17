import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

sys.path.append('../../py/')

from DREAM.DREAMSettings import DREAMSettings
from DREAM.DREAMOutput import DREAMOutput
from DREAM import runiface
import DREAM.Settings.Equations.IonSpecies as Ions
import DREAM.Settings.Equations.RunawayElectrons as Runaways
import DREAM.Settings.Solver as Solver
import DREAM.Settings.CollisionHandler as Collisions
import DREAM.Settings.TransportSettings as Transport
import DREAM.Settings.Equations.ColdElectrons as ColdElectrons
import DREAM.Settings.Equations.ColdElectronTemperature as ColdElectronTemperature
import DREAM.Settings.Equations.ElectricField as ElectricField
import DREAM.Settings.Equations.DistributionFunction as DistFunc

def setupEQ(E,T_initial, tMax, Nt, n_D, n_B, B, a, r_0, Nr,D):

    ds = DREAMSettings()
    ds.collisions.collfreq_type = Collisions.COLLFREQ_TYPE_PARTIALLY_SCREENED
    radie=np.linspace(0,a,Nr)
    ds.eqsys.E_field.setPrescribedData(E, radius=radie)

    ds.eqsys.T_cold.setType(ColdElectronTemperature.TYPE_SELFCONSISTENT)
    ds.eqsys.T_cold.setInitialProfile(T_initial,radius=radie)
    ds.eqsys.n_cold.setType(ColdElectrons.TYPE_SELFCONSISTENT)

    ds.eqsys.n_cold.setType(ColdElectrons.TYPE_SELFCONSISTENT)

    ds.eqsys.n_i.setIonization(Ions.IONIZATION_MODE_FLUID)
    ds.eqsys.n_i.addIon(name='D', Z=1, iontype=Ions.IONS_DYNAMIC_FULLY_IONIZED, n=n_D)
    ds.eqsys.n_i.addIon(name='B', Z=4, iontype=Ions.IONS_DYNAMIC_FULLY_IONIZED, n=n_B)

    ds.eqsys.n_re.setAvalanche(avalanche=Runaways.AVALANCHE_MODE_FLUID_HESSLOW)

    ds.eqsys.n_re.setDreicer(Runaways.DREICER_RATE_NEURAL_NETWORK)

    ds.hottailgrid.setEnabled(False)
    ds.runawaygrid.setEnabled(False)

    ds.radialgrid.setB0(B)
    ds.radialgrid.setMinorRadius(a)
    ds.radialgrid.setMajorRadius(r_0)
    ds.radialgrid.setWallRadius(a)
    ds.radialgrid.setNr(Nr)

    ds.eqsys.T_cold.transport.prescribeDiffusion(D)

    ds.solver.setType(Solver.LINEAR_IMPLICIT)

    ds.timestep.setTmax(tMax)
    ds.timestep.setNt(Nt)

    ds.other.include('fluid')

    return ds

def setupSelfconsistent(E_initial, T_initial, V_loop_wall, t, Z_D, Z_B, n_D, n_B, B, a, r_0, Nr, tMax, Nt, D):

    ds = DREAMSettings()
    ds.collisions.collfreq_type = Collisions.COLLFREQ_TYPE_PARTIALLY_SCREENED

    radie = np.linspace(0, a, Nr)
    ds.eqsys.E_field.setType(ElectricField.TYPE_SELFCONSISTENT)
    ds.eqsys.E_field.setInitialProfile(efield=E_initial, radius=radie)
    ds.eqsys.E_field.setBoundaryCondition(ElectricField.BC_TYPE_PRESCRIBED, V_loop_wall=V_loop_wall, times=t)

    ds.eqsys.T_cold.setType(ColdElectronTemperature.TYPE_SELFCONSISTENT)
    ds.eqsys.T_cold.setInitialProfile(T_initial,radius=radie)

    ds.eqsys.n_cold.setType(ColdElectrons.TYPE_SELFCONSISTENT)

    ds.eqsys.n_i.setIonization(Ions.IONIZATION_MODE_FLUID)
    ds.eqsys.n_i.addIon(name='D', Z=Z_D, iontype=Ions.IONS_DYNAMIC_FULLY_IONIZED, n=n_D)
    ds.eqsys.n_i.addIon(name='B', Z=Z_B, iontype=Ions.IONS_DYNAMIC_FULLY_IONIZED, n=n_B)

    ds.eqsys.n_re.setAvalanche(avalanche=Runaways.AVALANCHE_MODE_FLUID_HESSLOW)
    ds.eqsys.n_re.setDreicer(Runaways.DREICER_RATE_NEURAL_NETWORK)

    ds.hottailgrid.setEnabled(False)
    ds.runawaygrid.setEnabled(False)

    ds.radialgrid.setB0(B)
    ds.radialgrid.setMinorRadius(a)
    ds.radialgrid.setMajorRadius(r_0)
    ds.radialgrid.setWallRadius(a)
    ds.radialgrid.setNr(Nr)

    ds.eqsys.T_cold.transport.prescribeDiffusion(D)

    ds.solver.setType(Solver.LINEAR_IMPLICIT)

    ds.timestep.setTmax(tMax)
    ds.timestep.setNt(Nt)

    ds.other.include('fluid')

    return ds

def setupSelfconsistentD(E_initial, T_initial, V_loop_wall, t, Z_D, Z_B, n_D, n_B, B, a, r_0, Nr, tMax, Nt,D):

    ds = DREAMSettings()
    ds.collisions.collfreq_type = Collisions.COLLFREQ_TYPE_PARTIALLY_SCREENED

    radie = np.linspace(0, a, Nr)
    ds.eqsys.E_field.setType(ElectricField.TYPE_SELFCONSISTENT)
    ds.eqsys.E_field.setInitialProfile(efield=E_initial, radius=radie)
    ds.eqsys.E_field.setBoundaryCondition(ElectricField.BC_TYPE_PRESCRIBED, V_loop_wall=V_loop_wall, times=t)

    ds.eqsys.T_cold.setType(ColdElectronTemperature.TYPE_SELFCONSISTENT)
    ds.eqsys.T_cold.setInitialProfile(T_initial,radius=radie)

    ds.eqsys.n_cold.setType(ColdElectrons.TYPE_SELFCONSISTENT)

    ds.eqsys.n_i.setIonization(Ions.IONIZATION_MODE_FLUID)
    ds.eqsys.n_i.addIon(name='D', Z=Z_D, iontype=Ions.IONS_DYNAMIC_FULLY_IONIZED, n=n_D)
    ds.eqsys.n_i.addIon(name='B', Z=Z_B, iontype=Ions.IONS_DYNAMIC_FULLY_IONIZED, n=n_B)

    ds.eqsys.n_re.setAvalanche(avalanche=Runaways.AVALANCHE_MODE_FLUID_HESSLOW)
    ds.eqsys.n_re.setDreicer(Runaways.DREICER_RATE_NEURAL_NETWORK)

    ds.hottailgrid.setEnabled(False)
    ds.runawaygrid.setEnabled(False)

    ds.radialgrid.setB0(B)
    ds.radialgrid.setMinorRadius(a)
    ds.radialgrid.setMajorRadius(r_0)
    ds.radialgrid.setWallRadius(a)
    ds.radialgrid.setNr(Nr)

    ds.eqsys.T_cold.transport.prescribeDiffusion(D)

    ds.solver.setType(Solver.LINEAR_IMPLICIT)

    ds.timestep.setTmax(tMax)
    ds.timestep.setNt(Nt)

    ds.other.include('fluid')

    return ds

def generate_current_profile_fun(I_p,current_profile,radie,Tmax_0,Tmin_0,Tmax_a,Tmin_a, tMax, Nt, n_D, n_B, B, a, r_0, Nr, D):
    tol = 1
    flag=0
    r_vec = np.linspace(0, a, Nr)
    while abs(Tmax_0 - Tmin_0) > tol and abs(Tmax_a - Tmin_a) > tol:
        T_0 = (Tmin_0 + Tmax_0) / 2
        T_a = (Tmin_a + Tmax_a) / 2

        T_vec=T_0-(T_0-T_a)/a**2*r_vec**2

        # Skapar DREAM-settings och output för att få conductivity
        ds_conductivity = DREAMSettings()
        ds_conductivity.collisions.collfreq_type = Collisions.COLLFREQ_TYPE_PARTIALLY_SCREENED
        ds_conductivity.eqsys.n_cold.setType(ColdElectrons.TYPE_SELFCONSISTENT)
        ds_conductivity.eqsys.n_i.setIonization(Ions.IONIZATION_MODE_FLUID)
        ds_conductivity.eqsys.n_i.addIon(name='D', Z=1, iontype=Ions.IONS_PRESCRIBED_FULLY_IONIZED, n=n_D)
        ds_conductivity.eqsys.n_i.addIon(name='B', Z=4, iontype=Ions.IONS_PRESCRIBED_FULLY_IONIZED, n=n_B)
        ds_conductivity.eqsys.n_re.setAvalanche(avalanche=Runaways.AVALANCHE_MODE_FLUID_HESSLOW)
        ds_conductivity.eqsys.n_re.setDreicer(Runaways.DREICER_RATE_CONNOR_HASTIE)
        ds_conductivity.hottailgrid.setEnabled(False)
        ds_conductivity.runawaygrid.setEnabled(False)
        ds_conductivity.radialgrid.setB0(B)
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
        ds_conductivity.eqsys.T_cold.setPrescribedData(T_vec, radius=r_vec)

        do_conductivity = runiface(ds_conductivity, quiet=False)
        # Simulerar med de DREAMsettings som vi genererat och skapar ett output-objekt för att finna konduktivitet

        # Bestämmer E-fält som krävs för att få önskad strömprofil
        conductivity = do_conductivity.other.fluid.conductivity[-1, :]
        VpVol = do_conductivity.grid.VpVol[:]
        dr = do_conductivity.grid.dr
        r = do_conductivity.grid.r
        if len(radie) < 4:
            inter_cub_current_profile = interp1d(radie, current_profile, kind='linear')
        else:
            inter_cub_current_profile = interp1d(radie, current_profile, kind='cubic')
        j_prof_inter = inter_cub_current_profile(r)
        current_profile_integral = np.sum(j_prof_inter * VpVol * dr)
        j_0 = 2 * np.pi * I_p / current_profile_integral
        E = j_0 * j_prof_inter / conductivity

        ds = setupEQ(E, T_vec, tMax, Nt, n_D, n_B, B, a, r_0, Nr, D)
        do = runiface(ds, quiet=False)
        Tout_vec = do.eqsys.T_cold[-1]
        Tout_0=Tout_vec[0]
        Tout_a=Tout_vec[-1]
        if Tout_0 >= T_vec[0]:
            Tmin_0 = T_vec[0]
            Tmax_0 = Tout_0
        elif Tout_0 < T_vec[0]:
            Tmax_0 = T_vec[0]
            Tmin_0 = Tout_0


        if Tout_a >= T_vec[-1]:
            Tmin_a = T_vec[-1]
            Tmax_a = Tout_a
        elif Tout_a < T_vec[-1]:
            Tmax_a = T_vec[-1]
            Tmin_a = Tout_a


    if flag==0:
        Teq_0 = (Tmin_0 + Tmax_0) / 2
        Teq_a = (Tmin_a + Tmax_a) / 2
        Teq_vec = Teq_0 - (Teq_0 - Teq_a) / a ** 2 * r_vec ** 2
        # Skapar DREAM-settings och output för att få conductivity
        ds_conductivity = DREAMSettings()
        ds_conductivity.collisions.collfreq_type = Collisions.COLLFREQ_TYPE_PARTIALLY_SCREENED
        ds_conductivity.eqsys.n_cold.setType(ColdElectrons.TYPE_SELFCONSISTENT)
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
        ds_conductivity.eqsys.T_cold.setPrescribedData(Teq_vec,radius=r_vec)

        do_conductivity = runiface(ds_conductivity, quiet=False)  # , 'output_conductivity_run.h5', quiet=False)
        # Simulerar med de DREAMsettings som vi genererat och skapar ett output-objekt för att finna konduktivitet

        # Bestämmer E-fält som krävs för att få önskad strömprofil
        conductivity = do_conductivity.other.fluid.conductivity[-1, :]
        VpVol = do_conductivity.grid.VpVol[:]
        dr = do_conductivity.grid.dr
        r = do_conductivity.grid.r
        if len(radie) < 4:
            inter_cub_current_profile = interp1d(radie, current_profile, kind='linear')
        else:
            inter_cub_current_profile = interp1d(radie, current_profile, kind='cubic')
        j_prof_inter = inter_cub_current_profile(r)
        current_profile_integral = np.sum(j_prof_inter * VpVol * dr)
        j_0 = 2 * np.pi * I_p / current_profile_integral
        Eeq_vec = j_0 * j_prof_inter / conductivity

        ds = setupEQ(Eeq_vec, Teq_vec, tMax, Nt, n_D, n_B, B, a, r_0, Nr, D)
        do = runiface(ds, quiet=False)

    return do,Teq_vec,Eeq_vec

def generate_current_profile_fun_T(I,current_profile,radie,T_vec,Imax,Imin, tMax, Nt, n_D, n_B, B, a, r_0, Nr,D):
    tol = 1e0
    r_vec = radie
    T_want=T_vec[0]
    while abs(Imax - Imin) > tol:
        # Skapar DREAM-settings och output för att få conductivity
        ds_conductivity = DREAMSettings()
        ds_conductivity.collisions.collfreq_type = Collisions.COLLFREQ_TYPE_PARTIALLY_SCREENED
        ds_conductivity.eqsys.n_cold.setType(ColdElectrons.TYPE_SELFCONSISTENT)
        ds_conductivity.eqsys.n_i.setIonization(Ions.IONIZATION_MODE_FLUID)
        ds_conductivity.eqsys.n_i.addIon(name='D', Z=1, iontype=Ions.IONS_PRESCRIBED_FULLY_IONIZED, n=n_D)
        ds_conductivity.eqsys.n_i.addIon(name='B', Z=4, iontype=Ions.IONS_PRESCRIBED_FULLY_IONIZED, n=n_B)
        ds_conductivity.eqsys.n_re.setAvalanche(avalanche=Runaways.AVALANCHE_MODE_FLUID_HESSLOW)
        ds_conductivity.eqsys.n_re.setDreicer(Runaways.DREICER_RATE_CONNOR_HASTIE)
        ds_conductivity.hottailgrid.setEnabled(False)
        ds_conductivity.runawaygrid.setEnabled(False)
        ds_conductivity.radialgrid.setB0(B)
        ds_conductivity.radialgrid.setMinorRadius(a)
        ds_conductivity.radialgrid.setMajorRadius(r_0)
        ds_conductivity.radialgrid.setWallRadius(a)
        ds_conductivity.eqsys.T_cold.transport.prescribeDiffusion(D)
        ds_conductivity.radialgrid.setNr(Nr)
        ds_conductivity.solver.setType(Solver.LINEAR_IMPLICIT)
        ds_conductivity.timestep.setTmax(tMax)
        ds_conductivity.timestep.setNt(Nt)
        ds_conductivity.other.include('fluid')
        ds_conductivity.eqsys.E_field.setPrescribedData(0)
        ds_conductivity.timestep.setNt(1)
        ds_conductivity.eqsys.T_cold.setPrescribedData(T_vec, radius=r_vec)

        do_conductivity = runiface(ds_conductivity, quiet=False)
        # Simulerar med de DREAMsettings som vi genererat och skapar ett output-objekt för att finna konduktivitet

        # Bestämmer E-fält som krävs för att få önskad strömprofil
        conductivity = do_conductivity.other.fluid.conductivity[-1, :]
        VpVol = do_conductivity.grid.VpVol[:]
        dr = do_conductivity.grid.dr
        r = do_conductivity.grid.r
        if len(radie) < 4:
            inter_cub_current_profile = interp1d(radie, current_profile, kind='linear')
        else:
            inter_cub_current_profile = interp1d(radie, current_profile, kind='cubic')
        j_prof_inter = inter_cub_current_profile(r)
        current_profile_integral = np.sum(j_prof_inter * VpVol * dr)
        j_0 = 2 * np.pi * I / current_profile_integral
        E = j_0 * j_prof_inter / conductivity

        ds = setupEQ(E, T_vec, tMax, Nt, n_D, n_B, B, a, r_0, Nr,D)
        do = runiface(ds, quiet=False)
        T_cold=do.eqsys.T_cold[-1]
        T_vec=T_cold/T_cold[0]*T_want
        I_p=do.eqsys.I_p[:]

        if I_p[0] >= I_p[-1]:
            Imin=I
        elif I_p[0] < I_p[-1]:
            Imax=I
        I=(Imax+Imin)/2

    # Skapar DREAM-settings och output för att få conductivity
    ds_conductivity = DREAMSettings()
    ds_conductivity.collisions.collfreq_type = Collisions.COLLFREQ_TYPE_PARTIALLY_SCREENED
    ds_conductivity.eqsys.n_cold.setType(ColdElectrons.TYPE_SELFCONSISTENT)
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
    ds_conductivity.eqsys.T_cold.transport.prescribeDiffusion(D)
    ds_conductivity.radialgrid.setNr(Nr)
    ds_conductivity.solver.setType(Solver.LINEAR_IMPLICIT)
    ds_conductivity.timestep.setTmax(tMax)
    ds_conductivity.timestep.setNt(Nt)
    ds_conductivity.other.include('fluid')
    ds_conductivity.eqsys.E_field.setPrescribedData(0)
    ds_conductivity.timestep.setNt(1)
    ds_conductivity.eqsys.T_cold.setPrescribedData(T_vec,radius=r_vec)

    do_conductivity = runiface(ds_conductivity, quiet=False)  # , 'output_conductivity_run.h5', quiet=False)
    # Simulerar med de DREAMsettings som vi genererat och skapar ett output-objekt för att finna konduktivitet

    # Bestämmer E-fält som krävs för att få önskad strömprofil
    conductivity = do_conductivity.other.fluid.conductivity[-1, :]
    VpVol = do_conductivity.grid.VpVol[:]
    dr = do_conductivity.grid.dr
    r = do_conductivity.grid.r
    if len(radie) < 4:
        inter_cub_current_profile = interp1d(radie, current_profile, kind='linear')
    else:
        inter_cub_current_profile = interp1d(radie, current_profile, kind='cubic')
    j_prof_inter = inter_cub_current_profile(r)
    current_profile_integral = np.sum(j_prof_inter * VpVol * dr)
    j_0 = 2 * np.pi * I / current_profile_integral
    E_vec = j_0 * j_prof_inter / conductivity

    ds = setupEQ(E_vec, T_vec, tMax, Nt, n_D, n_B, B, a, r_0, Nr, D)
    do = runiface(ds, quiet=False)
    T_r=do.eqsys.T_cold[-1]
    I_r=do.eqsys.I_p[-1]
    E_r=do.eqsys.E_field[-1]

    return do,T_r,I_r,E_r
