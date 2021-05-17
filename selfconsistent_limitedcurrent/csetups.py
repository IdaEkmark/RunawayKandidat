import numpy as np
import sys
import matplotlib.pyplot as plt


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
import DREAM.Settings.Equations.IonSpecies as Ions

def setup1(E_initial, T_initial, V_loop_wall, t, Z_D, Z_B, n_D, n_B, B, a, r_0, Nr, tMax_c, Nt_c):

    ds_c = DREAMSettings()
    ds_c.collisions.collfreq_type = Collisions.COLLFREQ_TYPE_PARTIALLY_SCREENED #Är detta rätt?

    ds_c.eqsys.E_field.setType(ElectricField.TYPE_SELFCONSISTENT)
    ds_c.eqsys.E_field.setInitialProfile(efield=E_initial)
    ds_c.eqsys.E_field.setBoundaryCondition(ElectricField.BC_TYPE_PRESCRIBED, V_loop_wall=V_loop_wall, times=t)

    ds_c.eqsys.T_cold.setPrescribedData(T_initial, times=t)

    ds_c.eqsys.n_cold.setType(ColdElectrons.TYPE_SELFCONSISTENT)

    ds_c.eqsys.n_i.setIonization(Ions.IONIZATION_MODE_FLUID)
    ds_c.eqsys.n_i.addIon(name='D', Z=Z_D, iontype=Ions.IONS_DYNAMIC, Z0=1, n=n_D)
    ds_c.eqsys.n_i.addIon(name='B', Z=Z_B, iontype=Ions.IONS_DYNAMIC, Z0=3, n=n_B)

    ds_c.eqsys.n_re.setAvalanche(avalanche=Runaways.AVALANCHE_MODE_FLUID_HESSLOW)

    ds.eqsys.n_re.setDreicer(Runaways.DREICER_RATE_NEURAL_NETWORK)

    ds_c.hottailgrid.setEnabled(False)
    ds_c.runawaygrid.setEnabled(False)

    ds_c.radialgrid.setB0(B)#, times=t)
    ds_c.radialgrid.setMinorRadius(a)
    ds_c.radialgrid.setMajorRadius(r_0)
    ds_c.radialgrid.setWallRadius(a)
    ds_c.radialgrid.setNr(Nr)

    ds_c.solver.setType(Solver.NONLINEAR)

    ds_c.timestep.setTmax(tMax_c)
    ds_c.timestep.setNt(Nt_c)

    ds_c.other.include('fluid')

    return ds_c

def setup2(ds_c, tMax, Nt, E_initial, T_initial, V_loop_wall, t, Z_D, Z_B, n_D, n_B, B, a, r_0, Nr):

    ds = DREAMSettings(ds_c)
    do_c = DREAMOutput('output_SELFCONSISTENT1' + '.h5')

    ds.fromOutput('output_SELFCONSISTENT1' + '.h5')  # , ignore=['E_field','T_cold','n_cold','n_re'])


    ds.eqsys.T_cold.setType(ColdElectronTemperature.TYPE_SELFCONSISTENT)
    ds.eqsys.T_cold.setInitialProfile(T_initial)
    ds.eqsys.n_cold.setType(ColdElectrons.TYPE_SELFCONSISTENT)
    ds.collisions.collfreq_type = Collisions.COLLFREQ_TYPE_PARTIALLY_SCREENED

    # ds.collisions.bremsstrahlung_mode = Collisions.BREMSSTRAHLUNG_MODE_STOPPING_POWER

    ds.timestep.setTmax(tMax)
    ds.timestep.setNt(Nt)


    return ds

def setupEQ(E,T_initial, tMax, Nt, n_D, n_B, B, a, r_0, Nr):

    ds = DREAMSettings()
    ds.collisions.collfreq_type = Collisions.COLLFREQ_TYPE_PARTIALLY_SCREENED  # Är detta rätt?

    ds.eqsys.E_field.setPrescribedData(E)

    ds.eqsys.T_cold.setType(ColdElectronTemperature.TYPE_SELFCONSISTENT)
    ds.eqsys.T_cold.setInitialProfile(T_initial)
    ds.eqsys.n_cold.setType(ColdElectrons.TYPE_SELFCONSISTENT)

    ds.eqsys.n_cold.setType(ColdElectrons.TYPE_SELFCONSISTENT)

    ds.eqsys.n_i.setIonization(Ions.IONIZATION_MODE_FLUID)
    ds.eqsys.n_i.addIon(name='D', Z=1, iontype=Ions.IONS_DYNAMIC_FULLY_IONIZED, n=n_D)
    ds.eqsys.n_i.addIon(name='B', Z=4, iontype=Ions.IONS_DYNAMIC_FULLY_IONIZED, n=n_B)

    ds.eqsys.n_re.setAvalanche(avalanche=Runaways.AVALANCHE_MODE_FLUID_HESSLOW)

    ds.eqsys.n_re.setDreicer(Runaways.DREICER_RATE_NEURAL_NETWORK)

    ds.hottailgrid.setEnabled(False)
    ds.runawaygrid.setEnabled(False)

    ds.radialgrid.setB0(B)  # , times=t)
    ds.radialgrid.setMinorRadius(a)
    ds.radialgrid.setMajorRadius(r_0)
    ds.radialgrid.setWallRadius(a)
    ds.radialgrid.setNr(Nr)

    ds.solver.setType(Solver.NONLINEAR)

    ds.timestep.setTmax(tMax)
    ds.timestep.setNt(Nt)

    ds.other.include('fluid')

    return ds



def setupRE(ds_c,T_initial, E, t, Z_D, Z_B, n_D, n_B, B, a, r_0, Nr, tMax, Nt):
    ds = DREAMSettings(ds_c)
    do_c = DREAMOutput('output_SELFCONSISTENT1' + '.h5')

    ds.fromOutput('output_SELFCONSISTENT1' + '.h5')  # , ignore=['E_field','T_cold','n_cold','n_re'])

    ds.eqsys.T_cold.setType(ColdElectronTemperature.TYPE_SELFCONSISTENT)
    ds.eqsys.T_cold.setInitialProfile(T_initial)
    ds.eqsys.n_cold.setType(ColdElectrons.TYPE_SELFCONSISTENT)
    ds.collisions.collfreq_type = Collisions.COLLFREQ_TYPE_PARTIALLY_SCREENED

    # ds.collisions.bremsstrahlung_mode = Collisions.BREMSSTRAHLUNG_MODE_STOPPING_POWER
    ds.eqsys.E_field.setPrescribedData(efield=E,times=t)

    ds.timestep.setTmax(tMax)
    ds.timestep.setNt(Nt)

    return ds

def setupRE_FULLY_IONIZED(E_t, T_initial, t_RE, n_D, n_B, B, a, r_0, Nr, tMax_RE, Nt_RE):

    ds = DREAMSettings()
    ds.collisions.collfreq_type = Collisions.COLLFREQ_TYPE_PARTIALLY_SCREENED #Är detta rätt?

    ds.eqsys.E_field.setPrescribedData(efield=E_t, times=t_RE)

    ds.eqsys.T_cold.setType(ColdElectronTemperature.TYPE_SELFCONSISTENT)
    ds.eqsys.T_cold.setInitialProfile(T_initial)
    ds.eqsys.n_cold.setType(ColdElectrons.TYPE_SELFCONSISTENT)

    ds.eqsys.n_cold.setType(ColdElectrons.TYPE_SELFCONSISTENT)

    ds.eqsys.n_i.setIonization(Ions.IONIZATION_MODE_FLUID)
    ds.eqsys.n_i.addIon(name='D', Z=1, iontype=Ions.IONS_DYNAMIC_FULLY_IONIZED, n=n_D)
    ds.eqsys.n_i.addIon(name='B', Z=4, iontype=Ions.IONS_DYNAMIC_FULLY_IONIZED, n=n_B)

    ds.eqsys.n_re.setAvalanche(avalanche=Runaways.AVALANCHE_MODE_FLUID_HESSLOW)

    ds.eqsys.n_re.setDreicer(Runaways.DREICER_RATE_NEURAL_NETWORK)

    ds.hottailgrid.setEnabled(False)
    ds.runawaygrid.setEnabled(False)

    ds.radialgrid.setB0(B)#, times=t)
    ds.radialgrid.setMinorRadius(a)
    ds.radialgrid.setMajorRadius(r_0)
    ds.radialgrid.setWallRadius(a)
    ds.radialgrid.setNr(Nr)

    ds.solver.setType(Solver.NONLINEAR)

    ds.timestep.setTmax(tMax_RE)
    ds.timestep.setNt(Nt_RE)

    ds.other.include('fluid')

    return ds

def SCsetupRE_FULLY_IONIZED(E_initial_RE, V_loop_wall_RE, T_initial, t_RE, n_D, n_B, B, a, r_0, Nr, tMax_RE, Nt_RE):

    ds = DREAMSettings()
    ds.collisions.collfreq_type = Collisions.COLLFREQ_TYPE_PARTIALLY_SCREENED #Är detta rätt?

    ds.eqsys.E_field.setType(ElectricField.TYPE_SELFCONSISTENT)
    ds.eqsys.E_field.setInitialProfile(efield=E_initial_RE)
    ds.eqsys.E_field.setBoundaryCondition(ElectricField.BC_TYPE_PRESCRIBED, V_loop_wall=V_loop_wall_RE)

    ds.eqsys.T_cold.setType(ColdElectronTemperature.TYPE_SELFCONSISTENT)
    ds.eqsys.T_cold.setInitialProfile(T_initial)
    ds.eqsys.n_cold.setType(ColdElectrons.TYPE_SELFCONSISTENT)

    ds.eqsys.n_cold.setType(ColdElectrons.TYPE_SELFCONSISTENT)

    ds.eqsys.n_i.setIonization(Ions.IONIZATION_MODE_FLUID)
    ds.eqsys.n_i.addIon(name='D', Z=1, iontype=Ions.IONS_DYNAMIC_FULLY_IONIZED, n=n_D)
    ds.eqsys.n_i.addIon(name='B', Z=4, iontype=Ions.IONS_DYNAMIC_FULLY_IONIZED, n=n_B)

    ds.eqsys.n_re.setAvalanche(avalanche=Runaways.AVALANCHE_MODE_FLUID_HESSLOW)

    ds.eqsys.n_re.setDreicer(Runaways.DREICER_RATE_NEURAL_NETWORK)

    ds.hottailgrid.setEnabled(False)
    ds.runawaygrid.setEnabled(False)

    ds.radialgrid.setB0(B)#, times=t)
    ds.radialgrid.setMinorRadius(a)
    ds.radialgrid.setMajorRadius(r_0)
    ds.radialgrid.setWallRadius(a)
    ds.radialgrid.setNr(Nr)

    ds.solver.setType(Solver.NONLINEAR)

    ds.timestep.setTmax(tMax_RE)
    ds.timestep.setNt(Nt_RE)

    ds.other.include('fluid')

    return ds

def SCsetupRE(E_initial_RE, V_loop_wall_RE, T_initial, t_RE, n_D, n_B, B, a, r_0, Nr, tMax_RE, Nt_RE):

    ds = DREAMSettings()
    ds.collisions.collfreq_type = Collisions.COLLFREQ_TYPE_PARTIALLY_SCREENED #Är detta rätt?

    ds.eqsys.E_field.setType(ElectricField.TYPE_SELFCONSISTENT)
    ds.eqsys.E_field.setInitialProfile(efield=E_initial_RE)
    ds.eqsys.E_field.setBoundaryCondition(ElectricField.BC_TYPE_PRESCRIBED, V_loop_wall=V_loop_wall_RE)

    ds.eqsys.T_cold.setType(ColdElectronTemperature.TYPE_SELFCONSISTENT)
    ds.eqsys.T_cold.setInitialProfile(T_initial)
    ds.eqsys.n_cold.setType(ColdElectrons.TYPE_SELFCONSISTENT)

    ds.eqsys.n_cold.setType(ColdElectrons.TYPE_SELFCONSISTENT)

    ds.eqsys.n_i.setIonization(Ions.IONIZATION_MODE_FLUID)
    ds.eqsys.n_i.addIon(name='D', Z=1, iontype=Ions.IONS_DYNAMIC_FULLY_IONIZED, n=n_D)
    ds.eqsys.n_i.addIon(name='B', Z=4, iontype=Ions.IONS_DYNAMIC,Z0=2, n=n_B)

    ds.eqsys.n_re.setAvalanche(avalanche=Runaways.AVALANCHE_MODE_FLUID_HESSLOW)

    ds.eqsys.n_re.setDreicer(Runaways.DREICER_RATE_NEURAL_NETWORK)

    ds.hottailgrid.setEnabled(False)
    ds.runawaygrid.setEnabled(False)

    ds.radialgrid.setB0(B)#, times=t)
    ds.radialgrid.setMinorRadius(a)
    ds.radialgrid.setMajorRadius(r_0)
    ds.radialgrid.setWallRadius(a)
    ds.radialgrid.setNr(Nr)

    ds.solver.setType(Solver.NONLINEAR)

    ds.timestep.setTmax(tMax_RE)
    ds.timestep.setNt(Nt_RE)

    ds.other.include('fluid')

    return ds

def SC2setupRE_FULLY_IONIZED(E_initial_RE, V_loop_wall_RE, T_initial, t_RE, n_D, n_B, B, a, r_0, Nr, tMax_RE, Nt_RE):

    ds = DREAMSettings()
    ds.collisions.collfreq_type = Collisions.COLLFREQ_TYPE_PARTIALLY_SCREENED #Är detta rätt?

    ds.eqsys.E_field.setType(ElectricField.TYPE_SELFCONSISTENT)
    ds.eqsys.E_field.setInitialProfile(efield=E_initial_RE)
    ds.eqsys.E_field.setBoundaryCondition(ElectricField.BC_TYPE_PRESCRIBED, V_loop_wall=V_loop_wall_RE, times=t_RE)

    ds.eqsys.T_cold.setType(ColdElectronTemperature.TYPE_SELFCONSISTENT)
    ds.eqsys.T_cold.setInitialProfile(T_initial)
    ds.eqsys.n_cold.setType(ColdElectrons.TYPE_SELFCONSISTENT)

    ds.eqsys.n_cold.setType(ColdElectrons.TYPE_SELFCONSISTENT)

    ds.eqsys.n_i.setIonization(Ions.IONIZATION_MODE_FLUID)
    ds.eqsys.n_i.addIon(name='D', Z=1, iontype=Ions.IONS_DYNAMIC_FULLY_IONIZED, n=n_D)
    ds.eqsys.n_i.addIon(name='B', Z=4, iontype=Ions.IONS_DYNAMIC_FULLY_IONIZED, n=n_B)

    ds.eqsys.n_re.setAvalanche(avalanche=Runaways.AVALANCHE_MODE_FLUID_HESSLOW)

    ds.eqsys.n_re.setDreicer(Runaways.DREICER_RATE_NEURAL_NETWORK)

    ds.hottailgrid.setEnabled(False)
    ds.runawaygrid.setEnabled(False)

    ds.radialgrid.setB0(B)#, times=t)
    ds.radialgrid.setMinorRadius(a)
    ds.radialgrid.setMajorRadius(r_0)
    ds.radialgrid.setWallRadius(a)
    ds.radialgrid.setNr(Nr)

    ds.solver.setType(Solver.NONLINEAR)

    ds.timestep.setTmax(tMax_RE)
    ds.timestep.setNt(Nt_RE)

    ds.other.include('fluid')

    return ds

def SC2setupRE(E_initial_RE, V_loop_wall_RE, T_initial, t_RE, n_D, n_B, B, a, r_0, Nr, tMax_RE, Nt_RE):

    ds = DREAMSettings()
    ds.collisions.collfreq_type = Collisions.COLLFREQ_TYPE_PARTIALLY_SCREENED #Är detta rätt?

    ds.eqsys.E_field.setType(ElectricField.TYPE_SELFCONSISTENT)
    ds.eqsys.E_field.setInitialProfile(efield=E_initial_RE)
    ds.eqsys.E_field.setBoundaryCondition(ElectricField.BC_TYPE_PRESCRIBED, V_loop_wall=V_loop_wall_RE, times=t_RE)

    ds.eqsys.T_cold.setType(ColdElectronTemperature.TYPE_SELFCONSISTENT)
    ds.eqsys.T_cold.setInitialProfile(T_initial)
    ds.eqsys.n_cold.setType(ColdElectrons.TYPE_SELFCONSISTENT)

    ds.eqsys.n_cold.setType(ColdElectrons.TYPE_SELFCONSISTENT)

    ds.eqsys.n_i.setIonization(Ions.IONIZATION_MODE_FLUID)
    ds.eqsys.n_i.addIon(name='D', Z=1, iontype=Ions.IONS_DYNAMIC_FULLY_IONIZED, n=n_D)
    ds.eqsys.n_i.addIon(name='B', Z=4, iontype=Ions.IONS_DYNAMIC,Z0=2, n=n_B)

    ds.eqsys.n_re.setAvalanche(avalanche=Runaways.AVALANCHE_MODE_FLUID_HESSLOW)

    ds.eqsys.n_re.setDreicer(Runaways.DREICER_RATE_NEURAL_NETWORK)

    ds.hottailgrid.setEnabled(False)
    ds.runawaygrid.setEnabled(False)

    ds.radialgrid.setB0(B)#, times=t)
    ds.radialgrid.setMinorRadius(a)
    ds.radialgrid.setMajorRadius(r_0)
    ds.radialgrid.setWallRadius(a)
    ds.radialgrid.setNr(Nr)

    ds.solver.setType(Solver.NONLINEAR)

    ds.timestep.setTmax(tMax_RE)
    ds.timestep.setNt(Nt_RE)

    ds.other.include('fluid')

    return ds
