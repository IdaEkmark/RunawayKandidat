
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

def generate_current_profile_fun(I_p,current_profile,radie,DREAM_settings_object):
    #Skapar DREAM-settings och output för att få conductivity
    ds_conductivity = DREAMSettings(DREAM_settings_object, chain=False)
    ds_conductivity.eqsys.E_field.setPrescribedData(0)
    ds_conductivity.timestep.setNt(1)
    ds_conductivity.save('settings_conductivity_run.h5')
    do_conductivity = runiface(ds_conductivity, 'output_conductivity_run.h5', quiet=False)
    # Simulerar med de DREAMsettings som vi genererat och skapar ett output-objekt för att finna konduktivitet

    #Bestämmer E-fält som krävs för att få önskad strömprofil
    conductivity = do_conductivity.other.fluid.conductivity[-1,:]
    VpVol = do_conductivity.grid.VpVol[:]
    dr = do_conductivity.grid.dr
    r = do_conductivity.grid.r
    #r_f = do_conductivity.grid.r_f
    inter_cub_current_profile = interp1d(radie, current_profile, kind='cubic')
    j_prof_inter = inter_cub_current_profile(r)
    current_profile_integral = np.sum(j_prof_inter*VpVol*dr)
    j_0 = 2 * np.pi * I_p / current_profile_integral
    E_r = j_0 * j_prof_inter / conductivity

    #Skapar DREAM-settings och output för att få önskad strömprofil
    ds_fun = DREAMSettings(DREAM_settings_object,chain=False)
    ds_fun.eqsys.E_field.setPrescribedData(E_r, radius=r)
    ds_fun.save('settings_generate_current_profile_fun.h5')
    do_fun = runiface(ds_fun, 'output_generate_current_profile_fun.h5', quiet=False)

    return ds_fun, do_fun
