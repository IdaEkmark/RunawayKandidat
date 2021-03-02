# Importer som inte används atm
#from DREAM.DREAMOutput import DREAMOutput
#import DREAM.Settings.Equations.DistributionFunction as DistFunc
#import DREAM.Settings.Equations.IonSpecies as Ions
#import DREAM.Settings.Equations.RunawayElectrons as Runaways
#import DREAM.Settings.Solver as Solver
#import DREAM.Settings.CollisionHandler as Collisions

###################################################################################################
import numpy as np
import sys
#from scipy import integrate
from scipy import interpolate
import matplotlib.pyplot as plt

sys.path.append('../../py/')

from DREAM.DREAMSettings import DREAMSettings
from DREAM import runiface
import DREAM.Settings.TransportSettings as Transport

def generate_current_profile_fun(I_p,current_profile,radie,tid,DREAM_settings_object):
    #Skapar DREAM-settings och output för att få conductivity
    ds_conductivity = DREAMSettings(DREAM_settings_object, chain=False)
    ds_conductivity.eqsys.E_field.setPrescribedData(0)
    #ds_conductivity.timestep.setNt(1)
    ds_conductivity.save('settings_conductivity_run.h5')
    do_conductivity = runiface(ds_conductivity, 'output_conductivity_run.h5', quiet=False)
    # Simulerar med de DREAMsettings som vi genererat och skapar ett output-objekt för att finna konduktivitet

    #Bestämmer E-fält som krävs för att få önskad strömprofil
    conductivity = do_conductivity.other.fluid.conductivity[:,:] #Ändra temperatur efter konduktivitet
    VpVol = do_conductivity.grid.VpVol[:]

    dr = do_conductivity.grid.dr
    r = do_conductivity.grid.r
    t = do_conductivity.grid.t
    t = np.delete(t,0)
    #r_f = do_conductivity.grid.r_f
    inter_cub_current_profile = interpolate.interp2d(radie, tid, current_profile, kind='cubic')
    j_prof_inter = inter_cub_current_profile(r, t)
    inter_cub_I_p = interpolate.interp1d(tid, I_p, kind='cubic')
    I_p_inter = inter_cub_I_p(t)

    E_r_t = []
    i=0
    for elements in j_prof_inter:
        current_profile_integral = np.sum(elements*VpVol*dr) #np.sum???
        j_0 = 2 * np.pi * I_p_inter[i] / current_profile_integral
        E_r = j_0 * elements / conductivity[i]
        E_r_t.append(E_r)
        i=i+1
    E_r_t = np.array(E_r_t)

    #Skapar DREAM-settings och output för att få önskad strömprofil
    ds_fun = DREAMSettings(DREAM_settings_object,chain=False)
    ds_fun.eqsys.E_field.setPrescribedData(E_r_t, times=t, radius=r)

    # Runaway loss
    t = np.linspace(0.5 / 2, 0.5 / 2, 1)
    r = np.linspace(0, 0.977, 5)

    A0 = 0  # Storlek på advektionen
    A = A0 * np.ones((1, 5))  # istället för 'np.array([...])'
    D0 = 100    # Storlek på advektionen
    D = D0*np.ones((1,5))   # istället för 'np.array([...])'

    ds_fun.eqsys.n_re.transport.prescribeAdvection(ar=A, t=t, r=r)
    ds_fun.eqsys.n_re.transport.prescribeDiffusion(drr=D, t=t, r=r)
    ds_fun.eqsys.n_re.transport.setBoundaryCondition(Transport.BC_F_0)

    ds_fun.save('settings_generate_current_profile_fun.h5')
    do_fun = runiface(ds_fun, 'output_generate_current_profile_fun.h5', quiet=False)

    return ds_fun, do_fun
