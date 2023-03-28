import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
import random

def get_profil_random_from_dayfile(daytime, nb_profils):
    '''
    Fonction permet d'obtenir d'un nombre proposé des profils aléatoires dans un jour de données

    Input: 
    - daytime : list of times (of total profils) in dayfile 
    - nb_profils : number of random profiles choosen 

    Output
    - index or list of indices of random profiles 
    #-----------------
    '''
    daytime = pd.to_datetime(daytime)
    import random 
    ns = random.sample(range(len(daytime)), min(nb_profils, len(daytime)))   
    return ns

def get_TestData_from_1profile(dayfile, z_limite, nb_profils=1):
	z = dayfile['range'].values
    z_selected = z[z < z_limite] if isinstance(z_limite,int) else z[(z > z_limite[0])&(z < z_limite[1])]
	sr355 = (dayfile['calibrated']/dayfile['simulated']).sel(wavelength=355, range=z_selected).values
	sr532 = (dayfile['calibrated']/dayfile['simulated']).sel(wavelength=532, range=z_selected).values
	#-----------------
    #add flags and resample mean time
    import sys
    sys.path.append('/homedata/nmpnguyen/IPRAL/NETCDF/')
    from flag_functions import get_all_flags
    pattern = pd.to_datetime(dayfile['time'].values[10]).strftime("%Y%m%d")
    _, _, flags = get_all_flags(rawfilepath = sorted(Path('/bdd/SIRTA/pub/basesirta/1a/ipral').glob(f'**/**/**/ipral_1a_Lz1R15mF30sPbck_v01_{pattern}_000000_1440.nc'))[0], 
                                limitez = 14000, 
                                max_calibration_height = 4000)
    sr355 = sr355.set(time=flags).resample(time='15mins').mean(skipna=True)
    sr532 = sr532.set(time=flags).resample(time='15mins').mean(skipna=True)
    
	return 

def get_TestData_from_MultiProfiles():
	return 

def get_all_data_selected_2features():
	return