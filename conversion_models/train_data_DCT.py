import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
import random

'''
Ce script est pour construire le train data utilisé à créer le model Decision Tree 
Version : _v0.1_

Idée: prendre en aléatoire des profils de database, enlever les flags, les nan et configurer en format demandé du model
'''

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

def get_data_from_profil(dayfile, z_limite, nb_profils):
    '''
    Fonction permet d'obtenir les données correspondantes au profils aléatoires seléctionnés
    ! Cette fonction utilise les nouvelles versions de database Ipral 

    Input:
    - dayfile : data of 1 day file
    - z_limite : height limite where the profile have 
    - nb_profils : number of random profiles choosen

    Output:
    - 1 array in 2D (N, 2) where the 1st column is 355 signals and the second one is height values
    - 1 array in 1D of 532 signals 
    #-----------------
    '''
    z = dayfile['range'].values
    z_selected = z[z < z_limite] if isinstance(z_limite,int) else z[(z > z_limite[0])&(z < z_limite[1])]
    x355 = dayfile['Total_ScattRatio_355'].where(dayfile['flags_355']==0, drop=False).sel(range=z_selected)#
    x532 = dayfile['Total_ScattRatio_532'].where(dayfile['flags_532']==0, drop=False).sel(range=z_selected)#.isel(time=profils_selected).values
    x355 = x355.resample(time = '15min').mean(skipna=True)
    x532 = x532.resample(time = '15min').mean(skipna=True)
    times = np.intersect1d(x355.time, x532.time)
    #-----------------
    # get_profil_random_from_dayfile 
    profils_selected = get_profil_random_from_dayfile(times, nb_profils)
    x355 = x355.isel(time=profils_selected).values
    x532 = x532.isel(time=profils_selected).values
    #-----------------
    # remove nan values
    mask_nan = ~np.isnan(x355)&~np.isnan(x532)
    x355 = x355[mask_nan]
    x532 = x532[mask_nan]
    z2D_selected = np.tile(z_selected, (len(profils_selected), 1))[mask_nan]
    return x355, z2D_selected, x532

def get_data_from_profil_v2(dayfile, z_limite, nb_profils):
    '''
    Fonction permet d'obtenir les données correspondantes au profils aléatoires seléctionnés
    ! Cette fonction utilise le database en ancienne version 

    Input:
    - dayfile : data of 1 day file
    - z_limite : height limite where the profile have 
    - nb_profils : number of random profiles choosen

    Output:
    - 1 array in 2D (N, 2) where the 1st column is 355 signals and the second one is height values
    - 1 array in 1D of 532 signals 
    #-----------------
    '''
    z = dayfile['range'].values
    z_selected = z[z < z_limite] if isinstance(z_limite,int) else z[(z > z_limite[0])&(z < z_limite[1])]
    # x355 = dayfile['Total_ScattRatio_355'].where(dayfile['flags_355']==0, drop=False).sel(range=z_selected)#
    x355 = (dayfile['calibrated']/dayfile['simulated']).sel(wavelength=355,range=z_selected ) # before flags
    # x532 = dayfile['Total_ScattRatio_532'].where(dayfile['flags_532']==0, drop=False).sel(range=z_selected)#.isel(time=profils_selected).values
    x532 = (dayfile['calibrated']/dayfile['simulated']).sel(wavelength=532,range=z_selected ) # before flags
    #-----------------
    #add flags and resample mean time
    import sys
    sys.path.append('/homedata/nmpnguyen/IPRAL/NETCDF/')
    from flag_functions import get_all_flags
    pattern = pd.to_datetime(dayfile['time'].values[10]).strftime("%Y%m%d")
    _, _, flags = get_all_flags(rawfilepath = sorted(Path('/bdd/SIRTA/pub/basesirta/1a/ipral').glob(f'**/**/**/ipral_1a_Lz1R15mF30sPbck_v01_{pattern}_000000_1440.nc'))[0], 
                                limitez = 14000, 
                                max_calibration_height = 4000)
    x355 = x355.sel(time=flags).resample(time = '15min').mean(skipna=True)
    x532 = x532.sel(time=flags).resample(time = '15min').mean(skipna=True)
    #-----------------
    # get_profil_random_from_dayfile 
    profils_selected = get_profil_random_from_dayfile(x355['time'].values, nb_profils)
    x355 = x355.isel(time=profils_selected).values
    x532 = x532.isel(time=profils_selected).values
    # remove nan values
    mask_nan = ~np.isnan(x355)&~np.isnan(x532)
    x355 = x355[mask_nan]
    x532 = x532[mask_nan]
    z2D_selected = np.tile(z_selected, (len(profils_selected), 1))[mask_nan]
    return x355, z2D_selected, x532   

def get_all_data_selected_2features(alldaysfile, nb_profils, z_limite):
    '''
    Fonction 

    #-----------------
    '''
    all_X355 = np.array([[],[]])
    all_X532 = np.array([])
    for day in alldaysfile:
        day_data = xr.open_dataset(day)
#         profils_id = get_profil_random_from_dayfile(day_data, nb_profils)
        X355, Zselected, X532 = get_data_from_profil_v2(day_data, z_limite, nb_profils)
        # print(f'Shape de X355 et X532 of dayfile: {X355.shape}, {X532.shape}')
        all_X355 = np.concatenate([all_X355, [X355, Zselected]], axis=1)
        all_X532 = np.concatenate([all_X532, X532])
    return np.array(all_X355).T, np.array(all_X532)

def get_all_data_selected_3features(alldaysfile, nb_profils, z_limite):
    all_X355 = np.array([[], [], []])
    all_X532 = np.array([])
    for day in alldaysfile:
        day_data = xr.open_dataset(day)
#         profils_id = get_profil_random_from_dayfile(day_data, nb_profils)
        X355, Zselected, X532 = get_data_from_profil(day_data, z_limite, nb_profils)
        all_X355 = np.concatenate([all_X355, [X355, Zselected, np.square(Zselected)]], axis=1)
        all_X532 = np.concatenate([all_X532, X532])
    return np.array(all_X355).T, np.array(all_X532)

if __name__ == "__main__":
    Listfiles = sorted(Path('/homedata/nmpnguyen/IPRAL/RF/Calibrated/zone-3000-4000').glob('ipral_1a_Lz1R15mF30sPbck_v01_2018*.nc'))[:5]
    print(Listfiles)
    X355_Z, X532= get_all_data_selected_2features(Listfiles, 100, 14000)
    print(X355_Z.shape, X532.shape)