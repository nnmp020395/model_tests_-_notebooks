
# coding: utf-8

# In[2]:


import xarray as xr
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import tqdm


# In[2]:


'''
tester flag code et verifier avec les quicklooks
enregistrer les quicklooks sur /scratchx/nmpnguyen/IPRAL/raw/
'''

def filter_profile_file(raw, channel, limiteTop, limiteBottom):
    '''
    Critere 1: flagger si le signal en haut plus qu'en bas
    '''
    # 1. CORRIGER RANGE CORRECTED SIGNAL WITH BACKGROUND
    filecorrected = raw #(raw[channel]/np.square(raw['range']) - raw['bckgrd_'+channel])*np.square(raw['range'])
    # 2. MEAN TOP AND BOTTOM SIGNAL
    limite = (raw['range']>limiteTop[0]) & (raw['range']<limiteTop[1])
    meanTop = filecorrected.isel(range=limite).mean(dim='range')
    limite = (raw['range']>limiteBottom[0]) & (raw['range']<limiteBottom[1])
    meanBottom = filecorrected.isel(range=limite).mean(dim='range')
    # 3. GETTING GOOD PROFILE #selectionner le profil correct
    index_mask = (meanTop-meanBottom).values < 0 # attention si meantop-meanBottom vient du raw (sélectionner channel) ou filecorrected (pas selectionner channel) 
#     print((meanTop-meanBottom).values)
    return index_mask


def invalidated_profile(data):
    nb_nonzero = np.count_nonzero(data>0, axis=1)
    nb_points_by_profile = data.shape[1]
    fraction_nonzero = nb_nonzero/nb_points_by_profile
    index_mask = fraction_nonzero > 0.5  #selectionner le profil correct    
    return index_mask


def nonzeros_fraction(data):
    nb_nonzero = np.count_nonzero(data>0, axis=1)
    nb_points_by_profile = data.shape[1]
    fraction_nonzero = nb_nonzero/nb_points_by_profile
    return fraction_nonzero


# In[3]:


year = Path('2018')
IPRAL_PATH = Path('/bdd/SIRTA/pub/basesirta/1a/ipral/', year)
IPRAL_LISTFILES = sorted(IPRAL_PATH.glob('**/**/ipral_1a_Lz1R15mF30sPbck_v01_*_000000_1440.nc'))
OUTPUT_PATH = Path('/scratchx/nmpnguyen/IPRAL/raw/flag_test', year)


# In[105]:



range_limite_top = [26000,28000]
range_limite_bottom = [2000,3000]

nb_validprofiles = np.array([0,0], dtype=float)
nb_validprofiles2 = np.array([0,0], dtype=float)
nb_totalprofiles = 0
for file in tqdm.tqdm(IPRAL_LISTFILES[2:6]):
    print(file)
    dt = xr.open_dataset(file)
    limiteZ = np.where((dt['range']<20000))[0]
    filecorrectsignal12 = (dt['rcs_12']/np.square(dt['range']) - dt['bckgrd_'+'rcs_12'])*np.square(dt['range'])
    filecorrectsignal16 = (dt['rcs_16']/np.square(dt['range']) - dt['bckgrd_'+'rcs_16'])*np.square(dt['range'])
    idsmask12 = filter_profile_file(filecorrectsignal12, 'rcs_12', range_limite_top, range_limite_bottom)
    idsmask16 = filter_profile_file(filecorrectsignal16, 'rcs_16', range_limite_top, range_limite_bottom)
    print(sum(idsmask12), sum(idsmask16))
    # print(f'nombre des profiles validés par critère 1: {sum(idsmask)}')
    nb_validprofiles[0] = nb_validprofiles[0] + sum(idsmask12) 
    nb_validprofiles[1] = nb_validprofiles[1] + sum(idsmask12) 
    ids_nonzero_12 = invalidated_profile(filecorrectsignal12.values)
    ids_nonzero_16 = invalidated_profile(filecorrectsignal16.values)
    print(sum(ids_nonzero_12), sum(ids_nonzero_16))
    # print(f'nombre des profiles validés par critère 2: {sum(idsmask2)}')
    nb_validprofiles2[0] = nb_validprofiles2[0] + sum(ids_nonzero_12)
    nb_validprofiles2[1] = nb_validprofiles2[1] + sum(ids_nonzero_16)
    nb_totalprofiles = nb_totalprofiles + len(dt['time'])


# In[5]:


'''
Plot
'''
fg, (ax, ax2)=plt.subplots(figsize=[9,12], nrows=2)
# RCS 12 : 355 ANALOG
np.log(filecorrectsignal12).isel(range=limiteZ).plot(x='time', y='range', vmin=10, vmax=18, ax=ax)
ax.set(title='rcs_12 355 ANALOG : validation of profiles')
# ax.set_xlim(dateStart, dateEnd)
ax.scatter(dt['time'].values[idsmask12], np.array([800]*len(dt['time'].values[idsmask12])), color='r', label='by attenuation')
ax.scatter(dt['time'].values[ids_nonzero_12], np.array([500]*len(dt['time'].values[ids_nonzero_12])), color='b', label='by nonzero')
ax.legend()
# RCS 16 : 532 ANALOG
np.log(filecorrectsignal16).isel(range=limiteZ).plot(x='time', y='range',  ax=ax2)
ax2.set(title='rcs_16 532 ANALOG : validation of profiles')
ax2.scatter(dt['time'].values[idsmask16], np.array([800]*len(dt['time'].values[idsmask16])), color='r', label='by attenuation')
ax2.scatter(dt['time'].values[ids_nonzero_16], np.array([500]*len(dt['time'].values[ids_nonzero_16])), color='b', label='by nonzero')
# ax2.set_xlim(dateStart, dateEnd)
ax2.legend()
plt.tight_layout()
# plt.savefig(Path(OUTPUT_PATH, file.name.split('.')[0]+'_W355W532.png'))


# In[6]:


def dispersion_standard_deviation(signal, id_left, seuil, influence):
    '''
    signal : est un vecteur numpy
    id_left : le nombre de valeurs avant la valeur actuelle que nous voulons utiliser pour calculer la ligne de base mobile.
    seuil : le nombre d'écarts types par rapport à la ligne de base mobile qu'un pic doit dépasser pour être compté.
    influence : la quantité d'influence qu'un pic a sur la ligne de base mobile. Celui-ci doit être compris entre 0 et 1.
    '''
    peakIndex = []
    processedSignal = signal[0:id_left]
    for ids in range(id_left, len(signal)):
        y = signal[ids]
        avg = np.nanmean(processedSignal[id_left:ids])
        sd = np.nanstd(processedSignal[id_left:ids])
        if ((y-avg) > (sd*seuil)):
            peakIndex.append(ids)
            print(ids, len(processedSignal))
#             ajustedValued = (influence*y)+((1-influence)*processedSignal[ids-1])
        else:
            processedSignal = np.append(processedSignal, y)
    return peakIndex    

def verification_by_SR(rawpath, wave, id_Z, mask_profiles):
    '''
    rawpath : le chemin du fichier Opar --> le nom du fichier 
    instrument : LI1200 ou LIO3T
    ch : channel du données 00355.o.Low ou 00355.o.VeryLow
    id_Z : indices des altitudes à étudier les nuages 
    mask_profiles : indices bool des nuages détectés à vérifier
    '''
    # Retrouver le fichier calibré correspondant 
    OPAR_RF_PATH = Path('/homedata/nmpnguyen/IPRAL/RF/Calibrated')
    OPAR_RF_FILE = Path(OPAR_RF_PATH, rawpath.name.split('.')[0]+'.nc')
    print(OPAR_RF_FILE)
    # Caculer SR, range correspondant aux profiles à vérifier
    datacalib = xr.open_dataset(OPAR_RF_FILE)
    SR2Darray = (datacalib['calibrated']/datacalib['simulated']).isel(range=id_Z).sel(wavelength = wave).values
    Zlimite2Darray = np.array([datacalib['range'][id_Z].values] * len(datacalib['time']))
    # Retourner des indices indiqués les nuages 
    zcalib_top = datacalib.attrs['calibration height'][1]*1e-3
    selected_indices_profiles = np.where((np.ma.masked_array(SR2Darray, mask=mask_profiles)>1.7) & (np.ma.masked_array(Zlimite2Darray, mask=mask_profiles)<zcalib_top))
    final_indices_profiles = np.unique(selected_indices_profiles[0])
    return final_indices_profiles, zcalib_top


# In[ ]:


limiteZ = (dt['range']<20000)&(dt['range']>3000)
mask_profiles = np.zeros_like(filecorrectsignal12.isel(range=limiteZ).values, dtype=bool)

for t in range(len(filecorrectsignal12['time'])):
    idsmask3 = dispersion_standard_deviation(filecorrectsignal12.isel(time=9, range=limiteZ).values, 
                                            id_left = 5,
                                            seuil = 4, 
                                            influence = 0.1)
    mask_profiles[t, idsmask3] = True 



# In[61]:


mask_profiles 


# In[62]:


fg, ax = plt.subplots(figsize=[12,8])
filecorrectsignal12.isel(time=90, range=limiteZ).plot(y='range', ax=ax)

xscatter = filecorrectsignal12.isel(time=90, range=limiteZ).values[mask_profiles[90,:]]
yscatter = filecorrectsignal12.isel(range=limiteZ)['range'][mask_profiles[90,:]]
ax.scatter(xscatter, yscatter, color='r')


# In[120]:


date =(file.name.split('_')[4])
alt_max=4000
print(date)


# In[228]:


import sys
import datetime
def ipral_remove_cloud_profiles(date, alt_max, ipral_file):
    """
    Remove IPRAL profiles containing cloud below a defined altitude.

    Parameters
    ----------
    date : datetime.datetime
        The date of the file to process.
    alt_max : float
        The maximum altitude of clouds in meters.
    ipral_file : str or pathlib.Path
        The path of the IPRAL file to process.
    output : str or pathlib.Path
        The path to th eoutput file.

    """

    CHM15K_PATH = Path("/bdd/SIRTA/pub/basesirta/1a/chm15k")
    #     CHM15K_MASK = 'chm15k_1a_z1Ppr2R15mF15s_v01_*_1440.nc"
    CHM15K_TIME_RES = "15s"

    IPRAL_PATH = Path("/bdd/SIRTA/pub/basesirta/1a/ipral/")
    IPRAL_MASK = "ipral_1a_Lz1R15mF30sPbck_v01_*_1440.nc"
    IPRAL_TIME_RES = "30s"

    DATE_FMT = "days since 1970-01-01T00:00:00"

    ONE_DAY = datetime.timedelta(hours=23, minutes=59, seconds=59)


    #     print(f"Processing {date:%Y-%m-%d}")
    print(f"Removing IPRAL profiles with clouds below {alt_max:7.1f}")
    # read CHM15k file
    # ---------------------------------------------------------------------------------
    #     chm15k_file = find_file(CHM15K_PATH, CHM15K_MASK, date)
    chm15k_file = sorted(CHM15K_PATH.glob(f'**/**/**/chm15k_1a_z1Ppr2R15mF15s_v01_{date}_000000_1440.nc'))
    if not chm15k_file:
        print("No CHM15k file found.")
        print("Quitting.")
        sys.exit(1)

    chm15k_file = chm15k_file[0]
    print(f"CHM15k file found: {chm15k_file}")
    cbh = xr.open_dataset(chm15k_file)["cloud_base_height"][:, 0]
    cbh = cbh.resample(time='30s').mean()
    # read IPRAL data
    # ---------------------------------------------------------------------------------
    ipral_data = xr.open_dataset(file).sel(time=slice(pd.to_datetime(date), pd.to_datetime(date) + ONE_DAY))
    raw_profs = ipral_data.time.size
    print(f"{raw_profs} in IPRAL data")

    ipral_time = ipral_data.time#.dt.round(freq='15s') 
    cbh_time = cbh.time

    # only keep timesteps of CBH available in ipral data
    time_intersect1d = np.intersect1d(ipral_time.values, cbh_time.values)
    cbh = cbh.sel(time = time_intersect1d)
    # create to only keep data without cloud under the chosen altitude
    cbh_mask = (cbh > alt_max) | np.isnan(cbh)
#     print(len(cbh['time'][cbh_mask]))
    profs_to_keep = cbh_mask.values.astype("i2").sum()
    print(f"{raw_profs - profs_to_keep} profiles will be remove")
    return cbh['time'][cbh_mask]


# In[230]:


ids = np.array(ipral_remove_cloud_profiles(file.name.split('_')[4], 4000, file))


# In[241]:


idsmask3 = np.intersect1d(dt['time'].values, ids, return_indices=True)
np.where(idsmask3)[0]


# In[220]:


def ipral_remove_cloud_profiles(alt_max, ipral_file):
    """
    Remove IPRAL profiles containing cloud below a defined altitude.

    Parameters
    ----------
    date : datetime.datetime
        The date of the file to process.
    alt_max : float
        The maximum altitude of clouds in meters.
    ipral_file : str or pathlib.Path
        The path of the IPRAL file to process.
    output : str or pathlib.Path
        The path to th eoutput file.

    """    
    CHM15K_PATH = Path("/bdd/SIRTA/pub/basesirta/1a/chm15k")
#     CHM15K_MASK = 'chm15k_1a_z1Ppr2R15mF15s_v01_*_1440.nc"
    CHM15K_TIME_RES = "15s"

    IPRAL_PATH = Path("/bdd/SIRTA/pub/basesirta/1a/ipral/")
    IPRAL_MASK = "ipral_1a_Lz1R15mF30sPbck_v01_*_1440.nc"
    IPRAL_TIME_RES = "30s"

    DATE_FMT = "days since 1970-01-01T00:00:00"
    ONE_DAY = datetime.timedelta(hours=23, minutes=59, seconds=59)
#     print(f"Processing {date:%Y-%m-%d}")
    print(f"Removing IPRAL profiles with clouds below {alt_max:7.1f}")
    # read CHM15k file
    # ---------------------------------------------------------------------------------
    date = ipral_file.name.split('_')[4]
    chm15k_file = sorted(CHM15K_PATH.glob(f'**/**/**/chm15k_1a_z1Ppr2R15mF15s_v01_{date}_000000_1440.nc'))
    if not chm15k_file:
        print("No CHM15k file found.")
        print("Quitting.")
        sys.exit(1)

    chm15k_file = chm15k_file[0]
    print(f"CHM15k file found: {chm15k_file}")
    cbh = xr.open_dataset(chm15k_file)["cloud_base_height"][:, 0].to_dataframe()[
        "cloud_base_height"
    ]
    # round time to 15s to ease use
    cbh.index = cbh.index.round(freq=CHM15K_TIME_RES)
    # under sample chm15k data to 30s to have the time resolution as ipral
    cbh = cbh.resample(IPRAL_TIME_RES).first()
    # read IPRAL data
    # ---------------------------------------------------------------------------------
    ipral_data = xr.open_dataset(ipral_file).sel(time=slice(pd.to_datetime(date), pd.to_datetime(date) + ONE_DAY))
    raw_profs = ipral_data.time.size
    print(f"{raw_profs} in IPRAL data")

    # get cloud mask
    # ---------------------------------------------------------------------------------
    # round time to 30s to ease use
    ipral_time = ipral_data.time.to_dataframe().index.round(freq=IPRAL_TIME_RES)
    time_intersect1d = np.intersect1d(np.array(cbh.index.values), np.array(ipral_time), return_indices=True)
    print(time_intersect1d)
    # only keep timesteps of CBH available in ipral data
    cbh = cbh.loc[time_intersect1d[0]]
    # create to only keep data without cloud under the chosen altitude
    cbh_mask = np.where((cbh > alt_max) | np.isnan(cbh))
    profs_to_keep = cbh_mask.values.astype("i2").sum()
    print(f"{raw_profs - profs_to_keep} profiles will be remove")

    # apply mask
    # ---------------------------------------------------------------------------------
#     ipral_data = ipral_data.isel(time=cbh_mask)
    return cbh_mask, time_intersect1d


# In[221]:


ids = np.array(ipral_remove_cloud_profiles(4000, file))


# ## SR histogram from list of validated profiles

# In[3]:


csv = pd.read_csv('/scratchx/nmpnguyen/IPRAL/raw/detection_clouds_test/IPRAL_2018_validated_profiles1.csv')
csv


# In[4]:


year = '2018'
IPRAL_PATH = Path('/bdd/SIRTA/pub/basesirta/1a/ipral/', year)
CALIB_PATH = Path('/homedata/nmpnguyen/IPRAL/RF/Calibrated/')
IPRAL_LISTFILES = sorted(CALIB_PATH.glob(f'ipral_1a_Lz1R15mF30sPbck_v01_{year}*_000000_1440.nc'))
print(len(IPRAL_LISTFILES), IPRAL_LISTFILES[0], IPRAL_LISTFILES[-1])


# In[7]:


def sr_by_files(csv_listdates, calib_pathfolder):
    date_of_list = pd.to_datetime(csv_listdates.iloc[1]).strftime('%Y%m%d')
    IPRAL_FILE_MASK = f'ipral_1a_Lz1R15mF30sPbck_v01_{str(date_of_list)}_000000_1440.nc'
    file = sorted(calib_pathfolder.glob(f'{IPRAL_FILE_MASK}'))[0]
#     idfile = np.where([file.stem.split('_')[4] == str(date_of_list) for file in calib_listfiles])[0][0]
    datacalib = xr.open_dataset(file) 
    limiteZ = (datacalib['range']>5000)&(datacalib['range']<15000)
#     indice = np.intersect1d(datacalib['time'].values, np.array(pd.to_datetime(csv_listdates)), return_indices=True)[1]
#     print(indice)
#     datacalib = datacalib.isel(time=indice).resample(time='15min').mean()
#     atb355 = datacalib['calibrated'].sel(wavelength = 355).isel(range=limiteZ)
#     atb532 = datacalib['calibrated'].sel(wavelength = 532).isel(range=limiteZ)
    csv_listdates = pd.to_datetime(csv_listdates).dropna().values
    sr355 = (datacalib['calibrated']/datacalib['simulated']).sel(wavelength = 355, time=csv_listdates).isel(range=limiteZ)
    sr532 = (datacalib['calibrated']/datacalib['simulated']).sel(wavelength = 532, time=csv_listdates).isel(range=limiteZ)
    return sr355, sr532 #atb355, atb532 #

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (array - value).argmin()
    return idx


# In[8]:


### (2021.11.10) select SR 355 included [5.0, 10.0] and SR 532 included [1.0, 3.0]

allsr355, allsr532 = sr_by_files(csv.iloc[-1,:], CALIB_PATH)

for i in range(1, csv.shape[0]):
    try:
        sr355file, sr532file = sr_by_files(csv.iloc[i,:], CALIB_PATH)
    #     print(sr355file, sr532file)
#         allsr355 = xr.concat([allsr355, sr355file.resample(time='15min').mean()], dim='time')
#         allsr532 = xr.concat([allsr532, sr532file.resample(time='15min').mean()], dim='time')
        allsr355 = xr.concat([allsr355, sr355file], dim='time')
        allsr532 = xr.concat([allsr532, sr532file], dim='time')
    except:
        pass

# id_alltimebug = np.array([], dtype='datetime64[ns]')
# for i in range(csv.shape[0]):
#     try:
#         sr355file, sr532file = sr_by_files(csv.iloc[i,:], CALIB_PATH)
#         id_timefile355 = np.unique(np.where((allsr355 > 5.0) & (allsr355 < 10.0))[0])
#         id_timefile532 = np.unique(np.where((allsr355 > 1.0) & (allsr355 < 5.0))[0])
#         id_timefile = np.intersect1d(id_timefile355, id_timefile532)
#         id_alltimebug = np.concatenate([id_alltimebug, sr355file['time'].values[id_timefile]])
#     except:
#         pass


# In[18]:


new_ids =np.where((allsr355/allsr532 > 5))
print(np.unique(new_ids[0]).shape, allsr532.shape)
# np.save('/scratchx/nmpnguyen/IPRAL/raw/SR_histogram/time_sr355.40_sr532.0_v2.npy',allsr355['time'][new_ids[0]].values)
# np.save('/scratchx/nmpnguyen/IPRAL/raw/SR_histogram/range_sr355.40_sr532.0_v2.npy',allsr355['range'][new_ids[1]].values)

new_allsr532 = allsr532.isel(time=np.unique(new_ids[0])) #allsr532.values[~(allsr532/allsr355 > 10)]
new_allsr355 = allsr355.isel(time=np.unique(new_ids[0])) #allsr355.values[~(allsr532/allsr355 > 10)]


# In[16]:


print(allsr355.shape)
ids = np.where((allsr355>=39)&(allsr355<=40)&(allsr532>=0)&(allsr532<1))
allsr355['time'][ids[0]]
np.save('/scratchx/nmpnguyen/IPRAL/raw/SR_histogram/time_sr355.40_sr532.0_v3.npy',allsr355['time'][ids[0]].values)


# In[41]:


allsr355['time'][ids[0]], allsr355['range'][ids[1]]
# np.save('/scratchx/nmpnguyen/IPRAL/raw/SR_histogram/time_sr355.40_sr532.0.npy',allsr355['time'][ids[0]].values)
# np.save('/scratchx/nmpnguyen/IPRAL/raw/SR_histogram/range_sr355.40_sr532.0.npy',allsr355['range'][ids[1]].values)
pd.to_datetime(allsr355['time'][ids[0]].values[1]).strftime('%Y%m%d')


# In[61]:


listdatenpy = np.unique(pd.to_datetime(np.load('/scratchx/nmpnguyen/IPRAL/raw/SR_histogram/time_sr355.40_sr532.0.npy')))
# [sorted(CALIB_PATH.glob(f'ipral_1a_Lz1R15mF30sPbck_v01_{l}_000000_1440.nc'))[0] for l in listdatenpy]


# In[89]:


def sr_by_profiles(profile_time, calib_listfiles, output=None):
    date_of_list = pd.to_datetime(profile_time).strftime('%Y%m%d')
    idfile = np.where([file.stem.split('_')[4] == str(date_of_list) for file in calib_listfiles])[0][0]
    datacalib = xr.open_dataset(calib_listfiles[idfile]) 
    limiteZ = (datacalib['range']>5000)&(datacalib['range']<15000)
    datacalib = datacalib.resample(time='15min').mean()
#     indice = np.intersect1d(datacalib['time'].values, np.array(pd.to_datetime(profile_time)), return_indices=True)[1]
    sr355 = (datacalib['calibrated']/datacalib['simulated']).sel(time=profile_time, wavelength = 355).isel(range=limiteZ)
    sr532 = (datacalib['calibrated']/datacalib['simulated']).sel(time=profile_time, wavelength = 532).isel(range=limiteZ)
    atb355 = datacalib['calibrated'].sel(time=profile_time, wavelength = 355).isel(range=limiteZ)
    atb532 = datacalib['calibrated'].sel(time=profile_time, wavelength = 532).isel(range=limiteZ)
    if output=='sr':
        return sr355, sr532
    else:
        return atb355, atb532


# In[90]:


print(listdatenpy[-3])


# In[91]:


fig, (ax, ax2) = plt.subplots(ncols=2, figsize=(12,6))
atb355profile, atb532profile = sr_by_profiles(listdatenpy[-3], IPRAL_LISTFILES, 'atb')
ax.semilogx(atb532profile.values, atb532profile['range'].values, label='532')
ax.semilogx(atb355profile.values, atb355profile['range'].values, label='355')
ax.set(xlabel='ATB, m-1.sr-1', title=f'{listdatenpy[-3]}')
sr355profile, sr532profile = sr_by_profiles(listdatenpy[-3], IPRAL_LISTFILES, 'sr')
sr532profile.plot(y='range', label='532', ax=ax2)
sr355profile.plot(y='range', label='355', ax=ax2)
ax2.set(xlabel='SR', title=f'{listdatenpy[-3]}')
ax2.set_xlim(-10,15)
# ax.axhline(allsr355['range'][398].values, color='k', linestyle='--')
ax.legend()
ax2.legend()
# ax.set_ylim(0,10000)


# In[73]:


listtime = np.array(np.load('/scratchx/nmpnguyen/IPRAL/raw/SR_histogram/time_sr355.40_sr532.0_v2.npy'), dtype='datetime64[ns]')
print(listtime)
listrange = np.load('/scratchx/nmpnguyen/IPRAL/raw/SR_histogram/range_sr355.40_sr532.0_v2.npy')
fig, axs = plt.subplots(ncols=3, nrows=int(len(listtime)/3), sharey=True, figsize=(15,16))
for i, ax in enumerate(axs.flatten()):
    print(listtime[i], listrange[i])
    atb355profile, atb532profile = sr_by_profiles(listtime[i], IPRAL_LISTFILES, 'atb')
#     atb532profile.plot(y='range', label='532', ax=ax)
#     atb355profile.plot(y='range', label='355', ax=ax)
    ax.semilogx(atb532profile.values, atb532profile['range'].values, label='532')
    ax.semilogx(atb355profile.values, atb355profile['range'].values, label='355')
    ax.axhline(listrange[i], color='k', linestyle='--')
    ax.set(xlabel='ATB, m-1.sr-1', title=f'{listtime[i]}')
    ax.legend()


# In[45]:


### Load le liste des profils à récupérer
# get_time = np.load('/scratchx/nmpnguyen/IPRAL/raw/SR_histogram/time_a_recupere.npy')
new_allsr355=xr.open_dataset('/scratchx/nmpnguyen/IPRAL/raw/SR_histogram/IPRAL_2018_validated_profiles1_allsr355.nc')['__xarray_dataarray_variable__']
new_allsr532=xr.open_dataset('/scratchx/nmpnguyen/IPRAL/raw/SR_histogram/IPRAL_2018_validated_profiles1_allsr532.nc')['__xarray_dataarray_variable__']
# print(pd.DataFrame((new_allsr532/new_allsr355).values).describe())


# In[46]:


# rapport = (new_allsr532/new_allsr355).values
# print(rapport.shape)
rows_bugs, cols_bugs = np.where((rapport < 1)|(rapport > 5))
new_allsr355 = new_allsr355.values[rows_bugs,cols_bugs]


# In[48]:


print(new_allsr355.shape)
new_allsr532 = new_allsr532.values[(rapport < 1)|(rapport > 5)]
print(new_allsr532.shape)


# In[32]:


sr_limite = [-10, 100]
X532 = new_allsr532.values.ravel()
Y355 = new_allsr355.values.ravel()
H = np.histogram2d(X532, Y355, bins=100, range = [[sr_limite[0], sr_limite[1]], [sr_limite[0], sr_limite[1]]])
# H = np.histogram2d(X532[np.where(~np.isnan(X532))], Y355[np.where(~np.isnan(Y355))], bins=100, 
#                    range = [[sr_limite[0], sr_limite[1]], [sr_limite[0], sr_limite[1]]]) 
Hprobas = H[0]/len(X532[np.where(~np.isnan(X532))])*100
Xxedges, Yyedges = np.meshgrid(H[1], H[2])


# In[ ]:


# from matplotlib.colors import LogNorm
# from scipy import stats
# # X532 = allsr532.values.ravel()
# # Y355 = allsr355.values.ravel()
# with np.errstate(invalid='ignore'):
#     slope, intercept, r_value, p_value, std_err = stats.linregress(X532[np.where(~np.isnan(X532))], Y355[np.where(~np.isnan(Y355))])
    
# fitLine = slope * X532[np.where(~np.isnan(X532))] + intercept
# fitLine2 = (1/4)*X532[np.where(~np.isnan(X532))]#allsr532

# slope, intercept, r_value, p_value, std_err


# In[33]:


from matplotlib.colors import LogNorm
ff, (ax, axins) = plt.subplots(figsize=[6,10], nrows=2)
p = ax.pcolormesh(Xxedges, Yyedges, Hprobas.T, norm = LogNorm(vmax=1e0, vmin=1e-6))
# ax.plot(X532[np.where(~np.isnan(X532))], fitLine2, '-.', c='r')
c = plt.colorbar(p, ax=ax, label='%')
ax.set(xlabel='SR532', ylabel='SR355', 
       title= f'IPRAL, resolution: ... x 15m')
ax.set(xlim=(sr_limite[0],sr_limite[1]), ylim=(sr_limite[0],sr_limite[1]))

pins = axins.pcolormesh(Xxedges, Yyedges, Hprobas.T, norm = LogNorm(vmax=1e-02, vmin=1e-6))
# axins.plot(X532[np.where(~np.isnan(X532))], fitLine2, '-.', c='r')
cins = plt.colorbar(pins, ax=axins, label='%')
axins.set_ylim(-5, 40)
axins.set_xlim(-5, 40)
axins.set(xlabel='SR532', ylabel='SR355')
# plt.savefig(Path('/scratchx/nmpnguyen/IPRAL/raw/SR_histogram/','distributionSR_100x100_Ipral2019.png'))


# In[26]:


# ratioXY = X532[~np.isnan(X532)&~np.isnan(Y355)]/Y355[~np.isnan(X532)&~np.isnan(Y355)]

print(Yyedges/Xxedges)


# In[24]:


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (array - value).argmin()
    return idx

# find_nearest(Y355[np.where(~np.isnan(Y355))] - fitLine2,0)
fitLine2 = (1/4)*X532
id1 = np.where(np.abs((Y355 - fitLine2)) < 1e-2)
# allsr355['time'][np.unique(id1)]
id1[0]


# In[25]:


f, ax = plt.subplots()
# ax.scatter(allsr532[id1[0]],allsr355[id1[1]])
ax.scatter(X532[id1], Y355[id1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)


# In[95]:


fitLine2 = (1/4)*allsr532
id1 = np.where(np.abs((allsr532 - fitLine2)) < 1e-2)
new_time = allsr532['time'][np.unique(id1[0])].values
np.save('/homedata/nmpnguyen/IPRAL/new_time.npy', new_time)


# In[30]:


old_sr532 = xr.open_dataset('/homedata/nmpnguyen/IPRAL/SR532.nc')#['__xarray_dataarray_variable__']
old_sr355 = xr.open_dataset('/homedata/nmpnguyen/IPRAL/SR355.nc')#['__xarray_dataarray_variable__'].values


# In[37]:


old_allsr532 = old_sr532.sortby('time').resample(time='15min').mean()
old_allsr355 = old_sr355.sortby('time').resample(time='15min').mean()


# In[34]:


old_X532 = old_allsr532['__xarray_dataarray_variable__'].values
old_Y355 = old_allsr355['__xarray_dataarray_variable__'].values

old_H = np.histogram2d(old_X532.ravel(), old_Y355.ravel(), bins=100, range = [[0, sr_limite], [0, sr_limite]]) 
old_Hprobas = H[0]/len(dt_old_sr532.ravel())*100
old_Xxedges, old_Yyedges = np.meshgrid(old_H[1], old_H[2])


# In[236]:


ff, (ax, axins) = plt.subplots(figsize=[6,10], nrows=2)
p = ax.pcolormesh(old_Xxedges, old_Yyedges, old_Hprobas.T, norm = LogNorm(vmax=1e0, vmin=1e-6))
ax.plot(X532, fitLine2, '-.', c='r')
c = plt.colorbar(p, ax=ax, label='%')
ax.set(xlabel='SR532', ylabel='SR355', 
       title= f'IPRAL, resolution: 15min x 15m \nLinearRegression: {round(slope,5)}x + {round(intercept,3)}')
ax.set(xlim=(0,sr_limite), ylim=(0,sr_limite))
plt.savefig('/homedata/nmpnguyen/IPRAL/test26102021.png')


# In[49]:


def flag_clouds(data, id_Z, rawpath, wave, selected_time):
    def dispersion_standard_deviation(signal, id_left, seuil, influence):
        '''
        signal : est un vecteur numpy
        id_left : le nombre de valeurs avant la valeur actuelle que nous voulons utiliser pour calculer la ligne de base mobile.
        seuil : le nombre d'écarts types par rapport à la ligne de base mobile qu'un pic doit dépasser pour être compté.
        influence : la quantité d'influence qu'un pic a sur la ligne de base mobile. Celui-ci doit être compris entre 0 et 1.
        '''
        peakIndex = []
        processedSignal = signal[0:id_left]
        for ids in range(id_left, len(signal)):
            y = signal[ids]
            avg = np.nanmean(processedSignal[id_left:ids])
            sd = np.nanstd(processedSignal[id_left:ids])
            if ((y-avg) > (sd*seuil)):
                peakIndex.append(ids)
#                 print(ids, len(processedSignal))
    #             ajustedValued = (influence*y)+((1-influence)*processedSignal[ids-1])
            else:
                processedSignal = np.append(processedSignal, y)
        return peakIndex

    def verification_by_SR(rawpath, wave, id_Z, selected_time, mask_profiles):
        '''
        rawpath : le chemin du fichier Ipral --> le nom du fichier 
        wave : 532nm et 355nm 
        id_Z : indices des altitudes à étudier les nuages 
        mask_profiles : indices bool des nuages détectés à vérifier
        '''
        # Retrouver le fichier calibré correspondant 
        IPRAL_RF_PATH = Path('/homedata/nmpnguyen/IPRAL/RF/Calibrated')
        IPRAL_RF_FILE = Path(IPRAL_RF_PATH, rawpath.name.split('.')[0]+'.nc')
        print(IPRAL_RF_FILE)
        # Caculer SR, range correspondant aux profiles à vérifier
        datacalib = xr.open_dataset(IPRAL_RF_FILE)
        datacalib = datacalib.resample(time='15min').mean()
        SR2Darray = (datacalib['calibrated']/datacalib['simulated']).isel(range=id_Z).sel(time= selected_time, wavelength = wave).values
        Zlimite2Darray = np.array([datacalib['range'][id_Z].values] * len(selected_time))
        # Retourner des indices indiqués les nuages 
        zcalib_top = 5000 #datacalib.attrs['calibration height'][1]
        selected_indices_profiles = np.where((np.ma.masked_array(SR2Darray, mask=mask_profiles)>1.7) & (np.ma.masked_array(Zlimite2Darray, mask=mask_profiles)<zcalib_top))
        final_indices_profiles = np.unique(selected_indices_profiles[0])
        return final_indices_profiles, zcalib_top
        
    indices_profiles = np.zeros_like(data.isel(range=id_Z).values, dtype=bool)
    for t in range(len(data['time'])):
        indices = dispersion_standard_deviation(data.isel(time=t, range=id_Z).values, 
                                                id_left = 5,
                                                seuil = 4, 
                                                influence = 0.1)
        indices_profiles[t, indices] = True 

    indices_clouds_profiles, zcalib_top = verification_by_SR(rawpath, wave, id_Z, selected_time, indices_profiles)
    indices_clouds_profiles = np.in1d(np.arange(len(data['time'])), indices_clouds_profiles)
    return indices_clouds_profiles


# In[50]:


old_time = np.load('/homedata/nmpnguyen/IPRAL/old_time.npy')
new_time = np.load('/homedata/nmpnguyen/IPRAL/new_time.npy')
diff_time = np.setdiff1d(old_time, new_time)


# In[55]:


range_limite_top = [26000,28000]
range_limite_bottom = [2000,3000]
import warnings
warnings.filterwarnings("ignore")
final_time_bug = []
for file in IPRAL_LISTFILES:
    print(file)
    data = xr.open_dataset(file)
    limiteZ = (data['range']>3000)&(data['range']<20000)
    data = data.resample(time='15min').mean()
    select_time_file = diff_time[pd.to_datetime(diff_time).strftime('%Y%m%d') == file.stem.split('_')[4]]
    print(select_time_file)
    data_outlier = data.sel(time=select_time_file)
    data_outlier12 = (data_outlier['rcs_12']/np.square(data_outlier['range']) - data_outlier['bckgrd_'+'rcs_12'])*np.square(data_outlier['range'])
    idsmask1_outlier12 = filter_profile_file(data_outlier12, 'rcs_12', range_limite_top, range_limite_bottom)
    idsmask2_outlier12 = invalidated_profile(data_outlier12)
    print('crit 1: ', np.where(idsmask1_outlier12)[0])
    print('crit 2: ', np.where(idsmask2_outlier12)[0])
    idsmask3_outlier12 = flag_clouds(data_outlier12, limiteZ, file, 355, select_time_file)
    print('crit 3: ', idsmask3_outlier12.shape)
    idsmask_outlier12 = np.intersect1d(np.intersect1d(np.where(idsmask1_outlier12)[0], np.where(idsmask2_outlier12)[0], return_indices=True),
                                      np.where(~idsmask3_outlier12)[0], return_indices=True)[0]
    final_time_bug.append(select_time_file[idsmask_outlier12])


# In[66]:



np.save('/scratchx/nmpnguyen/IPRAL/raw/SR_histogram/time_a_recupere.npy',np.concatenate(final_time_bug, axis=0), allow_pickle=True)


# ### Etude l'average des données

# In[38]:


data = xr.open_dataset('/homedata/nmpnguyen/IPRAL/RF/Calibrated/ipral_1a_Lz1R15mF30sPbck_v01_20180224_000000_1440.nc')
data


# In[39]:


datetime_profile = np.array('2018-02-24T21:05:00.000000000', dtype='datetime64[ns]')
id_selectedtime = np.where(data.time.dt.round('30S') == datetime_profile)[0]
id_selectedtime, data['calibrated'].isel(time=id_selectedtime, wavelength=0)
limiteZ = data['range']<15000
sr = data['calibrated']/data['simulated']


# In[60]:


import matplotlib.dates as mdates
fig, (ax, ax2) = plt.subplots(nrows=2, figsize = (7,12))
plt.rcParams['font.size'] = '12'
plt.rcParams['axes.labelsize'] = '12'

sr.isel(wavelength=0, range=limiteZ).plot(x='time', y='range', cmap='turbo', ax=ax, 
                                          cbar_kwargs={'label':'SR: 355nm'}, robust=True)
ax.axvline(sr['time'].values[id_selectedtime], color='k', linestyle='--')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

sr.isel(wavelength=1, range=limiteZ).plot(x='time', y='range', cmap='turbo', ax=ax2,
                                          cbar_kwargs={'label':'SR: 532nm'}, robust=True)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax2.axvline(sr['time'].values[id_selectedtime], color='k', linestyle='--')


# In[72]:


import matplotlib.dates as mdates
fig, (ax, ax2) = plt.subplots(nrows=2, figsize = (7,12))
plt.rcParams['font.size'] = '12'
plt.rcParams['axes.labelsize'] = '12'

sr.isel(wavelength=0, range=limiteZ).plot(x='time', y='range', cmap='turbo', ax=ax, 
                                          cbar_kwargs={'label':'SR: 355nm'}, robust=True)
ax.axvline(datetime_profile, color='k', linestyle='--')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax.set_xlim(pd.to_datetime(datetime_profile) - pd.DateOffset(minutes=15), pd.to_datetime(datetime_profile) + pd.DateOffset(minutes=15))

sr.isel(wavelength=1, range=limiteZ).plot(x='time', y='range', cmap='turbo', ax=ax2,
                                          cbar_kwargs={'label':'SR: 532nm'}, robust=True)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax2.axvline(datetime_profile, color='k', linestyle='--')
ax2.set_xlim(pd.to_datetime(datetime_profile) - pd.DateOffset(minutes=15), pd.to_datetime(datetime_profile) + pd.DateOffset(minutes=15))


# In[46]:


fg, (ax1, ax) = plt.subplots(figsize=(10,6), sharey=True, ncols=2)
# ax.semilogx(data['calibrated'].isel(time=id_selectedtime, wavelength=0), data['range'], label='355nm', color='b')
# ax.semilogx(data['calibrated'].isel(time=id_selectedtime, wavelength=1), data['range'], label='532nm', color='g')
sr.isel(time=id_selectedtime, wavelength=0, range=limiteZ).plot(y='range', label='355nm', color='b', ax=ax)
sr.isel(time=id_selectedtime, wavelength=1, range=limiteZ).plot(y='range', label='532nm', color='g', ax=ax)
ax.axvline(1, color='r', label='SR=1')
ax.legend()
ax.set(title=f'IPRAL: {datetime_profile}', xlabel='SR')

data['calibrated'].isel(time=id_selectedtime, wavelength=0, range=limiteZ).plot(y='range', label='355nm', color='b', ax=ax1, xscale='log')
data['calibrated'].isel(time=id_selectedtime, wavelength=1, range=limiteZ).plot(y='range', label='532nm', color='g', ax=ax1, xscale='log')
ax1.legend()
ax1.set(title=f'IPRAL: {datetime_profile}', xlabel='ATB')

ax.grid()
ax1.grid()


# In[50]:


data_mean15min = data.resample(time='15min').mean()
sr_mean15min = (data['calibrated']/data['simulated']).resample(time='15min').mean()
id_selectedtime = np.where(data_mean15min.time.dt.hour == pd.to_datetime(datetime_profile).hour)[0]
sr_mean15min['time'][-1], sr['time'][-1]


# In[77]:


fg, axs = plt.subplots(figsize=(12,6), ncols=2, nrows=1, sharey=True)
# plt.rcParams['font.size'] = '14'
plt.rcParams['axes.labelsize'] = '14'
for i, ax in enumerate(axs.flatten()):
    sr_mean15min.isel(time=id_selectedtime[i], wavelength=0, range=limiteZ).plot.line(y='range', ax=ax, label='355nm', color='b')
    sr_mean15min.isel(time=id_selectedtime[i], wavelength=1, range=limiteZ).plot.line(y='range', ax=ax, label='532nm', color='g')
    ax.axvline(1, color='r', label='SR=1')
    #     ax.set_ylim(0, 20000)
#     ax.set_xlim(-1,25)
    ax.set(xlabel='SR', title=f'{sr_mean15min.time.values[id_selectedtime[i]]}')
    ax.legend()


# In[53]:


fg, (ax1, ax) = plt.subplots(figsize=(10,6), ncols=2, nrows=1, sharey=True)
plt.rcParams['axes.labelsize'] = '14'
ax1.grid()
ax.grid()

sr_mean15min.isel(time=id_selectedtime[0], wavelength=0, range=limiteZ).plot.line(y='range', ax=ax, label='355nm', color='b')
sr_mean15min.isel(time=id_selectedtime[0], wavelength=1, range=limiteZ).plot.line(y='range', ax=ax, label='532nm', color='g')
ax.axvline(1, color='r', label='SR=1')
#     ax.set_ylim(0, 20000)
#     ax.set_xlim(-1,25)
ax.set(xlabel='SR', title=f'{sr_mean15min.time.values[id_selectedtime[0]]}')
ax.legend()

data_mean15min['calibrated'].isel(time=id_selectedtime[0], wavelength=0, range=limiteZ).plot.line(y='range', label='355nm', color='b', ax=ax1, xscale='log')
data_mean15min['calibrated'].isel(time=id_selectedtime[0], wavelength=1, range=limiteZ).plot.line(y='range', ax=ax1, label='532nm', color='g', xscale='log')
ax1.set(xlabel='ATB', title=f'{sr_mean15min.time.values[id_selectedtime[0]]}')
ax1.legend()


# In[173]:


get_ipython().system("cat '/scratchx/nmpnguyen/IPRAL/raw/detection_clouds_test/detection_clouds_and_flags.py'")


# In[221]:


new_csv = pd.read_csv('/scratchx/nmpnguyen/IPRAL/raw/detection_clouds_test/IPRAL_2018_validated_profiles3.csv',
                      parse_dates=True)
new_csv


# In[192]:


np.load('/scratchx/nmpnguyen/IPRAL/raw/SR_histogram/time_sr355.40_sr532.0.npy')


# In[4]:


data = xr.open_dataset(list(Path('/homedata/nmpnguyen/IPRAL/RF/Calibrated/').glob('*20180224*1440.nc'))[0])
# len(list(Path('/bdd/SIRTA/pub/basesirta/1a/ipral/2019').glob('**/**/*2019*1440.nc')))


# In[21]:


from matplotlib.colors import LogNorm
fig, (ax, ax2) = plt.subplots(figsize=(9,14), nrows=2)
plt.rcParams['font.size']=12
data.sel(wavelength=355)['calibrated'].plot(y='range', x='time', ax=ax, norm=LogNorm(vmin=1e-7, vmax=1e-3), robust=True, ylim=(0,20000))
ax.axvline(np.array('2018-12-14T12:15:00.000000000', dtype='datetime64[ns]'), color='white', linestyle='--')
ax.axvline(np.array('2018-12-14T12:45:00.000000000', dtype='datetime64[ns]'), color='white', linestyle='--')
data.sel(wavelength=532)['calibrated'].plot(y='range', x='time', ax=ax2, norm=LogNorm(vmin=1e-7, vmax=1e-3), robust=True, ylim=(0,20000))
ax2.axvline(np.array('2018-12-14T12:15:00.000000000', dtype='datetime64[ns]'), color='white', linestyle='--')
ax2.axvline(np.array('2018-12-14T12:45:00.000000000', dtype='datetime64[ns]'), color='white', linestyle='--')


# In[22]:


fig, ax2 = plt.subplots(figsize=(9,7))
plt.rcParams['font.size']=12
data.sel(wavelength=355)['calibrated'].resample(time='2min').mean('time').plot(y='range', x='time', ax=ax2, norm=LogNorm(vmin=1e-6, vmax=1e-3), robust=True, ylim=(0,20000))
ax2.axvline(np.array('2018-07-26T14:00:00.000000000', dtype='datetime64[ns]'), color='white', linestyle='--')
ax2.axvline(np.array('2018-07-26T14:45:00.000000000', dtype='datetime64[ns]'), color='white', linestyle='--')


# In[29]:


# calculer le gradient vetical de chaque point
data_mean = data.resample(time='2min').mean('time')
testtt = (data_mean['calibrated']/data_mean['simulated']).sel(wavelength=532).isel(range=slice(0,1400)).values
testt = np.concatenate([np.empty((testtt.shape[0],1))*np.NAN, testtt[:,:-1]], axis=1)


# In[36]:


# plot
plt.pcolormesh((testtt-testt).T, cmap='turbo', vmin=-5, vmax=5)
plt.colorbar()


# In[39]:


plt.subplots()
plt.scatter((testtt-testt)[3,slice(0,1400)], data_mean.isel(range=slice(0,1400))['range'])
plt.axvline(0.3, zorder=10, color='k')
plt.axvline(-0.3, zorder=10, color='k')

