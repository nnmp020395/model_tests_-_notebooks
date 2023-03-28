
# coding: utf-8

# Objectif de cette session
# =============================
# 
# Vérification des filtres appliqués au traitement des données Ipral
# 
# Ces filtres se basent sur des données corrigées de la distance au carré et du fond de ciel "Range and Background corrected signal"

# In[8]:


import xarray as xr
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import tqdm
import datetime
from matplotlib.colors import LogNorm
import matplotlib.dates as mdates


# Charger le list des données, brutes et calibrées et un fichier de test

# In[2]:


year = Path('2018')
# IPRAL_PATH = Path('/bdd/SIRTA/pub/basesirta/1a/ipral/', year)
IPRAL_PATH = Path('/backupfs/sirta/pub/basesirta/1a/ipral/', year)
CALIB_PATH = Path('/homedata/nmpnguyen/IPRAL/RF/Calibrated/')
IPRAL_LISTFILES = sorted(IPRAL_PATH.glob('**/**/ipral_1a_Lz1R15mF30sPbck_v01_20180909_000000_1440.nc'))
wave = 532
print(IPRAL_LISTFILES)
filepath = IPRAL_LISTFILES[0]
filepath


# In[22]:


sorted(IPRAL_PATH.glob('**/**/ipral_1a_Lz1R15mF30sPbck_v01_*_000000_1440.nc'))


# In[23]:


import os 
os.getcwd()


# __Corriger les signaux bruts -> Range and background corrected signals__

# In[3]:


data = xr.open_dataset(filepath)
rcs_channel = 'rcs_12' if wave == 355 else 'rcs_16'
dateStart = pd.to_datetime(filepath.stem.split('_')[4])
dateEnd = dateStart + pd.DateOffset(1)
datacorrected = (data[rcs_channel]/np.square(data['range']) - data['bckgrd_'+rcs_channel])*np.square(data['range'])

limitez = (data['range']<20000)


# In[188]:


fig, (ax, ax2) = plt.subplots(figsize=(9,6), ncols=2)
data['rcs_12'].isel(time=585, range=limitez).plot(y='range', ax=ax, xscale='log', ylim=(0,20000), label='355:raw data', color='b')
data['rcs_16'].isel(time=585, range=limitez).plot(y='range', ax=ax, xscale='log', ylim=(0,20000), label='532:raw data', color='g')
ax.legend()
ax.set(xlabel='Signal')
data['rcs_12'].isel(time=585, range=limitez).plot(y='range', ax=ax2, xscale='log', ylim=(0,20000), label=f'355:raw data', color='b', linestyle='--')
datacorrecteds = (data['rcs_12']/np.square(data['range']) - data['bckgrd_'+'rcs_12'])*np.square(data['range'])
datacorrecteds.isel(time=585, range=limitez).plot(y='range', ax=ax2, xscale='log', ylim=(0,20000), label=f'355:corrected data', color='b')

data['rcs_16'].isel(time=585, range=limitez).plot(y='range', ax=ax2, xscale='log', ylim=(0,20000), label=f'532:raw data', color='g', linestyle='--')
datacorrecteds = (data['rcs_16']/np.square(data['range']) - data['bckgrd_'+'rcs_16'])*np.square(data['range'])
datacorrecteds.isel(time=585, range=limitez).plot(y='range', ax=ax2, xscale='log', ylim=(0,20000), label=f'532:corrected data', color='g')
ax2.legend()
ax2.set(xlabel='Range and Background corrected Signal')
plt.tight_layout()


# In[60]:


datacorrected = datacorrected.sel(time=slice(dateStart, dateEnd))


# __Crit 1: Enlever les profils ayant plus de signaux en haut qu'en bas__
# 
# Comparer entre la valeur moyennée du signal dans une zone indiquée à haute altitude et à basse altitude, flagger si cette valeur est plus grande en haut qu'en bas,

# In[61]:


def filter_profile_file(filecorrected, channel, limiteTop, limiteBottom):
    '''
    Critere 1: flagger si le signal en haut plus qu'en bas
    Input: 
        raw: range & background corrected signal
        channel: canal utilisé
        limiteTop: la zone à haute altitude pour le filtre
        limiteBottom: la zone à basse altitude pour le filtre
    Output:
        index_mask: index indiquant des profils validés/invalidés 
    '''
    # 1. MEAN TOP AND BOTTOM SIGNAL
    limite = (filecorrected['range']>limiteTop[0]) & (filecorrected['range']<limiteTop[1])
    meanTop = filecorrected.isel(range=limite).mean(dim='range')
    limite = (filecorrected['range']>limiteBottom[0]) & (filecorrected['range']<limiteBottom[1])
    meanBottom = filecorrected.isel(range=limite).mean(dim='range')
    # 2. GETTING GOOD PROFILE #selectionner le profil correct
    index_mask = (meanTop-meanBottom) < 0 # attention si meantop-meanBottom vient du raw (sélectionner channel) ou filecorrected (pas selectionner channel) 
    return index_mask


# In[62]:


print(f'{len(datacorrected.time.values)} profils total dans ce fichier Ipral')
# entrer les zones indiquées
range_limite_top = [26000,28000]
range_limite_bottom = [2000,3000]
# appliquer le filtre 1
mask_crit1 = filter_profile_file(datacorrected, rcs_channel, range_limite_top, range_limite_bottom)
print(mask_crit1)
# bool array to index array
id_crit1 = np.where(mask_crit1)[0]
print(f'{(id_crit1)} profils gardés selon CRIT 1')
# variable pour les profils invalides
id_invalid_crit1 = np.where(mask_crit1==False)[0]
print(f'{len(id_invalid_crit1)} profils jetés selon CRIT 1')


# Illustration: __quicklook Range and Background corrected signal non filtré__

# In[101]:



fig, ax = plt.subplots(figsize=(7,6))
datacorrected.plot(x='time', y='range', norm=LogNorm(vmin=1e2, vmax=1e8), ax=ax, robust=True, cmap='turbo',
                   ylim=(0,20000), cbar_kwargs={'label':'Range & Background corrected signal',
                                               'orientation':'horizontal'})
# ax.scatter(datacorrected[index_mask]['time'].values, [19000]*len(datacorrected[index_mask]['time'].values), color='y')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax.set(title=f'{filepath.stem} (initial)')


# Illustration: __Plot des profils mauvais__
# 
# Constater que ces profils augmentent en fonction de l'altitude, donc c'est irréel. Ils sont tous enlenver. 

# In[102]:


# limitez = datacorrected['range'] < 20000
# datacorrected.isel(time=id_invalid_crit1).plot.line(y='range', col='time', col_wrap=3, ylim=(0,30000))


# Illustration: __le QL filtré__

# In[63]:


from matplotlib.colors import LogNorm
import matplotlib.dates as mdates
Data1 = datacorrected.where(mask_crit1, drop=False)
fig, ax = plt.subplots(figsize=(7,6))
Data1.plot(x='time', y='range', norm=LogNorm(vmin=1e2, vmax=1e8), ax=ax, robust=True, cmap='turbo',
           ylim=(0,20000), cbar_kwargs={'label':'Range & Background corrected signal',
                                               'orientation':'horizontal'})
# ax.scatter(datacorrected[index_mask]['time'].values, [19000]*len(datacorrected[index_mask]['time'].values), color='y')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax.set(title=f'filtre 1: garder les profils diminués en fonction de l altitude')


# __*Stocker ces index des profils validés dans une variable:*__ id_crit1

# __Crit 2: Enveler les profils ayant trop de valeurs NaN et bizarres__
# 
# Etape 1: calculer la proportion des valeurs invalides par profil

# In[4]:


def proportion_values_profile(dataipral):
    nb_nonzero = ((dataipral>0) & ~np.isnan(dataipral)).sum(axis=1)
    nb_points_by_profile = dataipral.shape[1]
    fraction_nonzero = nb_nonzero/nb_points_by_profile   
    return fraction_nonzero
    
# index_mask = fraction_nonzero > 0.7  #selectionner le profil correct    
# index_mask
# datacorrected[index_mask]


# In[11]:


#----------------------------------------------------
# sur tous les fichiers 
all_validated_profile = None
IPRAL_LISTFILES = sorted(IPRAL_PATH.glob('**/**/ipral_1a_Lz1R15mF30sPbck_v01_*_000000_1440.nc'))

import tqdm
for filepath in tqdm.tqdm(IPRAL_LISTFILES):
    data = xr.open_dataset(filepath)
    datacorrecteds = (data[rcs_channel]/np.square(data['range']) - data['bckgrd_'+rcs_channel])*np.square(data['range'])
    limitez = (data['range']<20000)
    if (all_validated_profile is None):
        all_validated_profile = proportion_values_profile(datacorrecteds.isel(range=limitez))
    else:
        all_validated_profile = xr.concat([all_validated_profile, proportion_values_profile(datacorrecteds.isel(range=limitez))], dim='time')


# Etape 2: Calculer le seuil de définition des profils invalides

# In[12]:


all_validated_profile.to_dataframe('proportion').describe()


# In[13]:


mean_seuil = all_validated_profile.mean().values
print(f'{mean_seuil}', file=open('ipral_validated_profiles_seuil_mean.txt', 'a'))


# In[14]:


# Histogramme de la distribution des proportions sur une base de données avec le seuil 
plt.hist(all_validated_profile, range=[0.0, 1.0], bins=50)
plt.minorticks_on()
plt.grid(b=True, which='minor', color='k', linestyle='--', alpha=0.2)
plt.grid(b=True, which='major', color='k', linestyle='--', alpha=0.2)
plt.axvline(mean_seuil, color='r', label='fraction mean -> threshold')
plt.legend()
plt.ylabel('number of validated profiles')
plt.xlabel('fraction')


# Etape 3: Appliquer le filtre avec ce seuil ci-dessus et trouver les profils satisfaisants à cette critère 2.

# In[5]:


def validated_profile(dataipral): 
    nb_nonzero = ((dataipral>0) & ~np.isnan(dataipral)).sum(axis=1)
    nb_points_by_profile = dataipral.shape[1]
    fraction_nonzero = nb_nonzero/nb_points_by_profile
    seuil = np.mean(fraction_nonzero.values)
    index_mask = fraction_nonzero > seuil  #selectionner le profil correct    
    return fraction_nonzero, index_mask


# In[6]:


fraction_test, id_crit2 = validated_profile(datacorrected.isel(range=limitez))
# fraction_test, len(np.where(id_crit2)[0]), id_crit2


# Illustration: __le QL de RCS avant d être filtré et l'évolution de la proportion__

# In[175]:


from matplotlib.colors import LogNorm
import matplotlib.dates as mdates
fig, ax2 = plt.subplots(figsize=(7,4))
# fraction_test.plot(x='time', ax=ax2,marker="o")
ax2.scatter(fraction_test['time'].values, fraction_test.values)
ax2.axhline(np.mean(fraction_test.values), linestyle='--', color='red', label='threshold')
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax2.set_ylim(0.0, 1.0)
ax2.legend()
ax2.set(xlabel='time', ylabel='proportion of validared values per profile', 
        title=f'{filepath.stem}')


# Illustration: __le QL filtré__

# In[9]:


Data2 = datacorrected.where(id_crit2, drop=False) # creer un masque 0/1 ou True/False pour where 

plt.clf()
fig, ax = plt.subplots(figsize=(7,6))
Data2.plot(x='time', y='range', norm=LogNorm(vmin=1e2, vmax=1e8), ax=ax, robust=True,cmap='turbo',
                   ylim=(0,20000), cbar_kwargs={'label':'Range & Background corrected signal', 'orientation':'horizontal'})
# ax.scatter(datacorrected[index_mask]['time'].values, [19000]*len(datacorrected[index_mask]['time'].values), color='y')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax.set(title=f'filtre 2: garder les profils ayant proportion des valeurs signficatives > seuil')


# Voir ces proportions spécifiquement du cas 2018-11-06, on peut aussi faire des statistiques pour déterminer un seuil individuel pour chacun du cas de jour.  

# __*Stocker ces index des profils validés dans une variable:*__ id_crit2

# In[68]:


id_crit2 = np.where(id_crit2)[0]


# __Crit 3: Filtrer les nuages à basse altitude avec le filtre de Marc Antoine__

# In[119]:



def ipral_remove_cloud_profiles(alt_max, ipral_file):
    import sys
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
    CHM15K_PATH = Path('/backupfs/sirta/pub/basesirta/1a/chm15k')#Path("/bdd/SIRTA/pub/basesirta/1a/chm15k")
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
    cbh = xr.open_dataset(chm15k_file)["cloud_base_height"][:, 0]#.to_dataframe()["cloud_base_height"]
    # before rounding time, find clouds under the chosen altitude
    cbh_raw = cbh#(cbh > alt_max) | np.isnan(cbh)
    # ---------------------------------------------------------------------------------
    # round time to 15s to ease use
    cbh['time'] = cbh['time'].dt.round(freq=CHM15K_TIME_RES)
#     cbh.index = cbh.index.round(freq=CHM15K_TIME_RES)
    # under sample chm15k data to 30s to have the time resolution as ipral
    cbh = cbh.resample(time=IPRAL_TIME_RES).first()

    # read IPRAL data
    # ----------------
    date = pd.to_datetime(date)
    ipral_data = xr.open_dataset(ipral_file).sel(time=slice(date, date + ONE_DAY))
    raw_profs = ipral_data.time.size
    print(f"{raw_profs} in IPRAL data")

    # get cloud mask
    # ---------------
    # round time to 30s to ease use
#     ipral_time = ipral_data.time.to_dataframe().index.round(freq=IPRAL_TIME_RES)
    ipral_time = ipral_data['time'].dt.round(freq=IPRAL_TIME_RES)
    # only keep timesteps of CBH available in ipral data
#     cbh = cbh.loc[ipral_time]
    cbh = cbh.sel(time=ipral_time)
    # create to only keep data without cloud under the chosen altitude
    cbh_mask = (cbh > alt_max) | np.isnan(cbh)
    profs_to_keep = cbh_mask.values.astype("i2").sum()
    print(f"{raw_profs - profs_to_keep} profiles will be remove")
    # ---------------------------------------------------------------------------------

    # apply mask
    # ---------------------------------------------------------------------------------
    cbh_mask['time'] = ipral_data['time']
    return cbh_mask, cbh_raw


# In[120]:


def ipral_remove_cloud_profiles_v2(alt_max, ipral_file):  
    import sys
    date = ipral_file.name.split('_')[4]
    # read CHM15k file
    # ----------------
    CHM15K_PATH = Path('/backupfs/sirta/pub/basesirta/1a/chm15k')#Path("/bdd/SIRTA/pub/basesirta/1a/chm15k")
    chm15k_file = sorted(CHM15K_PATH.glob(f'**/**/**/chm15k_1a_z1Ppr2R15mF15s_v01_{date}_000000_1440.nc'))
    if not chm15k_file:
        print("No CHM15k file found.")
        print("Quitting.")
        sys.exit()

    chm15k_file = chm15k_file[0]
    print(f"CHM15k file found: {chm15k_file}")
    df_cbh = xr.open_dataset(chm15k_file)["cloud_base_height"][:, 0].to_dataframe()#["cloud_base_height"]
    # read IPRAL data
    # ----------------
    ipral_data = xr.open_dataset(ipral_file)
    ipral_time_array = ipral_data['time'].values
    # get cloud height
    # ---------------    
    cloud_height_array = np.zeros_like(ipral_time_array, dtype='float')    
    for j in range(len(ipral_time_array)):
        print(ipral_time_array[j])
        print(df_cbh.iloc[df_cbh.index.get_loc(ipral_time_array[j], method='nearest')])
        cloud_height_array[j] = df_cbh.iloc[df_cbh.index.get_loc(ipral_time_array[j], method='nearest')]['cloud_base_height']
    
    cloud_over_altmax = (cloud_height_array > alt_max)#|np.isnan(cloud_height_array)
    cloud_mask_xarray = xr.DataArray(data=cloud_over_altmax, 
                                     dims=['time'], 
                                     coords=dict(time=ipral_time_array))
    print(f"{len(ipral_time_array)} in IPRAL data")
    profs_to_keep = cloud_over_altmax.astype("i2").sum()
    print(f"{len(ipral_time_array) - profs_to_keep} profiles will be remove")
    return cloud_mask_xarray


# Attention: ce filtre pourrait être appliqué sur des données ayant ou non calibrées, car il se base sur le ceilomètre pour rendre l'index des profils validés.

# Etape 1: Appliquer ce filtre sur un dataset (un fichier)
# L'altitude choisie est 4km, donc le filtre va chercher et indiquer les profils n'ont pas de nuages __*(cbh=Nan)*__ ou possèdent des nuages > 4km __*(cbh > 4km)*__. 

# In[121]:


filepath = sorted(IPRAL_PATH.glob('**/**/ipral_1a_Lz1R15mF30sPbck_v01_20180909_000000_1440.nc'))[0]
mask_profile, mask_raw = ipral_remove_cloud_profiles(4000, filepath)
id_crit3 = np.where(mask_profile)[0]

mask_profile = ipral_remove_cloud_profiles_v2(4000, filepath)
id_crit3 = np.where(mask_profile)[0]


# In[123]:


id_crit3


# Illustration: __Le QL en 2018-11-06 avant être filtrées__

# In[43]:


timeStart = pd.to_datetime('2018-11-06 12:00:00')
timeEnd = pd.to_datetime('2018-11-06 15:00:00')

from matplotlib.colors import LogNorm
import matplotlib.dates as mdates
fig, ax = plt.subplots()
datacorrected.plot(x='time', y='range', norm=LogNorm(vmin=1e2, vmax=1e8), ylim=(0,20000), #xlim=(timeStart, timeEnd),
                                           cbar_kwargs={'label':'Range & Background corrected signal'})
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax.axhline(4000, linestyle='--', color='k')


# Illustration: __Le QL filtré__

# In[124]:


Data3 = datacorrected.where(mask_profile, drop=False)

fig, ax = plt.subplots(figsize=(7,6))
Data3.plot(x='time', y='range', norm=LogNorm(vmin=1e2, vmax=1e8), ylim=(0,20000), cmap='turbo',#xlim=(timeStart, timeEnd),
           cbar_kwargs={'label':'Range & Background corrected signal', 'orientation':'horizontal'})
ax.set(title='filtre 3: garder les profils sans nuages ou nuages > 4km')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax.axhline(4000, linestyle='--', color='k')


# In[9]:


Data3 = datacorrected.where(mask_profile, drop=False)
pddf1 = mask_raw.to_dataframe('cloud_base_height')
# pddf2 = Data3.to_dataframe('mask_cloud')

cloud_heigh = np.zeros_like(Data3["time"].values, dtype='float')
for i in range(len(datacorrected["time"].values)):
#     print(f'Ipral {Data3["time"].values[i]}')
#     print(pddf1.iloc[pddf1.index.get_loc(Data3["time"].values[i], method='nearest')])
    cloud_heigh[i] = pddf1.iloc[pddf1.index.get_loc(datacorrected["time"].values[i], method='nearest')]['cloud_base_height']


# In[14]:


Data3supp = datacorrected.where(datacorrected['time'].isin(datacorrected['time'].values[(cloud_heigh > 4000)]), drop=False)
#|(np.isnan(cloud_heigh))


# In[17]:


timeStart = pd.to_datetime('2018-11-06 00:00:00')
timeEnd = pd.to_datetime('2018-11-06 23:59:00')

plt.clf()
fig, ax = plt.subplots(figsize=(7,6))
Data3supp.plot(x='time', y='range', norm=LogNorm(vmin=1e2, vmax=1e8), ylim=(0,20000), xlim=(timeStart, timeEnd),
           cbar_kwargs={'label':'Range & Background corrected signal', 'orientation':'horizontal'})
ax.plot(datacorrected['time'].values, cloud_heigh, marker='.')
ax.set(title='filtre 3: garder les profils sans nuages ou nuages > 4km')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax.axhline(4000, linestyle='--', color='k')
ax.set_xlim(timeStart, timeEnd)


# In[275]:



print(cloud_heigh[(datacorrected['time'].values > timeStart)&(datacorrected['time'].values < timeEnd)])
print(datacorrected['time'].values[(datacorrected['time'].values > timeStart)&(datacorrected['time'].values < timeEnd)])


# In[276]:


fig, ax = plt.subplots(figsize=(7,6))
ax.plot(datacorrected['time'].values, cloud_heigh, marker='.')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax.set_xlim(timeStart, timeEnd)
plt.minorticks_on()
ax.grid(b=True, which='minor', color='k', linestyle='--', alpha=0.2)
ax.grid(b=True, which='major', color='k', linestyle='--', alpha=0.2)


# __*Stocker ces index des profils validés dans une variable:*__ id_crit3 = mask_profile

# Profils finaux gardés
# =============
# 
# L'objectif est de retourner l'ensemble des profils validés après les filtres pour intégrer à la suite dans l'étude statistique des SR

# In[125]:


print(f'CRIT 1:{len(id_crit1)}, CRIT 2:{len(id_crit2)}, CRIT 3:{len(id_crit3)}')


# In[126]:


id_total_mask = np.intersect1d(np.intersect1d(id_crit1, id_crit2, return_indices=True)[0],
                            id_crit3, return_indices=True)[0]

time_profil_to_save = datacorrected['time'][id_total_mask].values
print(f'{len(datacorrected.time.values)} profils en total')
print(f'{len(id_total_mask)} profils gardés après les filtres')


# In[127]:


total_mask = np.zeros((len(datacorrected['time'].values),), dtype='bool')
total_mask[id_total_mask] = True
total_mask = xr.DataArray(data = total_mask, 
                          dims = ['time'],
                          coords = dict(time=datacorrected['time'].values),)
print(total_mask)


# Illustration: __le QL final__

# In[128]:


timeStart = pd.to_datetime('2018-11-06 00:00:00')
timeEnd = pd.to_datetime('2018-11-06 23:59:58')

Data_final = datacorrected.where(total_mask, drop=False)

fig, ax = plt.subplots(figsize=(7,6))
Data_final.plot(x='time', y='range', norm=LogNorm(vmin=1e2, vmax=1e8), cmap='turbo',
                ylim=(0,20000), #xlim=(timeStart, timeEnd), 
                cbar_kwargs={'label':'Range & Background corrected signal', 'orientation':'horizontal'})
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax.axhline(4000, linestyle='--', color='k')
ax.set(title=f'{filepath.stem}\n données totalement filtrées')


# Moyennage des données
# ==================
# 
# Créer un compteur des profils dans un paquet de temps avant de moyennage 

# In[185]:


print(f'{filepath}')
IPRAL_LISTFILES = sorted(Path(CALIB_PATH, 'zone-3000-4000').glob(f'{filepath.name}'))
wave = 532

filepath = IPRAL_LISTFILES[0]
filepath


# In[186]:


Data_final = xr.open_dataset(filepath)['calibrated'].sel(wavelength=wave).where(total_mask, drop=False)
Data_final


# In[187]:


nb_profils_mean = Data_final.dropna(dim='time').resample(time='15min').count('time')
Data_final_mean = Data_final.dropna(dim='time').resample(time='15min', skipna=True).mean('time')


# In[190]:


timeStart = pd.to_datetime('2018-12-14 00:00:00')
timeEnd = pd.to_datetime('2018-12-14 15:00:00')
fig, ax = plt.subplots(figsize=(7,6))
Data_final_mean.plot(x='time', y='range', norm=LogNorm(vmin=1e-7, vmax=1e-5), 
                ylim=(0,20000), #xlim=(timeStart, timeEnd), #cmap='turbo',
                cbar_kwargs={'label':'ATB (m-1.sr-1)', 'orientation':'horizontal'})
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax.axhline(4000, linestyle='--', color='k')
ax.set(title=f'{filepath.stem}\n moyenner 15min')


# In[102]:


Data3_mean = Data3.dropna(dim='time').resample(time='15min', skipna=True).mean('time')
Data4 = Data3_mean.where(nb_profils_mean > int(30*0.25), np.nan)
# Data4.sel(time=slice(timeStart, timeEnd))


# In[104]:


Data4 = Data_final_mean.where(nb_profils_mean > int(30*0.25), np.nan)
plt.clf()
fig, ax = plt.subplots()
Data4.plot(x='time', y='range', norm=LogNorm(vmin=1e2, vmax=1e8), 
                ylim=(0,20000), #xlim=(timeStart, timeEnd), #cmap='turbo',
                cbar_kwargs={'label':'Range & Background corrected signal'})
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax.axhline(4000, linestyle='--', color='k')


# In[5]:


def processed(inputfolder, filepath, rangeLimiteZ, wave):
    data = xr.open_dataset(filepath)
    rcs_channel = 'rcs_12' if wave == 355 else 'rcs_16'
    datacorrected = (data[rcs_channel]/np.square(data['range']) - data['bckgrd_'+rcs_channel])*np.square(data['range'])
    # par CRIT 1
    range_limite_top = [26000,28000]
    range_limite_bottom = [2000,3000]
    mask_crit1 = filter_profile_file(datacorrected, rcs_channel, range_limite_top, range_limite_bottom)
    mask_crit1 = np.where(mask_crit1)[0]
    # par CRIT 2
    mask_crit2 = invalidated_profile(datacorrected.values)
    mask_crit2 = np.where(mask_crit2)[0]
    # par CRIT 3
    mask_crit3 = np.where(ipral_remove_cloud_profiles(4000, filepath))
    # id_Z = np.where((data['range']>rangeLimiteZ[0])&(data['range']<rangeLimiteZ[1]))[0]
    # mask_crit3 = flag_clouds(datacorrected, id_Z, filepath, wave=wave)
    # mask_crit3 = np.where(~mask_crit3)[0]
    # TOTAL INDICES
    total_mask = np.intersect1d(np.intersect1d(mask_crit1, mask_crit2, return_indices=True)[0], 
                                mask_crit3, return_indices=True)[0]
    time_to_save = data['time'][total_mask].values
    print(f'wave = {wave}\ntime to save : {time_to_save}')
    return time_to_save


# ## check clouds filter

# In[431]:


# list_time = xr.open_dataset(filepath)['time'].values
# list_time_indice = pd.DataFrame([0]*len(list_time), index=list_time, columns=['indice'])
# list_time_indice['indice'].iloc[mask_profile]=1


# In[59]:


CALIB_PATH = Path('/homedata/nmpnguyen/IPRAL/RF/Calibrated/')
filepath = sorted(CALIB_PATH.glob('**/**/ipral_1a_Lz1R15mF30sPbck_v01_20181106_000000_1440.nc'))[0]
timeStart = pd.to_datetime('2018-11-06 15:00:00')
timeEnd = pd.to_datetime('2018-11-06 16:00:00')


# In[436]:


data_aftermask.isel(time = ((data_aftermask.time > timeStart) & (data_aftermask.time < pd.to_datetime('2018-11-06 15:30:00'))), wavelength=1)['calibrated'].plot.line(y='range', col='time', col_wrap=4, ylim=(0,20000))


# In[437]:


# (data_aftermask.time > timeStart) & (data_aftermask.time < timeEnd)
data_aftermask.isel(time = ((data_aftermask.time > timeStart) & (data_aftermask.time < timeEnd)), wavelength=1, range=range(20,30)).resample(time='15min').mean()['calibrated']


# In[438]:


profiles_validated1 = pd.read_csv('/scratchx/nmpnguyen/IPRAL/raw/detection_clouds_test/IPRAL_2018_validated_profiles1.csv')
profiles_validated1.iloc[112,]


# In[439]:


ceilometer_before = pd.DataFrame(ceilometer_before, columns=['cloud_base_height']).astype(int)
ceilometer_after = pd.DataFrame(ceilometer_after, columns=['cloud_base_height'])
# ceilometer_before[(ceilometer_before.index > pd.to_datetime("2018-11-06 15:15:00"))&(ceilometer_before.index < pd.to_datetime("2018-11-06 15:30:00"))]
ceilometer_after[(ceilometer_after.index > pd.to_datetime("2018-11-06 15:15:00"))&(ceilometer_after.index < pd.to_datetime("2018-11-06 15:30:00"))]


# In[440]:


fig, ax = plt.subplots()
ceilometer_after.reset_index().plot(kind='scatter', x='time', y='cloud_base_height', use_index=True, ax=ax, color='r', label='ceilometer_after')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax.set(title='ceilometer: cloud_base_height')
ax.legend()
ax.set_xlim(timeStart, timeEnd)
ax.axhline(4000, linestyle='--', color='k')
plt.minorticks_on()
ax.grid(b=True, which='minor', color='k', linestyle='--', alpha=0.2)
ax.grid(b=True, which='major', color='k', linestyle='--', alpha=0.2)


# In[441]:


fig, ax = plt.subplots()

ax.scatter(pd.to_datetime(profiles_validated1.iloc[112,]), [4.2]*len(profiles_validated1.iloc[-2,]), color='b', label='profiles validated in data processing')
(list_time_indice*4.5).reset_index().plot(kind='scatter', x='index', y='indice', ax=ax, color='g', label='profiles after filter MA')

# ceilometer_after.reset_index().plot(kind='scatter', x='time', y='cloud_base_height', use_index=True, ax=ax, color='r', label='ceilometer_after')
(ceilometer_before*4.8).reset_index().plot(kind='scatter', x='time', y='cloud_base_height', use_index=True, ax=ax, color='y', label='ceilometer_before')
ax.set_xlim(timeStart, timeEnd)
ax.set(xlabel='time', ylabel=' ', title= ' ')
# ax.set_ylim(0,20000)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax.legend()
plt.minorticks_on()
ax.grid(b=True, which='minor', color='k', linestyle='--', alpha=0.2)
ax.grid(b=True, which='major', color='k', linestyle='--', alpha=0.2)


# In[6]:


year = Path('2018')
IPRAL_PATH = Path('/bdd/SIRTA/pub/basesirta/1a/ipral/', year)
#----------------------------------------------------
# sur tous les fichiers 
altmax = 4000 #m
list_time_indice = None
list_file_error = []
IPRAL_LISTFILES = sorted(IPRAL_PATH.glob('**/**/ipral_1a_Lz1R15mF30sPbck_v01_*_000000_1440.nc'))
for filepath in IPRAL_LISTFILES:
    try :
        mask_profile, ipral_noclouds, ceilometer_before, ceilometer_after = ipral_remove_cloud_profiles(altmax, filepath)
        mask_profile = pd.DataFrame(mask_profile).astype(int)
        if list_time_indice is None:
            list_time_indice = mask_profile
        else:
            list_time_indice = pd.concat([list_time_indice, mask_profile])
    except KeyError:
        list_file_error.append(filepath)
        pass
    


# In[44]:


# list_time_indice.to_csv('/scratchx/nmpnguyen/IPRAL/raw/SR_histogram/clouds_list_under_4000.csv')


# ## Verify SR distribution processing 

# In[32]:


allsr355 = xr.open_dataset('/scratchx/nmpnguyen/IPRAL/raw/SR_histogram/IPRAL_2018_validated_profiles3_allsr355-3000-4000.nc')
allsr532 = xr.open_dataset('/scratchx/nmpnguyen/IPRAL/raw/SR_histogram/IPRAL_2018_validated_profiles3_allsr532-3000-4000.nc')


# In[34]:


xr_concat = xr.concat([allsr355['__xarray_dataarray_variable__'], allsr532['__xarray_dataarray_variable__']], dim='wavelength')#.set_index()
# xrnight = xr_concat.where(((xr_concat.time.dt.hour>=18)|(xr_concat.time.dt.hour<6)), drop=True)

# xrnight355 = allsr355.where(((xr_concat.time.dt.hour>=18)|(xr_concat.time.dt.hour<6)), drop=True)['__xarray_dataarray_variable__']
# xrnight532 = allsr532.where(((xr_concat.time.dt.hour>=18)|(xr_concat.time.dt.hour<6)), drop=True)['__xarray_dataarray_variable__']
# xrday355 = allsr355.where(((xr_concat.time.dt.hour>=6)&(xr_concat.time.dt.hour<18)), drop=True)['__xarray_dataarray_variable__']
# xrday532 = allsr532.where(((xr_concat.time.dt.hour>=6)&(xr_concat.time.dt.hour<18)), drop=True)['__xarray_dataarray_variable__']
# dfnight = pd.concat([allsr355['__xarray_dataarray_variable__'].drop('wavelength').to_dataframe(['sr355']), allsr532['__xarray_dataarray_variable__'].drop('wavelength').to_dataframe(['sr532'])], axis='index')
# df_concat = df_concat.reset_index(level='range')
# df_concat['hour'] = df_concat.index.hour


# In[66]:


def get_params_histogram(srlimite, X532, Y355):
    if len(X532[np.where(~np.isnan(X532))]) > len(Y355[np.where(~np.isnan(Y355))]):
        H = np.histogram2d(X532[~np.isnan(Y355)], Y355[~np.isnan(Y355)], bins=100, range = srlimite)
        print('A')
        Hprobas = H[0]*100/len(Y355[~np.isnan(Y355)])
        noNaNpoints = len(Y355[~np.isnan(Y355)])
    elif len(X532[np.where(~np.isnan(X532))]) < len(Y355[np.where(~np.isnan(Y355))]):
        H = np.histogram2d(X532[~np.isnan(X532)], Y355[~np.isnan(X532)], bins=100, range = srlimite)
        print('B')
        Hprobas = H[0]*100/len(X532[~np.isnan(X532)])
        noNaNpoints = len(Y355[~np.isnan(Y355)])
    else:
        H = np.histogram2d(X532[np.where(~np.isnan(X532))], Y355[np.where(~np.isnan(Y355))], bins=100, range = srlimite)
        print('C')
        Hprobas = H[0]*100/len(X532[~np.isnan(X532)])
        noNaNpoints = len(X532[~np.isnan(X532)])
    print(f'nombre de points no-NaN: {noNaNpoints}')
    xedges, yedges = np.meshgrid(H[1], H[2])
    return xedges, yedges, Hprobas, noNaNpoints


# In[37]:


get_ipython().magic('matplotlib notebook')
from matplotlib.colors import LogNorm
from scipy import stats

Y532 = allsr532['__xarray_dataarray_variable__']#xrnight532.values.ravel()
X355 = allsr355['__xarray_dataarray_variable__']#xrnight355.values.ravel()
ff, (ax,ax2) = plt.subplots(figsize=[12,5], ncols=2)
plt.minorticks_on()
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
'''
DAY
'''
Xxedges, Yyedges, probas, points = get_params_histogram(sr_limite, X355.values.ravel(), Y532.values.ravel())
p = ax.pcolormesh(Xxedges, Yyedges, probas.T, norm = LogNorm(vmax=1e0, vmin=1e-5))
# ax.plot(X532[np.where(~np.isnan(X532))], fitLine2, '-.', c='r')
c = plt.colorbar(p, ax=ax, label='%')
ax.set(ylabel='SR532', xlabel='SR355', 
       title= f'IPRAL - Resolution 15min x 15m, \n{points} points, {len(allsr532.time)} profiles')#\nLinearRegression: {round(slope,5)}x + {round(intercept,3)}
ax.set(xlim=(-5,40), ylim=(-10,80))
ax.grid(b=True, which='minor', color='k', linestyle='--', alpha=0.2)
ax.grid(b=True, which='major', color='k', linestyle='--', alpha=0.2)
# '''
# NIGHT
# '''
# Xxedges, Yyedges, probas, points = get_params_histogram(sr_limite, xrnight532.values.ravel(), xrnight355.values.ravel())
# p = ax2.pcolormesh(Xxedges, Yyedges, probas.T, norm = LogNorm(vmax=1e0, vmin=1e-5))
# # ax.plot(X532[np.where(~np.isnan(X532))], fitLine2, '-.', c='r')
# c = plt.colorbar(p, ax=ax2, label='%')
# ax2.set(xlabel='SR532', ylabel='SR355', 
#        title= f'IPRAL - NIGHT, \n{points} points, {len(allsr532.time)} profiles')#\nLinearRegression: {round(slope,5)}x + {round(intercept,3)}
# ax2.set(xlim=(-10,sr_limite), ylim=(-10,sr_limite))
# plt.grid(b=True, which='minor', color='k', linestyle='--', alpha=0.2)
# plt.grid(b=True, which='major', color='k', linestyle='--', alpha=0.2)


# In[102]:


# allsr532 = allsr532['__xarray_dataarray_variable__']
# allsr355 = allsr355['__xarray_dataarray_variable__']
allsr355supp = allsr355.where((allsr532 >10))
allsr532supp = allsr532.where((allsr532 >10))


# In[103]:


# np.where(~np.isnan(allsr532['__xarray_dataarray_variable__'].values.ravel())&~np.isinf(allsr532['__xarray_dataarray_variable__'].values.ravel()))#,np.where(np.isnan(allsr355['__xarray_dataarray_variable__'].values.ravel()))

# Y = allsr532['__xarray_dataarray_variable__'].values.ravel()[~np.isnan(allsr532['__xarray_dataarray_variable__'].values.ravel())&~np.isinf(allsr532['__xarray_dataarray_variable__'].values.ravel())]
# X = allsr355['__xarray_dataarray_variable__'].values.ravel()[~np.isnan(allsr532['__xarray_dataarray_variable__'].values.ravel())&~np.isinf(allsr532['__xarray_dataarray_variable__'].values.ravel())]

Y = allsr532supp.values.ravel()[~np.isnan(allsr532supp.values.ravel())&~np.isinf(allsr532supp.values.ravel())]
X = allsr355supp.values.ravel()[~np.isnan(allsr532supp.values.ravel())&~np.isinf(allsr532supp.values.ravel())]

# Test function with coefficients as parameters
def test(x, a, b):
    return a * x + b 

from scipy.optimize import curve_fit
param, param_cov = curve_fit(test, X, Y, p0=[0,0])
print(param, param_cov)


# In[110]:


# Get the standard deviations of the parameters (square roots of the # diagonal of the covariance)
stdevs = np.sqrt(np.diag(param_cov))
print(stdevs)


# In[114]:


from scipy import stats
# Define confidence interval.
ci = 0.95
# Convert to percentile point of the normal distribution.
# See: https://en.wikipedia.org/wiki/Standard_score
pp = (1. + ci) / 2.
# Convert to number of standard deviations.
nstd = stats.norm.ppf(pp)
print(nstd)


# In[115]:


param_up = param + nstd*stdevs
param_down = param - nstd*stdevs


# In[120]:


def predband(x, xd, yd, p, func, conf=0.95):
    # x = requested points
    # xd = x data
    # yd = y data
    # p = parameters
    # func = function name
    alpha = 1.0 - conf    # significance
    N = xd.size          # data sample size
    var_n = len(p)  # number of parameters
    # Quantile of Student's t distribution for p=(1-alpha/2)
    q = stats.t.ppf(1.0 - alpha / 2.0, N - var_n)
    # Stdev of an individual measurement
    se = np.sqrt(1. / (N - var_n) *                  np.sum((yd - func(xd, *p)) ** 2))
    # Auxiliary definitions
    sx = (x - xd.mean()) ** 2
    sxd = np.sum((xd - xd.mean()) ** 2)
    # Predicted values (best-fit model)
    yp = func(x, *p)
    # Prediction band
    dy = q * se * np.sqrt(1.0+ (1.0/N) + (sx/sxd))
    # Upper & lower prediction bands.
    lpb, upb = yp - dy, yp + dy
    return lpb, upb


# In[121]:


predband(np.linspace(0, 20, 50), X, Y, param, test, conf=0.95)


# In[119]:


sr_limite = [[-10,40],[-10,100]]
XH, YH, probas, points = get_params_histogram(sr_limite, X, Y)

get_ipython().magic('matplotlib notebook')
fig, ax = plt.subplots()
ax.plot(X, Y, '.')
xX = np.linspace(0, 20, 50)
# p = ax.pcolormesh(XH, YH, probas.T, norm = LogNorm())
ax.plot(xX, test(xX, *param), color='k')
ax.plot(xX, test(xX, *param_up), color='r', linestyle='--')
ax.plot(xX, test(xX, *param_down), color='r', linestyle='--')
# c = plt.colorbar(p, ax=ax, label='%')


# In[109]:


allsr532_new = test(allsr355.values.ravel(), *param)
sr_limite = [[-10,40],[-10,100]]
XH, YH, probas, points = get_params_histogram(sr_limite, allsr355.values.ravel(), allsr532.values.ravel())

f, ax = plt.subplots()
p = ax.pcolormesh(XH, YH, probas.T, norm = LogNorm())
ax.plot(X, test(X, *param), color='k')
# ax.plot(allsr355.values.ravel(), test(allsr355.values.ravel(), *param), color='r')


# *Illustration le filtre sur un fichier de données:*
# dont le ql est en "Range and Background corrected signal" et scatter plot représente la fraction par profil avec le seuil de filtrage. Les profils gardés, dits validés, a la fraction supérieur à seuil. 

# ## Distribution ATB532 & ATB355

# In[122]:


allatb355 = xr.open_dataset('/scratchx/nmpnguyen/IPRAL/raw/SR_histogram/IPRAL_2018_validated_profiles4_allatb355.nc')
allatb532 = xr.open_dataset('/scratchx/nmpnguyen/IPRAL/raw/SR_histogram/IPRAL_2018_validated_profiles4_allatb532.nc')


# In[123]:


allatb355


# In[142]:


np.log(X355)[~np.isnan(np.log(X355))].max()


# In[153]:


get_ipython().magic('matplotlib notebook')
from matplotlib.colors import LogNorm
from scipy import stats

Y532 = allatb532['calibrated'].values.ravel()
X355 = allatb355['calibrated'].values.ravel()
ff, ax = plt.subplots(figsize=[6,5])
plt.minorticks_on()
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12


Xxedges, Yyedges, probas, points = get_params_histogram([[-20, -5],[-20, -5]], np.log(X355), np.log(Y532))
p = ax.pcolormesh(Xxedges, Yyedges, probas.T, norm = LogNorm(vmax=1e0, vmin=1e-5))
# ax.plot(X532[np.where(~np.isnan(X532))], fitLine2, '-.', c='r')
c = plt.colorbar(p, ax=ax, label='%')
ax.set(ylabel='ATB532', xlabel='ATB355', 
       title= f'IPRAL - Resolution 15min x 15m, \n{points} points, {len(allsr532.time)} profiles')#\nLinearRegression: {round(slope,5)}x + {round(intercept,3)}
# ax.set(xlim=(-5,40), ylim=(-10,80))
ax.grid(b=True, which='minor', color='k', linestyle='--', alpha=0.2)
ax.grid(b=True, which='major', color='k', linestyle='--', alpha=0.2)


# ## check flagcode

# In[4]:


get_ipython().system('cat /scratchx/nmpnguyen/IPRAL/raw/detection_clouds_test/detection_clouds_and_flags.py')


# In[9]:


year = Path('2018')
IPRAL_PATH = Path('/bdd/SIRTA/pub/basesirta/1a/ipral/', year)
IPRAL_LISTFILES = sorted(IPRAL_PATH.glob('**/**/ipral_1a_Lz1R15mF30sPbck_v01_*_000000_1440.nc'))
print('#_______insert npy file to study list of bugs______')
listdatenpy = np.unique(pd.to_datetime(np.load('/scratchx/nmpnguyen/IPRAL/raw/SR_histogram/time_sr355.40_sr532.0.npy')).strftime('%Y%m%d'))
print(listdatenpy)
IPRAL_LISTFILES = [sorted(IPRAL_PATH.glob(f'**/**/ipral_1a_Lz1R15mF30sPbck_v01_{l}_000000_1440.nc'))[0] for l in listdatenpy]
print(len(IPRAL_LISTFILES))


# In[27]:


wave=355

data = xr.open_dataset(IPRAL_LISTFILES[3])

rcs_channel = 'rcs_12' if wave == 355 else 'rcs_16'
datacorrected = (data[rcs_channel]/np.square(data['range']) - data['bckgrd_'+rcs_channel])*np.square(data['range'])


# In[54]:


# par CRIT 1
range_limite_top = [26000,28000]
range_limite_bottom = [2000,3000]
mask_crit1 = filter_profile_file(datacorrected, rcs_channel, range_limite_top, range_limite_bottom)
mask_crit1 = np.where(mask_crit1)[0]
# par CRIT 2
mask_crit2 = invalidated_profile(datacorrected.values)
mask_crit2 = np.where(mask_crit2)[0]
# par CRIT 3
rangeLimiteZ=[3000, 20000]
id_Z = np.where((data['range']>rangeLimiteZ[0])&(data['range']<rangeLimiteZ[1]))[0]
mask_crit3 = np.where(ipral_remove_cloud_profiles(1000, IPRAL_LISTFILES[3])) #flag_clouds(datacorrected, id_Z, IPRAL_LISTFILES[3], wave=wave) #
# mask_crit3 = np.where(~mask_crit3)[0]
# TOTAL INDICES
total_mask = np.intersect1d(np.intersect1d(mask_crit1, mask_crit2, return_indices=True)[0], 
                            mask_crit3, return_indices=True)[0]
time_to_save = data['time'][total_mask].values
print(f'wave = {wave}\ntime to save : {time_to_save}')


# In[46]:


from matplotlib.colors import LogNorm
fig, ax = plt.subplots(figsize=(9,7))
plt.rcParams['font.size']=12
datacorrected.plot(y='range', x='time', ax=ax, norm=LogNorm(vmin=1e4), robust=True, ylim=(0,20000))
# ax.axvline(np.array(time_to_save, dtype='datetime64[ns]'), color='white', linestyle='--')
# ax.axvline(np.array(time_to_save[1], dtype='datetime64[ns]'), color='white', linestyle='--')
ax.axhline(5000, color='k')


# In[61]:


pd.DataFrame([time_to_save, time_to_save[:10]]).to_csv('test.csv', index=False, mode='w')


# ### Verifier profils bugs

# In[19]:


print('#_______insert npy file to study list of bugs______')
listdatenpy = np.unique(pd.to_datetime(np.load('/scratchx/nmpnguyen/IPRAL/raw/SR_histogram/time_sr355.40_sr532.0.npy')).strftime('%Y%m%d'))
print(listdatenpy)
IPRAL_LISTFILES = [sorted(IPRAL_PATH.glob(f'**/**/ipral_1a_Lz1R15mF30sPbck_v01_{l}_000000_1440.nc'))[0] for l in listdatenpy]
print(IPRAL_LISTFILES)
list_times_final = []
file = IPRAL_LISTFILES[0]
# # for file in tqdm.tqdm(IPRAL_LISTFILES[3:5]):
# #     # idsmask1, idsmask2, idsmask3, idsmask_total = processed(IPRAL_PATH, OUTPUT_PATH, file, rangeLimiteZ=[3000, 20000], wave=532)
time_to_save532 = processed(IPRAL_PATH, file, rangeLimiteZ=[2000, 20000], wave=532)
time_to_save355 = processed(IPRAL_PATH, file, rangeLimiteZ=[2000, 20000], wave=355)
list_times_final.append(np.intersect1d(time_to_save532, time_to_save355))


# In[20]:


data = xr.open_dataset(file)
print(len(list_times_final[0]), len(data['time'].values))
time_valid = list_times_final[0]
time_no_valid = np.setxor1d(time_valid, data['time'].values)
print(len(time_valid), len(time_no_valid))


# In[5]:


# profils = pd.read_csv('/scratchx/nmpnguyen/IPRAL/raw/detection_clouds_test/IPRAL_2018_validated_profiles3.csv')


# In[73]:


profiles_bug = np.load('/scratchx/nmpnguyen/IPRAL/raw/SR_histogram/time_sr355.40_sr532.0_v3.npy')
profiles_bug = profiles_bug[pd.to_datetime(profiles_bug).strftime('%Y%m%d')=='20181214']
len(profiles_bug), profiles_bug


# In[71]:


print(len(np.intersect1d(profiles_bug, time_valid)))
print((np.intersect1d(profiles_bug, time_no_valid)))


# In[65]:


def sr_by_files(csv_listdates, calib_pathfolder):
    date_of_list = pd.to_datetime(csv_listdates[1]).strftime('%Y%m%d')
    IPRAL_FILE_MASK = f'ipral_1a_Lz1R15mF30sPbck_v01_{str(date_of_list)}_000000_1440.nc'
    file = sorted(calib_pathfolder.glob(f'{IPRAL_FILE_MASK}'))[0]
#     idfile = np.where([file.stem.split('_')[4] == str(date_of_list) for file in calib_listfiles])[0][0]
    datacalib = xr.open_dataset(file) 
    limiteZ = (datacalib['range']>5000)&(datacalib['range']<15000)
#     time_valid = np.intersect1d(datacalib['time'].values, np.array(pd.to_datetime(csv_listdates)))
#     time_valid = datacalib['time'][indice].values
    sr355 = (datacalib['calibrated']/datacalib['simulated']).sel(wavelength = 355)#, time=csv_listdates).isel(range=limiteZ)
    sr532 = (datacalib['calibrated']/datacalib['simulated']).sel(wavelength = 532)#, time=csv_listdates).isel(range=limiteZ)
#     print(np.where((sr355>39)&(sr355<40)&(sr532>0)&(sr532<1)))
    idx = np.where((sr355>39)&(sr355<40)&(sr532>0)&(sr532<1))
    time_outlier = sr355['time'].values[idx[0]]
    range_outlier = sr355['range'].values[idx[1]]
    return csv_listdates, time_outlier, range_outlier, sr355, sr532


# In[66]:


time_valid_from_SR, time_outlier_from_SR, range_outlier_from_SR, SR355, SR532 = sr_by_files(csv_listdates = time_valid, calib_pathfolder = Path('/homedata/nmpnguyen/IPRAL/RF/Calibrated/'))
# print(np.setxor1d(time_valid_from_SR, time_valid))
print(np.intersect1d(time_outlier_from_SR, time_no_valid))
print((time_valid_from_SR))


# In[54]:


# from matplotlib.colors import LogNorm
# data = xr.open_dataset(IPRAL_LISTFILES[-1])
# datacorrected = (data['rcs_12']/np.square(data['range']) - data['bckgrd_'+'rcs_12'])*np.square(data['range'])

fig, ax = plt.subplots(figsize=(9,7))
datacorrected.plot(x='time', y='range', robust=True, ylim=(0,14000), norm=LogNorm(vmin=1e4), ax=ax)
ax.scatter(time_valid, np.array([1000]*len(time_valid)), color='r')
ax.scatter(time_outlier_from_SR, range_outlier_from_SR, color='yellow')
ax.vlines(np.array(time_outlier_from_SR, dtype='datetime64[ns]'),ymin=0, ymax=14000, color='yellow')


# In[70]:


fig, (ax, ax2) = plt.subplots(figsize=(9,12), nrows=2)
SR355.plot(x='time', y='range', ax=ax, vmin=-1, vmax= 20, robust=True, ylim=(0,14000))
ax.scatter(time_outlier_from_SR, range_outlier_from_SR, color='yellow')
SR532.plot(x='time', y='range', ax=ax2, vmin=-1, vmax= 20, robust=True, ylim=(0,14000))
ax2.scatter(time_outlier_from_SR, range_outlier_from_SR, color='yellow')


# ## TEST DENOISING SIGNALS

# In[3]:


#%% Simple example   
import numpy as np  
import matplotlib.pyplot as plt  
from vmdpy import VMD 


# In[4]:


#. Time Domain 0 to T  
T = 1000  
fs = 1/T  
t = np.arange(1,T+1)/T  
freqs = 2*np.pi*(t-0.5-fs)/(fs)  
print(t, freqs)


# In[5]:


#. center frequencies of components  
f_1 = 2  
f_2 = 24  
f_3 = 288  

#. modes  
v_1 = (np.cos(2*np.pi*f_1*t))  
v_2 = 1/4*(np.cos(2*np.pi*f_2*t))  
v_3 = 1/16*(np.cos(2*np.pi*f_3*t))  

f = v_1 + v_2 + v_3 + 0.1*np.random.randn(v_1.size)  


# In[32]:


#. some sample parameters for VMD  
alpha = 2000       # moderate bandwidth constraint  
tau = 1.08            # noise-tolerance (no strict fidelity enforcement)  
K = 5            # 3 modes  
DC = 0             # no DC part imposed  
init = 1           # initialize omegas uniformly  
tol = 1e-6  


# In[33]:


#. Run actual VMD code  
u, u_hat, omega = VMD(f, alpha, tau, K, DC, init, tol)  


# In[36]:


#. Visualize decomposed modes
plt.figure(figsize=(15,9))
plt.subplot(2,1,1)
plt.plot(f)
plt.title('Original signal')
plt.xlabel('time (s)')
plt.subplot(2,1,2)
plt.plot(f)
plt.plot(u[0])
# plt.plot(u.T)
plt.title('Decomposed modes')
plt.xlabel('time (s)')
# plt.legend(['original', 'decomposed'])
plt.legend(['original',['Mode %d'%m_i for m_i in range(u.shape[0])]])
plt.tight_layout()


# In[181]:


data = xr.open_dataset(list(Path('/homedata/nmpnguyen/IPRAL/RF/Calibrated/').glob('*20180224*1440.nc'))[0])
limitez = (data['range']<14000)&(data['range']>1000)
signal_to_vmd = (data['calibrated']/data['simulated']).isel(time=-2, wavelength=0, range=limitez)


# In[182]:


#. some sample parameters for VMD  
alpha = 2050       # moderate bandwidth constraint  
tau = 1.22          # noise-tolerance (no strict fidelity enforcement)  
K = 6           #  modes  
DC = 0            # no DC part imposed  
init = 1          # initialize omegas uniformly  
tol = 1e-7  

#. Run actual VMD code  
u, u_hat, omega = VMD(signal_to_vmd, alpha, tau, K, DC, init, tol)  


# In[185]:


#. Visualize decomposed modes
plt.figure(figsize=(15,4))
plt.plot(signal_to_vmd)
plt.plot(u[0])
# plt.plot(u.T)
plt.title(f'Decomposed modes {signal_to_vmd.time.values}')
plt.xlabel('time (s)')
plt.legend(['original', 'denoised'])
# plt.legend(['original',['Mode %d'%m_i for m_i in range(u.shape[0])]])
plt.grid()
plt.xlim(300,800)
# plt.ylim(0,1.5)
plt.tight_layout()


# In[184]:


rmse_compute = np.sqrt(1/len(signal_to_vmd)*np.sum(np.square(signal_to_vmd[1:] - u[0])))
print(rmse_compute.values, np.mean(signal_to_vmd))


# In[165]:


def get_denoised_signal(signal_input):
    #. some sample parameters for VMD  
    alpha = 2000       # moderate bandwidth constraint  
    tau = 0.5          # noise-tolerance (no strict fidelity enforcement)  
    K = 6           #  modes  
    DC = 0            # no DC part imposed  
    init = 1          # initialize omegas uniformly  
    tol = 1e-7  
    
    #. Run actual VMD code  
    decomposed_signal, u_hat, omega = VMD(signal_to_vmd, alpha, tau, K, DC, init, tol)  
    return decomposed_signal[0], rmse_compute


# In[146]:


data_to_vmd = (data['calibrated']/data['simulated']).isel(wavelength=0, range=limitez).values


# In[139]:


denoised_data = np.array([get_denoised_signal(data_to_vmd[i,:]) for i in range(20)])


# In[167]:


#. Visualize decomposed modes
fig, axs = plt.subplots(figsize=(15,4*10), ncols=2, nrows=10, sharex=True)
for i, ax in enumerate(axs.flatten()):
    ax.plot(data_to_vmd[i,:])
    ax.plot(denoised_data[i,:])
    rmse_signal = np.sqrt(1/len(data_to_vmd[i,:])*np.sum(np.square(data_to_vmd[i,1:]- denoised_data[i,:])))
    ax.set(title=f'{data.time.values[i]}, {rmse_signal}', xlabel='range (m)')
#     ax.xlabel('range (m)')
    ax.legend(['original', 'denoised'])
    # plt.legend(['original',['Mode %d'%m_i for m_i in range(u.shape[0])]])
    ax.grid()
    plt.xlim(500,800)
    # plt.ylim(0,1.5)
#     plt.tight_layout()


# In[164]:


rmse_compute = np.sqrt(1/len(data_to_vmd)*np.sum(np.square(data_to_vmd[:20,1:] - denoised_data), axis=1))
print(rmse_compute, np.mean(signal_to_vmd))


# In[26]:


profiles_validated1 = pd.read_csv('/scratchx/nmpnguyen/IPRAL/raw/detection_clouds_test/IPRAL_2018_validated_profiles1.csv')


# In[27]:


print(profiles_validated1.iloc[76])


# In[28]:


selecttime = np.array(pd.to_datetime(profiles_validated1.iloc[76]))#.astype('datetime64[ns]')
selecttime = selecttime[~np.isnan(selecttime)]
selecttime[584]


# In[29]:


CALIB_PATH = Path('/homedata/nmpnguyen/IPRAL/RF/Calibrated/')
IPRAL_LISTFILES = sorted(CALIB_PATH.glob('**/ipral_1a_Lz1R15mF30sPbck_v01_20180909_000000_1440.nc'))
print(IPRAL_LISTFILES[0])

dataselect = xr.open_dataset(IPRAL_LISTFILES[0], decode_times=True)
limitez = dataselect['range'] < 20000
dataselect.isel(time=584)


# In[215]:


from matplotlib.colors import LogNorm
import matplotlib.dates as mdates
# Data_final = dataselect.sel(wavelength=355)['calibrated'].where(total_mask, drop=False)
fig, ax = plt.subplots(figsize=(7,6))
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
(Data_final*1e3).plot(x='time', y='range', norm=LogNorm(vmin=1e-4, vmax=1e-1), ax=ax, robust=True, cmap='turbo',
                   ylim=(0,20000), cbar_kwargs={'label':'Backscatter, 1/km.sr', 'orientation':'horizontal'})
# ax.scatter(datacorrected[index_mask]['time'].values, [19000]*len(datacorrected[index_mask]['time'].values), color='y')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax.set(ylabel='Altitude, m', xlabel='Time',
      title=f'IPRAL - Attenuated backscatter\n{IPRAL_LISTFILES[0].stem}\n')


# In[17]:


dataselected = dataselect.sel(time = selecttime[584]).isel(range=limitez)
print(dataselected)
fig, (ax, ax2) = plt.subplots(figsize=(5,12), nrows=2)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
(dataselected.sel(wavelength=355)['simulated']*1e3).plot(y='range', xlim=(1e-4, 1e-1), ylim=(0,20000), xscale='log',
                                                                label='355: ATBmol', color='b', ax=ax)
(dataselected.sel(wavelength=532)['simulated']*1e3).plot(y='range', xlim=(1e-4, 1e-1), ylim=(0,20000), xscale='log',
                                                                label='532: ATBmol', color='g', ax=ax)
ax.legend()
ax.set(xlabel='Backscatter, 1/km.sr', ylabel='Range above ground level, m', title='Molecular Backscatter profile')

(dataselected.sel(wavelength=355)['calibrated']*1e3).plot(y='range', xlim=(1e-4, 1e-1), ylim=(0,20000), xscale='log', zorder=2,
                                                                       label='355: ATB attn', color='b', ax=ax2)
(dataselected.sel(wavelength=532)['calibrated']*1e3).plot(y='range', xlim=(1e-4, 1e-1), ylim=(0,20000), xscale='log', zorder=2,
                                                                       label='532: ATB attn', color='g', ax=ax2)
(dataselected.sel(wavelength=355)['simulated']*1e3).plot(y='range', xlim=(1e-4, 1e-1), ylim=(0,20000), xscale='log', zorder=10,
                                                                label='355: ATBmol attn', color='b', linestyle='--', ax=ax2)
(dataselected.sel(wavelength=532)['simulated']*1e3).plot(y='range', xlim=(1e-4, 1e-1), ylim=(0,20000), xscale='log', zorder=12,
                                                                label='532: ATBmol attn', color='g', linestyle='--', ax=ax2)
ax2.axhspan(ymin=5000, ymax=7000, alpha=0.2, color='y', label='[z_min, z_max]')
ax2.legend()
ax2.set(xlabel='Backscatter, 1/km.sr', ylabel='Range above ground level, m', title='Total Attenuated Backscatter profile')


# In[30]:


data_mean = dataselect.resample(time='15min').mean(dim='time')
data_mean.isel(time=31)


# In[31]:


get_ipython().magic('matplotlib notebook')
import matplotlib.pyplot as plt

fig, ax2 = plt.subplots(figsize=(5,6))
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
(data_mean['calibrated']/data_mean['simulated']).isel(time=31, range=limitez).sel(wavelength=355).plot(y='range', ylim=(0,20000), ax=ax2,
                                                                         label='355:SR', color='g')
(data_mean['calibrated']/data_mean['simulated']).isel(time=31, range=limitez).sel(wavelength=532).plot(y='range', ylim=(0,20000), ax=ax2,
                                                                         label='532:SR', color='b')
ax2.axvline(1, linestyle='--', zorder=10, color='k', label='SR=1')
ax2.axhspan(ymin=5000, ymax=7000, alpha=0.2, color='y', label='[z_min, z_max]')
ax2.legend()
ax2.set(xlabel='SR', ylabel='Range above ground level, m', title='Scattering Ratio profile, 15min x 15m')


# In[13]:


from skimage.measure import block_reduce

np.mean((data_mean['calibrated']/data_mean['simulated']).isel(time=31, range=limitez).values.reshape((-1, int(limitez.sum()/4))), axis=1)


# In[20]:


1333/3

