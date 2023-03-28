
# coding: utf-8

# In[2]:


import xarray as xr 
from pathlib import Path
import numpy as np 
import pandas as pd 
import glob, os
import matplotlib.pyplot as plt
import scipy.interpolate as spi
from datetime import datetime, timedelta


# In[3]:


pd.__version__


# ERA5 for IPRAL
# ========

# In[2]:


oparpath = Path('/home/nmpnguyen/OPAR/LIO3T.daily/2019-04-24.nc4')
d = xr.open_dataset(oparpath)


# In[4]:


print('-----GET IPRAL BCK CORRECTED FILE-----')

time = d.time.values
YEAR = pd.to_datetime(time[0]).strftime('%Y')
MONTH = pd.to_datetime(time[0]).strftime('%m')
lon_opar = round(4*float(d['signal'].longitude))/4 #round(float(d.geospatial_lon_min),2)
lat_opar = round(4*float(d['signal'].latitude))/4
print(f'longitude: {lon_opar}')
print(f'latitude: {lat_opar}')
#----


# In[5]:


print('-----GET ERA5 FILE-----')
ERA_FOLDER = Path("/bdd/ERA5/NETCDF/GLOBAL_025/hourly/AN_PL")
ERA_FILENAME = YEAR+MONTH+".ap1e5.GLOBAL_025.nc"
GEOPT_PATH = ERA_FOLDER / YEAR / Path("geopt."+ERA_FILENAME)
TA_PATH = ERA_FOLDER / YEAR / Path("ta."+ERA_FILENAME)
print(f'path of temperature {TA_PATH}')
print(f'path of geopotential {GEOPT_PATH}')
geopt = xr.open_dataset(GEOPT_PATH)
ta = xr.open_dataset(TA_PATH)
#----


# In[7]:


print('-----CONVERT TIME AND LOCALISATION-----')
# date_start = pd.to_datetime(time[0])
# date_end = pd.to_datetime(time[-1])
time = pd.to_datetime(time).strftime('%Y-%m-%dT%H:00:00.000000000')
time = time.astype('datetime64[ns]')
time_unique = np.unique(time)
LAT = geopt.latitude[np.where(np.abs(geopt.latitude.values - lat_opar) <=0.25)[0][1]].values
LON = geopt.longitude[np.where(np.abs(geopt.longitude.values - lon_opar) <=0.25)[0][1]].values
#----
from timeit import default_timer as timer
TIME = timer()
geopt_for_ipral = geopt.sel(time=time_unique, latitude=LAT, longitude=LON).to_dataframe()#['geopt']
ta_for_ipral = ta.sel(time=time_unique, latitude=LAT, longitude=LON).to_dataframe()#['ta']
print(f'Time loading {timer()-TIME}')
#----


# In[8]:


print('-----GETTING PRESSURE AND TEMPERATURE-----')
lat_opar = np.deg2rad(lat_opar)
acc_gravity = 9.78032*(1+5.2885e-3*(np.sin(lat_opar))**2 - 5.9e-6*(np.sin(2*lat_opar))**2)
r0 = 2*acc_gravity/(3.085462e-6 + 2.27e-9*np.cos(2*lat_opar) - 2e-12*np.cos(4*lat_opar))
g0 = 9.80665
geopt_for_ipral['geopt_height'] = geopt_for_ipral["geopt"]/g0
geopt_for_ipral['altitude'] = (geopt_for_ipral['geopt_height']*r0)/(acc_gravity*r0/g0 - geopt_for_ipral['geopt_height'])
M = 28.966E-3 
R = 8.314510
T = (15 + 273.15)
const = -(M*g0)/(R*T)
p0 = 101325
geopt_for_ipral['pression'] = p0*np.exp(const*geopt_for_ipral['altitude'])
output_era = pd.merge(geopt_for_ipral, ta_for_ipral['ta'], left_index=True, right_index=True) 
print('variables_from_era --> end')


# In[75]:


import scipy.interpolate as spi
def interpolate_atb_mol(lidar_name, opar_file, era):     
    """
    the Input is the output dataframe of simulate_atb_mol function
    """
    print('-----BEFORE INTERPOLATE-----')
    d = xr.open_dataset(opar_file)
    r = d.range.values*1e3 + 2160
    timeOpar = d.time.values
    timeEra = np.unique(era.index.get_level_values(1)) 
    time_tmp = np.array(pd.to_datetime(timeOpar).strftime('%Y-%m-%dT%H:00:00')).astype('datetime64[ns]')
    if len(time_tmp) != len(timeOpar):
        print("Time Error")
        sys.exit(1)
    #------
    columns_names = ['altitude', 'pression', 'ta']#, 'beta355mol', 'beta532mol', 'beta355', 'beta532', 'alpha355', 'alpha532', 'tau355', 'tau532'
    pression_interp, ta_interp = [[] for _ in range(len(columns_names)-1)] #beta355mol_interp ,beta532mol_interp, beta355_interp ,beta532_interp ,tau355_interp ,tau532_interp ,alpha355_interp ,alpha532_interp ,
    new_index = pd.MultiIndex.from_product([timeEra, r], names = ['time', 'range'])
    # df_new = pd.DataFrame(index = new_index, columns = era.columns)
    print('-----INTERPOLATE ATTENUATED BACKSCATTERING FROM ERA5-----')
    for t1 in timeEra:
        a = era.loc[pd.IndexSlice[:, t1], columns_names]
        f9 = spi.interp1d(a['altitude'], a['pression'], kind = 'linear', bounds_error=False, fill_value="extrapolate")
        f10 = spi.interp1d(a['altitude'], a['ta'], kind = 'linear', bounds_error=False, fill_value="extrapolate")
        pression_interp, ta_interp = np.append(pression_interp, np.array(f9(r))), np.append(ta_interp, np.array(f10(r)))

    new_df = pd.DataFrame(index = new_index, data = np.array([pression_interp, ta_interp]).T, columns = columns_names[1:]) # ,beta355mol_interp ,beta532mol_interp
    #, beta355_interp ,beta532_interp ,alpha355_interp ,alpha532_interp ,tau355_interp ,tau532_interp
    
    print(Path("/homedata/nmpnguyen/OPAR/Processed/",lidar_name.upper(),opar_file.name.split('.')[0]+"_simul.pkl"))
#     new_df.to_pickle(Path("/homedata/nmpnguyen/OPAR/Processed/",lidar_name.upper(),opar_file.name.split('.')[0]+"_simul.pkl"))
    print('interpolate_atb_mol --> end')
    return new_df


# In[76]:


output = interpolate_atb_mol("lio3t", oparpath, output_era)


# In[69]:


pres = output_era['pression'].unstack(level=1).iloc[:,0]
ta = output_era['ta'].unstack(level=1).iloc[:,0]
alt = output_era['altitude'].unstack(level=1).iloc[:,0]


# In[62]:


'''
2. Calculer le profil BetaMol[z]*Tr2(AlphaMol(z))[z0] 
Et calculer son integrale entre zmin et zmax
'''
def get_backscatter_mol(p, T, w):
    '''
    Fonction permet de calculer le coef. de backscatter moléculaire 
    p(Pa), T(K), w(um)
    '''
    k = 1.38e-23
    betamol = p/(k*T) * 5.45e-32 * (w/0.55)**(-4.09)
    alphamol = betamol/0.119
    return alphamol, betamol


AlphaMol, BetaMol = get_backscatter_mol(pres, ta, 0.532)
print(BetaMol)


# In[78]:


output.to_pickle('/home/nmpnguyen/ear5withoutinterp.pkl')


# In[2]:


pd.read_pickle('/homedata/nmpnguyen/OPAR/Processed/LIO3T/2019-04-24_simul.pkl')


# In[64]:


simulpklpath = sorted(Path("/homedata/nmpnguyen/IPRAL/RF/Simul/").glob('ipral_1a_Lz1R15mF30sPbck_v01_20180909_000000_1440*.pkl'))[0]
print(simulpklpath)
data = pd.read_pickle(simulpklpath)

# !pip3 install pickle5
# import pickle5 as pickle
# with open(simulpklpath, "rb") as fh:
#   data = pickle.read(fh)
# data = pickle.load( open( simulpklpath, "rb" ) )


# In[69]:


# pression = data['pression'].unstack()
pression.iloc[585]


# In[67]:


temperature = data['ta'].unstack()
np.array(temperature.iloc[285]), np.array(pd.DataFrame(temperature.iloc[285]).index)


# In[72]:


fig, ax = plt.subplots(figsize=(4,6))
ax.plot(np.array(pression.iloc[585]), np.array(pd.DataFrame(pression.iloc[285]).index), 
        label='Pression, Pa', color='r')
ax.set_ylim(0, 20000)
ax.set(ylabel='Altitude, m', xlabel='Pression, Pa', title = 'Interpolated data from ERA5 hourly\n 2018-09-09 05:00:42')

ax2 = ax.twiny()
ax2.set(xlabel='Temperature, K')
ax2.plot(np.array(temperature.iloc[585]), np.array(pd.DataFrame(temperature.iloc[285]).index), 
         label='Temperature, K', color='b')

ax.legend(loc='right')
ax2.legend()

# plt.gca().invert_yaxis()

# plt.savefig('tempe_pression_illutra2022.png')

