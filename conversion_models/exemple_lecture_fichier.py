
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path #pour manipuler facilement les chemins des fichiers
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


# In[119]:


dt= xr.open_dataset('/home/nmpnguyen/giems2_wgs84_v2.nc', decode_cf=False)


# In[121]:


dt['Wetland area']


# In[64]:


dt = xr.open_dataset('/homedata/nmpnguyen/IPRAL/SCC_produits/output/output/20200514/elic_355_concat_data.nc')
dt


# In[71]:


fig, ax = plt.subplots()
(dt['molecular_extinction']*0.119*2*dt['molecular_transmissivity_at_emission_wavelength']).isel(channel=0, time=0).plot(y='level', label='emission', ax=ax)
# dt['molecular_transmissivity_at_detection_wavelength'].isel(channel=0, time=0).plot(y='level', label='detection', ax=ax)


# In[56]:


elic = np.array(['temperature', 'pressure', 'attenuated_backscatter', 'molecular_extinction', 'molecular_transmissivity_at_emission_wavelength', 'molecular_transmissivity_at_detection_wavelength'])
elic[np.isin(elic, list(dt.keys()))]

dt[elic[np.isin(elic, list(dt.keys()))]]


# -----------------------

# In[2]:


listfiles = sorted(Path('/homedata/nmpnguyen/IPRAL/NETCDF/v_simple/2020').glob('ipral*.nc'))

dt = xr.open_dataset(listfiles[60])
mask = np.logical_and(dt['flags'].sel(wavelength=355).values == 0, dt['flags'].sel(wavelength=532).values == 0)

sr355 = (dt['calibrated']/dt['simulated']).isel(wavelength = 0)
sr355.plot(x='time', y='range', norm=LogNorm(vmin=1e-2), ylim=(0, 15000))


# In[37]:


listfiles = sorted(Path('/homedata/nmpnguyen/IPRAL/NETCDF/v_simple/2020/').glob('ipral*.nc'))
listdays = ['20200207' , '20200401' , '20200912' , '20200914' , '20200916' , '20200921' , '20200923' , '20201128' ,
'20200326' , '20200911' , '20200913' , '20200915' , '20200918' , '20200922' , '20201105']

allsr355 = None
allsr532 = None

# for file in listfiles:
#     print(file)
#     dt = xr.open_dataset(file)

for f in listdays:
    file = Path('/homedata/nmpnguyen/IPRAL/NETCDF/v_simple/2020/', f'ipral_1a_Lz1R15mF30sPbck_v01_{f}_000000_1440.nc')
    dt = xr.open_dataset(file)
    mask = np.logical_and(dt['flags'].sel(wavelength=355).values == 0, dt['flags'].sel(wavelength=532).values == 0)
    zlim = (dt['range'] < 15000)
    if mask.any():
        sr355 = (dt['calibrated']).isel(wavelength = 0, time = mask, range = zlim).resample(time='5min').mean(dim = 'time')
        sr532 = (dt['calibrated']).isel(wavelength = 1, time = mask, range = zlim).resample(time='5min').mean(dim = 'time')
        if allsr355 is None : 
            allsr355 = sr355
            allsr532 = sr532
        else:
            allsr355 = xr.concat([allsr355, sr355], dim='time')
            allsr532 = xr.concat([allsr532, sr532], dim='time')
    else:
        print('Any data to compute !')
        print(mask)
        


# In[38]:


allsr355


# In[39]:


fig, ax = plt.subplots(figsize=(9,9))
plt.rcParams['font.size']=14
ranges = [[0,5e-5], [0,5e-5]] #[[-1, 50], [-1, 50]]#

h= ax.hist2d(allsr355.values.ravel(), allsr532.values.ravel(), range=ranges, 
         bins=100, norm=LogNorm())
plt.colorbar(h[3], ax=ax, label='Counts')
ax.set(xlabel='ATB355, 1/m.sr', ylabel='ATB532, 1/m.sr', title=f'Attenuated Backscatter, Ipral, \n5min x 15m, {allsr355.shape[0]} profiles, {len(listdays)} days')
# ax.set(xlabel='SR355', ylabel='SR532', title=f'Scattering Ratio, Ipral, \n5min x 15m, {allsr355.shape[0]} profiles, {len(listdays)} days')

plt.grid()
plt.savefig('/homedata/nmpnguyen/IPRAL/NETCDF/v_simple/attn_backscatter_ipral.png')
# plt.savefig('/homedata/nmpnguyen/IPRAL/NETCDF/v_simple/scattering_ratio_ipral.png')


# In[18]:


allsr355_mean = allsr355.resample(time='15min').mean(dim='time')
allsr532_mean = allsr532.resample(time='15min').mean(dim='time')
# allsr355.sortby('time')
allsr355_mean.to_netcdf('/homedata/nmpnguyen/IPRAL/NETCDF/v_simple/allatb355_ipral_2020.nc')
allsr532_mean.to_netcdf('/homedata/nmpnguyen/IPRAL/NETCDF/v_simple/allatb532_ipral_2020.nc')


# In[59]:


dt = xr.open_dataset('/homedata/nmpnguyen/IPRAL/NETCDF/v_simple/2020/ipral_1a_Lz1R15mF30sPbck_v01_20201107_000000_1440.nc')


# In[5]:


x = (dt['calibrated']/dt['simulated']).sel(wavelength = 355)
dtalt = dt['range'].values


# In[11]:


from scipy.integrate import cumtrapz

y = cumtrapz(x[100,:], dtalt, initial=0)
print(y.shape)


# In[34]:


# 3e feature of X_mat
from tqdm import tqdm
X3 = np.zeros(dtalt.shape)
X4 = np.zeros(dtalt.shape)
print(X3.shape)
# X3[0,:] = np.nan
for j in tqdm(range(1, dtalt.shape[0])):
#     X4[j] = x[100,j-1] + x[100,j]*(dtalt[j] - dtalt[j-1])
    X3[j] = X3[j-1] + x[100,j]*(dtalt[j] - dtalt[j-1])


# In[43]:


get_ipython().magic('matplotlib notebook')
plt.plot(x[100,:1500], dtalt[:1500], color='r', label='SR355')
plt.plot(y[:1500], dtalt[:1500], color='b')
plt.plot(X3[:1500], dtalt[:1500], color='g', label='SR355 intégré sur z')
plt.legend(loc='best', frameon=False)
# plt.plot(X4[:1500], dtalt[:1500], color='b')
# plt.xlim(-5,600)


# In[60]:


flags = (dt['flags'].sel(wavelength=355) == 0)

dt['calibrated'].sel(wavelength=355).where(dt['flags'].sel(wavelength=355) == 0, drop=False).plot(x='time', y='range', ylim=(0,20000), norm=LogNorm(vmin=1e-8))#.where(dt['flags'].sel(wavalength=355)==0, drop=True)


# In[62]:


get_ipython().magic('matplotlib inline')
dtsr = (dt['calibrated']/dt['simulated']).sel(wavelength=355).where(dt['flags'].sel(wavelength=355) == 0, drop=False)
dtsr.plot(x='time', y='range', ylim=(0,20000), norm=LogNorm(vmin=1e-8))


# In[16]:


fig, ax = plt.subplots()
p = ax.pcolormesh(result['latitude'].values, result['longitude'].values, result.values.T)
plt.colorbar(p, ax=ax, label='temperature')


# In[17]:


fig, ax = plt.subplots()
result.plot(x='latitude', y='longitude')


# In[12]:


for subdir in sorted(Path('/homedata/nmpnguyen/IPRAL/SCC_produits/output/output/').iterdir()):
    subsubdir = sorted(subdir.iterdir())[0]
    print(sorted(subsubdir.glob('elic')))
    
def get_products_selected(main_path, day, product_name):
    product_path = Path(list_path).glob(f'**/{product_name}')
    def get_path_subdirectories(main_path, day):
        return list_path
    return product_path


# -------------------------------------------

# ### Lecture du fichier des données brutes (LEVEL 1)

# In[5]:


# DATA LEVEL 1
# il faut remplacer le chemin avec le votre
RAW_PATH = Path('/bdd/SIRTA/pub/basesirta/1a/ipral/2018/09/28/ipral_1a_Lz1R15mF30sPbck_v01_20180928_000000_1440.nc')

data1 = xr.open_dataset(RAW_PATH)
print(data1.variables)


# In[11]:


# plot 1 profil 
fig, ax = plt.subplots()
data1.isel(time=0)['rcs_12'].plot(y='range', ylim=(0, 15000), xscale='log', label='P(R) x R^2, Near Range at 355nm', color='b')
data1.isel(time=0)['rcs_16'].plot(y='range', ylim=(0, 15000), xscale='log', label='P(R) x R^2, Near Range at 532nm', color='g')
ax.legend()


# In[12]:


# plot tous les profils = 1 quicklook, 532nm, colorbar en log 
from matplotlib.colors import LogNorm

fig, ax = plt.subplots()
data1['rcs_12'].plot(y='range', x='time', ylim=(0, 15000), cmap='turbo', norm=LogNorm())


# ### Lecture du fichier des données calibrées (LEVEL 2)

# In[13]:


CALIB_PATH = Path('/homedata/nmpnguyen/IPRAL/NETCDF/ipral_calib_v01_20180928_000000_1440.nc')

data2 = xr.open_dataset(CALIB_PATH)
data2


# In[14]:


fig, ax = plt.subplots()
data2.isel(time=0)['Total_Calib_Attn_Backscatter_532'].plot(y='range', ylim=(0, 15000), xscale='log', xlim=(1e-8, 1e-5), label='Total_Calib_Attn_Backscatter_532')
data2.isel(time=0)['Attn_Molecular_Backscatter_532'].plot(y='range', ylim=(0, 15000), xscale='log', label='Attn_Molecular_Backscatter_532')
ax.legend()


# In[16]:


data1.to_netcdf('ipral_1a_raw_v01_20180928_000000_1440.nc', 'w')


# In[14]:


x = xr.open_dataset('/scratchx/nmpnguyen/IPRAL/raw/SR_histogram/IPRAL_2018_validated_profiles4_allatb355.nc')
y = xr.open_dataset('/scratchx/nmpnguyen/IPRAL/raw/SR_histogram/IPRAL_2018_validated_profiles4_allatb532.nc')


# In[18]:


x['calibrated'][0,:]


# In[15]:


fig, ax = plt.subplots()
h = ax.hist2d(x.values.ravel(), y.values.ravel(), bins=100, range=[[-1, 40],[-1, 80]], norm=LogNorm())
plt.colorbar(h[3], ax=ax)
ax.set(xlabel='sr355_mesured', ylabel='sr532_mesured')


# In[41]:



xarray 

xr.open_dataset('/bdd/CALIPSO/Lidar_L1/CAL_LID_L1.v4.11/2021/2021_08_07/CAL_LID_L1-Standard-V4-11.2021-08-07T12-53-14ZD.hdf')


# In[42]:


xr.__version__

