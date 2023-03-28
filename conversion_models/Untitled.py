
# coding: utf-8

# In[2]:


import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


# In[3]:


xr.open_dataset('/homedata/nmpnguyen/ORACLES-ER2/HSRL2_ER2_20160819_R8.h5', group= 'DataProducts')


# In[13]:


a = np.arange(2006,2036)
b = [0]*len(a)
print(a, b)


# In[17]:


f, ax=plt.subplots(figsize=(12,3))
ax.plot(a, [0]*len(a), '|', color='k')
ax.axhline(0, color='k')
ax.plot(2037, 0, '>', color='k')
ax.set_ylim(-1,1)


# In[6]:


df = pd.read_pickle('/homedata/nmpnguyen/OPAR/Processed/LI1200/2018-06-19_simul.pkl')


# In[8]:


dt_to_xrDataset = xr.Dataset.from_dataframe(df)
print(dt_to_xrDataset)


# In[95]:


xr.open_dataset('./OPAR/LI1200.daily/2020-08-18.nc4').to_dict(data=False)['data_vars']['signal']['attrs']


# In[171]:


def option_database(type_database, filepath):
    if type_database == 'opar':
        output_dataset_dict = transfert_opar_database(filepath)
#     elif type_database == 'ipral':
#     elif type_database == 'er2_hsrl2':
#     elif type_database == 'lng_hsrl':
    else:
        print('Please entry type_database')

def transfert_opar_database(filepath):
    data = xr.open_dataset(filepath)
    time_arr = data['time'].values
    range_arr = data['range'].values
    latitude_sol = data['signal'].attrs['latitude']
    longitude_sol = data['signal'].attrs['longitude']
    alt_station = data['signal'].attrs['alt_station(km)']
    start_time = data['signal'].attrs['start_time']
    end_time = data['signal'].attrs['end_time']
#     attrs_dict = data.to_dict(data=False)['data_vars']['signal']['attrs']    
    data_dict = {
        "coords": {
            "time": {"dims": "time", "data": time_arr},
            "range": {"dims": "range", "data": range_arr*1e3, "attrs": {"units": "m"}},
            },
        "attrs": {
            "lat": float(latitude_sol),
            "lon": float(longitude_sol),
            "start_time": pd.to_datetime(start_time),
            "end_time": pd.to_datetime(end_time),
             },
        "dims": {"time": len(time_arr), "range": len(range_arr)},
        "data_vars": {
            "altitude": {"dims": "range", "data": (range_arr+alt_station)*1e3, "attrs": {"units": "m"}},
            },
    }
    return data_dict 

def transfert_ipral_database(filepath):
    data = xr.open_dataset(filepath)
    if (pd.to_datetime(data['time'].values[0]).day == pd.to_datetime(Path(filepath).stem.split('_')[4]).day):
        start_time = data['time'].values[0].astype('str')
    else:
        start_time = pd.to_datetime(Path(filepath).stem.split('_')[4])
    end_time = data['time'].values[-1].astype('str')
    print(type(end_time))
    time_arr = data['time'].values
    range_arr = data['range'].values  
    altitude = data['altitude'].values 
    latitude_sol = data.attrs['geospatial_lat_min']
    longitude_sol = data.attrs['geospatial_lon_min']
    data_dict = {
        "coords": {
            "time": {"dims": "time", "data": time_arr},
            "range": {"dims": "range", "data": range_arr},
            },
        "attrs": {
            "lat": float(latitude_sol),
            "lon": float(longitude_sol),
            "start_time": start_time,
            "end_time": end_time,
             },
        "dims": {"time": len(time_arr), "range": len(range_arr)},
        "data_vars": {
            "altitude": {"dims": "range", "data": altitude+range_arr},
            },
    }
    return data_dict


# In[151]:


# transfert_opar_database('./OPAR/LI1200.daily/2020-08-18.nc4')
tmp_file = xr.Dataset.from_dict(transfert_opar_database('./OPAR/LI1200.daily/2020-08-18.nc4'))


# In[172]:


tmp_file = xr.Dataset.from_dict(transfert_ipral_database('/bdd/SIRTA/pub/basesirta/1a/ipral/2020/01/06/ipral_1a_Lz1R15mF30sPbck_v01_20200106_000000_1440.nc'))
tmp_file.to_netcdf('tmp.nc', 'w')
# xr.open_dataset('/bdd/SIRTA/pub/basesirta/1a/ipral/2020/01/06/ipral_1a_Lz1R15mF30sPbck_v01_20200106_000000_1440.nc').attrs['geospatial_lat_min']


# In[177]:


str(pd.to_datetime(tmp_file.attrs['start_time']).year)


# In[160]:


np.unique(pd.to_datetime(tmp_file.time).strftime('%Y-%m-%dT%H:00:00.000000').astype('datetime64[ns]'))

