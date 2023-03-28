
# coding: utf-8

# In[1]:


import xarray as xr
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


filename = 'cl_RF_PCp_20200506.txt'


# In[3]:


import csv
with open(filename, 'r') as f:
#     content = csv.reader(f, delimiter=',')
#     first_row = [column[0] for column in csv.reader(f,delimiter='\t')]
    all_data = [line.strip() for line in f.readlines()]
    height_line = all_data[:6]
    data = all_data[7:]
    


# In[4]:


len(data)


# In[5]:


new_row = [[float (element) for element in row.split(',')] for row in data]
import numpy as np
new_data = np.array(new_row)


# In[6]:


new_data.shape


# In[15]:


fig, ax = plt.subplots()
ax.plot(new_data[:,1], new_data[:,0], color='r', label='attn rayleigh bsc', zorder=10)
ax.axhspan(11.76, 14.78, color='y', alpha=0.2)
ax.plot(new_data[:,2]*0.8e-13, new_data[:,0], color='g', label='rcs')
ax.legend()


# In[19]:


sr = new_data[:,2]*0.85e-13/new_data[:,1]
fig, ac = plt.subplots()
ac.plot(sr, new_data[:,0])
ac.vlines(1, 0, 30, 'r', zorder=10)


# In[47]:


#lower and upper Rayleigh height mimits = 11.76, 14.78
idx = (new_data[:,0] >= 11.76) & (new_data[:,0] <= 14.78)
const = np.mean(new_data[idx,1])/np.mean(new_data[idx,2])
fig, (ax, ax2) = plt.subplots(figsize=[10,6], nrows=1, ncols=2)
ax.semilogx(new_data[:,1], new_data[:,0], color='r', label='attn RayleighBSC', zorder=10)
ax.axhspan(11.76, 14.78, color='y', alpha=0.2, label='calibration height')
ax.semilogx(new_data[:,2]*const, new_data[:,0], color='g', label='Normalized Signal')
ax.legend()
ax.set(xlabel='Backscatter Coef. [1/m.sr]')
sr = new_data[:,2]*const/new_data[:,1]
ax2.plot(sr, new_data[:,0])
ax2.vlines(1, 0, new_data[:,0].max(), 'r', zorder=10, label='cloud-free condition')
ax2.set(xlabel='Scattering Ratio')
ax2.axhspan(11.76, 14.78, color='y', alpha=0.2, label='calibration height')
ax2.legend()



#lower and upper Rayleigh height mimits = 5 et 10km
idx = (new_data[:,0] >= 3.76) & (new_data[:,0] <= 7.78)
const = np.mean(new_data[idx,1])/np.mean(new_data[idx,2])
fig, (ax, ax2) = plt.subplots(figsize=[10,6], nrows=1, ncols=2)
ax.semilogx(new_data[:,1], new_data[:,0], color='r', label='attn RayleighBSC', zorder=10)
ax.axhspan(3.76, 7.78, color='y', alpha=0.2, label='calibration height')
ax.semilogx(new_data[:,2]*const, new_data[:,0], color='g', label='Normalized Signal')
ax.legend()
ax.set(xlabel='Backscatter Coef. [1/m.sr]')
sr = new_data[:,2]*const/new_data[:,1]
ax2.plot(sr, new_data[:,0])
ax2.vlines(1, 0, new_data[:,0].max(), 'r', zorder=10, label='cloud-free condition')
ax2.set(xlabel='Scattering Ratio')
ax2.axhspan(3.76, 7.78, color='y', alpha=0.2, label='calibration height')
ax2.legend()




# In[64]:


#moyenner verticalement 30m
rcs_Av30 = np.mean(np.reshape(new_data[:3732], (int(new_data.shape[0]/4), 3, 4)), axis=2)


# In[79]:


z30 = np.mean(np.reshape(new_data[:3732,0], (-1,12)), axis=1)
attnRBSC30 = np.mean(np.reshape(new_data[:3732,1], (-1,12)), axis=1)
rcs30 = np.mean(np.reshape(new_data[:3732,2], (-1,12)), axis=1)


# In[103]:


#lower and upper Rayleigh height mimits = 11.76, 14.78
ztop = 5
zbottom = 3.9
idx = (z30 > zbottom) & (z30 < ztop)
const = np.mean(attnRBSC30[idx])/np.mean(rcs30[idx])
print(const)
fig, (ax, ax2) = plt.subplots(figsize=[10,6], nrows=1, ncols=2)
ax.semilogx(attnRBSC30, z30, color='r', label='attn RayleighBSC', zorder=10)
ax.axhspan(zbottom, ztop, color='y', alpha=0.2, label='calibration height')
ax.semilogx(rcs30*const, z30, color='g', label='Normalized Signal')
leg = ax.legend(loc='lower left')
leg.set_title('Vertical average 30m ')
ax.set(xlabel='Backscatter Coef. [1/m.sr]')
sr = rcs30*const/attnRBSC30
ax2.plot(sr, z30)
ax2.vlines(1, 0, z30.max(), 'r', zorder=10, label='cloud-free condition')
ax2.set(xlabel='Scattering Ratio')
ax2.axhspan(zbottom, ztop, color='y', alpha=0.2, label='calibration height')
leg = ax2.legend()
leg.set_title('Vertical average 30m ')



# In[7]:


new_data[:,0]


# ### RECUPERATION WMO RADIOSONDE 

# In[9]:


wmofilename = 'WMO-Radiosonde-Nimes-20200506-00UTC'
wmofile = open(wmofilename, 'r')


# In[15]:


import csv
with open(wmofilename, 'r') as f:
#     content = csv.reader(f, delimiter=',')
#     first_row = [column[0] for column in csv.reader(f,delimiter='\t')]
    all_data = [line.strip() for line in f.readlines()]
    height_line = all_data[0]
    
wmodata = np.array(all_data[1:])


# In[50]:


(wmodata[0].split('-9999')[:])

