
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


# In[2]:


import seaborn as sns


# In[54]:


mesures = pd.read_pickle('/home/nmpnguyen/conversion_model/comparaison/ipral_learned_TESTtarget_dataset.pkl')
predictes = pd.read_pickle('/home/nmpnguyen/conversion_model/comparaison/ipral_learned_TESTpredict_dataset.pkl')


# In[55]:


mesures.shape, predictes.shape


# In[60]:


# data = pd.concat([mesures, predictes], axis=1)
# data = data.dropna(axis=0, how='any')
data.columns = ['mesures', 'predictes']
data


# In[61]:


H = np.histogram2d(data['mesures'], data['predictes'], 
                   bins=[np.arange(0, 80, 0.1), np.arange(0, 80, 0.1)])
H


# In[62]:



fig, ax = plt.subplots(figsize=(7.5,7.5))
p = ax.pcolormesh(H[1], H[2], (H[0]/np.sum(H[0])).T, norm=LogNorm())
plt.colorbar(p, ax=ax, extend='both', label='Percent %')
ax.plot(np.arange(0,80), np.arange(0,80), '-k')
ax.plot(np.arange(0,80)+0.1, np.arange(0,80), '-y', label='+/- 0.1')
ax.plot(np.arange(0,80)-0.1, np.arange(0,80), '-y')
ax.plot(np.arange(0,80)+0.1, np.arange(0,80), '-g', label='+/- 0.15')
ax.plot(np.arange(0,80)-0.1, np.arange(0,80), '-g')
ax.legend()
# grid
ax.set_axisbelow(True)
ax.yaxis.grid(color='gray', linestyle='dashed', alpha=0.3)
ax.xaxis.grid(color='gray', linestyle='dashed', alpha=0.3)
# TITLE
plt.suptitle('Theorical method', x=0.125, y=0.98, ha='left', fontsize=16)
# SUBTITLE
plt.title('+/- 0.1 : 68.97% \n+/- 0.15 : 79.24%', loc='left', fontsize=11)
# AXIS LABELS
plt.ylabel('SR532 predicted')
plt.xlabel('SR532 measured')
# CAPTION
plt.text(-0.5, -10.5, 'Dataset: Ipral-2020-all dataset', ha='left', 
         fontsize = 11, alpha=0.9)

plt.tight_layout()


# In[19]:


plt.hist2d(data['mesures'], data['predictes'], range=[[0,80], [0,80]], bins=80, norm=LogNorm())


# In[66]:


from datetime import datetime

now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
now

