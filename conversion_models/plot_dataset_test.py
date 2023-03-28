# Basics librairies
#------------------
import numpy as np
import pandas as pd 
import xarray as xr 
from pathlib import Path 
from datetime import datetime

# Librairies to plots
#------------------
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.dates as mdates


def find_nearest_time(time_array, value):
    time_array = pd.to_datetime(time_array)
    idt = (np.abs(time_array - pd.to_datetime(value))).argmin()
    time_value = time_array[idt]
    return idt, time_value


from argparse import Namespace, ArgumentParser
parser = ArgumentParser()
parser.add_argument('--filepath', '-file', required=True, type=str, help='Path of predicted file')
opts = parser.parse_args()

file_path = Path(f'{Path(opts.filepath).stem}_conversion_test_355_to_532.nc')
print(file_path)

original_path = Path(opts.filepath)
mol355 = xr.open_dataset(original_path)['simulated'].sel(wavelength=355)
mol532 = xr.open_dataset(original_path)['simulated'].sel(wavelength=532)

dt = xr.open_dataset(file_path)
dttime = dt['time'].values
dtrange = dt['range'].values
time_moment = find_nearest_time(dttime, '2020-09-04 06:00:00')[0]

fig, (ax3, ax, ax2) = plt.subplots(ncols=3, figsize=(15,5))
cmap = plt.cm.turbo
cmap.set_under('lightgrey')
label = 'sr'
p = ax.pcolormesh(dttime, dtrange, (dt['Y']).T, cmap=cmap, vmin=0, vmax=20)
plt.colorbar(p, ax=ax, label=label, extend='both')
ax.set(xlabel='time', ylabel='range', title='signal 532 original')
ax.set_ylim(0,14000)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
p = ax2.pcolormesh(dttime, dtrange, (dt['Ypredicted']).T, cmap=cmap,  vmin=0, vmax=20)
plt.colorbar(p, ax=ax2, label=label, extend='both')
ax2.set(xlabel='time', ylabel='range', title='signal 532 predict')
ax2.set_ylim(0,14000)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
# cmap3 = plt.cm.Spectral_r
# cmap3.set_under('lightgrey')
p = ax3.pcolormesh(dttime, dtrange, (dt['X']).T, cmap=cmap,  vmin=0, vmax=20)
plt.colorbar(p, ax=ax3, label=label, extend='both')
ax3.set(xlabel='time', ylabel='range', title='signal 355 original')
ax3.set_ylim(0,14000)
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.savefig(f'/home/nmpnguyen/conversion_model/QL_{file_path.stem}.png')
plt.clf()
plt.close()

fig, (ax, ax2) = plt.subplots(ncols=2, figsize=(15,5))
dt['X'].isel(time=time_moment).plot(y='range', label='355 original', ax=ax)
dt['Y'].isel(time=time_moment).plot(y='range', label='532 original', ax=ax)
dt['Ypredicted'].isel(time=time_moment).plot(y='range', label='532 predicted', ax=ax)
ax.legend()
ax.set_ylim(0,14000)
ax.set_xlim(-5,20)

(dt['Y']-dt['Ypredicted']).isel(time=time_moment).plot(y='range', label='original-predict', ax=ax2)
ax2.legend()
ax2.set_ylim(0,14000)
ax2.set_xlim(-5,20)
plt.savefig(f'/home/nmpnguyen/conversion_model/profil_{file_path.stem}.png')
plt.clf()
plt.close()

fig, (ax, ax1, ax2) = plt.subplots(ncols=3, figsize=(15,5))
h = ax.hist2d(dt['X'].values.ravel(), dt['Y'].values.ravel(), range=[[-10,30], [-10,80]], bins=100, norm=LogNorm())
plt.colorbar(h[3], ax=ax, label='Counts')
ax.set(xlabel = 'SR 355 original', ylabel='SR 532 original')

h = ax1.hist2d(dt['X'].values.ravel(), dt['Ypredicted'].values.ravel(), range=[[-10,30], [-10,80]], bins=100, norm=LogNorm())
plt.colorbar(h[3], ax=ax1, label='Counts')
ax1.set(xlabel = 'SR 355 original', ylabel='SR 532 predicted', title=f'dt.attrs.mean_absolute_error')

h = ax2.hist2d(dt['Y'].values.ravel(), (dt['Ypredicted']).values.ravel(), range=[[-10,80], [-10,80]], bins=100, norm=LogNorm())
plt.colorbar(h[3], ax=ax2, label='Counts')
ax2.set(xlabel = 'SR 532 original', ylabel='SR 532 predicted', title=f'dt.attrs.mean_absolute_error')

plt.savefig(f'/home/nmpnguyen/conversion_model/Figs/sr_355_532_predicted_{file_path.stem}.png')
plt.clf()
plt.close()

