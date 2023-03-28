
# coding: utf-8

# In[7]:


import xarray as xr
import numpy as np
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt


# In[80]:


mainpath = Path('/bdd/CFMIP/CFMIP_OBS_LOCAL/GOCCP_v3/instant_SR_CR_DR/v3.1.2/2019/day/')
path = list(mainpath.glob("instant_SR_CR_DR_2019-12-11*_CFMIP1_*.nc"))
len(path)

d = xr.open_dataset(path[1])


# In[81]:


d


# In[82]:


fig, axs = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True, figsize=[16,12])
for i, ax in enumerate(axs.flatten()):
    d.ATB[i,:].plot.line(y='altitude',color='r', label='atb', ax=ax)
    d.ATB_mol[i,:].plot.line(y='altitude', color='b', label='atb_mol', ax=ax)
    ax.legend()
    ax.set(title=str(d.time[i].values))
    ax.set_xlim(0, 0.02)


# In[18]:


fig, axs = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True, figsize=[16,12])
for i, ax in enumerate(axs.flatten()):
    d.instant_SR[i,:].plot.line(y='altitude',color='r', label='instant_sr', ax=ax)
    #(d.ATB[i,:]/d.ATB_mol[i,:]).plot.line(y='altitude',color='b', label='sr', ax=ax)
    ax.vlines(1, ymin=d.altitude[0], ymax=d.altitude.max(), linestyle='--', color='k')
    ax.legend()
    ax.set_xlim(-2, 5)
    ax.set(title=str(d.time[i].values))


# In[11]:


d.altitude


# In[58]:


#d.altitude.values*1e3, d.TEMP.values[1,:]+273.15
C = pressure_from_temperature(d.TEMP.values[1,:]+273.15, d.altitude.values*1e3)    


# In[54]:


pression


# In[59]:


C


# In[92]:


from math import exp, log
def isa(alt):
    if alt <= 11000:
        pressure = 101325 * (1 - 2.2556e-5*alt)**5.25616
    else:
        pressure = 0.223356 * 101325 * exp(-0.000157688*(alt-11000))
    return pressure

def MolATB_from_PT(P,T,alt,w):
    const = (5.45e-32/1.38e-23)*(w*1e-3/0.55)**(-4.09)
    beta = const*np.float(P/T)
    alpha = 2*beta/0.119
    tau = np.zeros_like(beta)
    for i in range(1, alt.shape[0]):
        tau[i] = tau[i-1] + alpha[i] * (alt[i] - alt[i-1])
        
   
    #attenuation = beta*exp(-2*tau)
    return beta, tau#attenuation


def equiq_hydro(lat, alt):
    lat = np.deg2rad(lat)
    acc_gravity = 9.78032*(1+5.2885e-3*(np.sin(lat))**2 - 5.9e-6*(np.sin(2*lat))**2)
    r0 = 2*acc_gravity/(3.085462e-6 + 2.27e-9*np.cos(2*lat) - 2e-12*np.cos(4*lat))
    g0 = 9.80665
    #geopt_for_ipral['geopt_height'] = geopt_for_ipral["geopt"]/g0
    #geopt_for_ipral['altitude'] = (geopt_for_ipral['geopt_height']*r0)/(acc_gravity*r0/g0 - geopt_for_ipral['geopt_height'])
    M = 28.966E-3 
    R = 8.314510
    T = (15 + 273.15)
    const = -(M*g0)/(R*T)
    p0 = 101325
    pression = p0*np.exp(const*alt)
    return pression


# In[177]:


pressure = [isa(alt*1e3) for alt in d.altitude.values]
MolATBcompute = MolATB_from_PT(pressure, d.TEMP[100,:].values+273.16, d.altitude.values*1e3, 532)


# In[263]:


pression = pression(d.latitude.values, d.altitude.values*1e3)


# In[265]:


fig, ax=plt.subplots(figsize=[12,10])
d.TEMP[0,:].plot.line(y='altitude', color='r', label='T_goccp, C', ax=ax)
#ax.legend()
ax2= ax.twiny()
ax2.plot(pressure, d.altitude.values, label='P1_calcul, Pa', color='g')
ax2.plot(pression, d.altitude.values, label='P2_calcul, Pa', color='b')
#ax2.legend(loc=0)
lines_1, labels_1 = ax.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
lines = lines_1 + lines_2
labels = labels_1 + labels_2
ax.legend(lines, labels, loc=0)


# In[178]:


fig, ax=plt.subplots(figsize=[12,10])
(d.ATB_mol[100,:]*1e-3).plot.line(y='altitude',label='MolATB, goccp', color='r', ax=ax)
(d.ATB[100,:]*1e-3).plot.line(y='altitude',label='ATB, goccp', color='b', ax=ax)
ax.plot(MolATBcompute, d.altitude.values, label='calcul', color='g')
ax.legend()
ax.set_xlim(-2e-6, 2e-5)
ax.set(title='Compare Molecular Attn Backscatter between GOCCP and Theoretical compute')


# In[103]:


alt_ref = (d.altitude>=33)&(d.altitude<=35)


# In[104]:


alt_ref


# In[62]:


import pandas as pd
from pathlib import Path
oparmol=pd.read_pickle(Path('/homedata/nmpnguyen/OPAR/Processed/LIO3T/2019-01-21_simul.pkl'))
oparatb=pd.read_pickle(Path("/homedata/nmpnguyen/OPAR/Processed/LIO3T/2019-01-21_00532.p00532.s_ATB.pkl"))
oparraw = xr.open_dataset(Path('/home/nmpnguyen/OPAR/LIO3T.daily/2019-01-21.nc4'))


# In[63]:


oparatb


# In[65]:


beta532mol = oparmol['beta532mol'].unstack(level=1)
beta532mol = beta532mol.iloc[:,:oparatb.shape[1]]
oparalt = beta532mol.columns


fig, ax = plt.subplots(figsize=[12,10])
ax.plot(beta532mol.iloc[0,:], oparalt, label='355mol')
ax.plot(oparatb.iloc[0,:], oparalt, label='atb')
ax.legend()
#zoom en-dessous de 5km
#ax.set_ylim(5000,7000)
ax.set_xlim(0, 0.5e-5)


# In[85]:


oparalt_ref = (oparalt>=5000) & (oparalt<=6000)


# In[74]:


np.nanmean((oparatb.iloc[0,oparalt_ref].values/beta532mol.iloc[0,oparalt_ref].values).astype("float"))


# In[75]:


y = beta355mol.iloc[0,oparalt_ref].values
x1 = oparatb.iloc[0,oparalt_ref].values
x2 = oparalt[oparalt_ref]
#--------------------------> Equation de régression: f(z) = K.exp(-z)
# np.polyfit -> log(y) = log(K)+log(exp(-Z))
curve_fit = np.polyfit(x1, np.log(y), 1)
print(curve_fit)
yy = np.exp(curve_fit[1])*np.exp(curve_fit[0]*x1)
plt.plot(x1, oparalt[oparalt_ref], label='x1', color='r')
plt.plot(y, oparalt[oparalt_ref], label='y', color='g')
plt.plot(yy, oparalt[oparalt_ref], label='yy', color='b')
plt.legend()


# In[15]:


#------------------------------> Appliquer CURVE_FIT et tracer un plot de comparaison 
oparalt_ref = (oparalt>=5000) & (oparalt<=7000)
yy = np.exp(curve_fit[1])*np.exp(curve_fit[0]*oparatb.iloc[0,:])
fig, ax = plt.subplots(figsize=[12,10])
ax.plot(beta355mol.iloc[0,oparalt_ref], oparalt[oparalt_ref], label='355mol', color='g')
ax.plot(oparatb.iloc[0,oparalt_ref], oparalt[oparalt_ref], label='atb', color='r')
ax.plot(yy[oparalt_ref], oparalt[oparalt_ref], label='fit exp', color='b')
ax.legend()
ax.set_xlim(0, 0.5e-5)


# In[253]:


plt.plot(yy/beta355mol.iloc[0,:].values, oparalt)
plt.vlines(1, ymin=0, ymax=oparalt[-1], linestyle='--')


# In[76]:


#------------------------> Calculer la pression par 2 méthodes: 
#------Equilibre hydrostatique-------- 
pression = oparmol['pression'].unstack(level=1)
Tempe = oparmol['ta'].unstack(level=1)
#molAttn1 = MolATB_from_PT(pression.iloc[0,:].values, Tempe.iloc[0,:].values, alt=pression.columns+2160,w=532)
molAttn1 = oparmol['beta532mol'].unstack(level=1)


# In[105]:


#------International Standard Atmosphere-------
pressure = [isa(z+2160) for z in Tempe.columns]
alt=np.array(pression.columns+2160)
const = (5.45e-32/1.38e-23)*(532*1e-3/0.55)**(-4.09)
beta = const*(pressure/np.array(Tempe.iloc[0,:].values)).astype('float')
alpha = 2*beta/0.119
tau = np.zeros_like(beta)
for i in range(1, alt.shape[0]):
    tau[i] = tau[i-1] + alpha[i] * (alt[i] - alt[i-1])

print(-2*tau)
molAttn2 = beta*np.exp(-2*tau)
#molAttn2 = MolATB_from_PT(pressure, np.array(Tempe.iloc[0,:].values), alt=np.array(pression.columns+2160), w=532)


# In[106]:


oparalt = oparraw['range']*1e3
#------Comparaison-------
fig, (ax,ax2)=plt.subplots(nrows=1, ncols=2,figsize=[12,10])
ax2.plot(pressure, pression.columns, label='by isa, Pa', color='g')
ax2.plot(pression.iloc[0,:], pression.columns, label='by era, Pa', color='b')
ax2.legend(loc=0)
ax2.set(ylabel='range, m')
ax.plot(molAttn1.iloc[0,:], np.array(molAttn1.columns)+2160, label='by era', color='b')
ax.plot(molAttn2, alt, label='by isa', color='g')
#ax.plot(oparatb.iloc[0,:], oparalt[:oparatb.shape[1]]+2160, label='atb', color='r')
ax.legend()
ax.set(ylabel='range, m')
ax.set_ylim(0,25000)
ax.set_xlim(0, 0.5e-5)


# In[133]:


signal = oparraw['signal'][0,:,6] + oparraw['signal'][0,:,7]
oparalt = oparraw['range'].values*1e3
oparalt_ref = np.where((oparalt>=5000) & (oparalt<=5200))


# In[146]:


m1= np.mean(signal[oparalt_ref[0]]/molAttn1.iloc[0,oparalt_ref[0]])
m2= np.mean(signal[oparalt_ref[0]]/molAttn2[oparalt_ref[0]])
print(m1, m2)


# new1 = signal/m1
# new2 = signal/m2
# print(new1[:molAttn1.shape[1]])
# 

# In[163]:


fig, ax = plt.subplots(figsize=[12,10])
ax.plot(new1, oparalt, label='atb from era', color='b')
#ax.plot(new2, oparalt, label='atb from isa', color='g')
ax.plot(molAttn1.iloc[0,:], np.array(molAttn1.columns)+2160, linestyle='--', label='mol from era', color='b')
ax.plot(molAttn2, alt, linestyle='--', label='atb from isa', color='g')
ax.legend()
ax.set_ylim(0,20000)

