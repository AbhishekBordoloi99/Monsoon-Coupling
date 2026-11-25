import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib import gridspec
import xarray as xr
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmaps
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path
from scipy.signal import detrend
from scipy.stats import pearsonr, norm
from itertools import combinations
from scipy.ndimage import label
from scipy.stats import linregress
from scipy.stats import ks_2samp
from scipy.signal import butter, filtfilt
import pandas as pd
import itertools
from eofs.standard import Eof
import cmaps
from statsmodels.stats.multitest import multipletests
import seaborn as sns
import random
from collections import Counter
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
import scipy.stats as stats
from sklearn.mixture import GaussianMixture

def ght_distribution():
    path='/era5_global_geopotential_700hPa_1979_2020_point1deg.nc'
    ds = xr.open_dataset(path)
    lon1= 70
    lon2= 90
    lat1=28
    lat2=18
    z700 = ds['z'].sel(latitude=slice(lat1, lat2))
    z700 = xr.apply_ufunc(detrend,z700,input_core_dims=[['time']],output_core_dims=[['time']],kwargs={'axis': -1},vectorize=True,dask='parallelized',output_dtypes=[z700.dtype])
    z700 = (z700.groupby('time.dayofyear') - z700.groupby('time.dayofyear').mean('time'))
    print(z700)
    year=np.linspace(1979,2020,42,dtype=int)
    META_Z=[]
    for i in year:
        for j in range(150,250):
            META_Z.append(z700.sel(time=str(i)).sel(longitude=slice(lon1,lon2)).groupby('time.dayofyear')[j].mean(('latitude','longitude','time')).values)
    META_Z=np.array(META_Z)
    negZ=META_Z[META_Z<0]
    cut1=np.percentile(negZ,80)
    cut2=np.percentile(negZ,20)

    sigma = np.std(META_Z)

    bins_z = np.linspace(META_Z.min(), META_Z.max(), 101)
    counts, edges = np.histogram(META_Z, bins=bins_z)

    colors_z = []
    for left, right in zip(edges[:-1], edges[1:]):
        if right <= cut2:
            colors_z.append('#D55E00')  
        elif right <= cut1:
            colors_z.append('#999999') 
        elif right <= 0:
            colors_z.append('#0072B2')
        else:
            colors_z.append((0.6, 0.6, 0.6, 0.4)) 

    def get_precip(i): 
        return xr.open_dataset(f'/precip/tp.{i}.nc')['tp'].sel(lat=slice(lat2,lat1), lon=slice(lon1,lon2)).mean(('lat','lon'))
    precip_all = [get_precip(i) for i in year]
    precip = xr.concat(precip_all, dim='time').sortby('time')

    def detrend_add_mean(da): 
        fit = da.polyfit(dim='time', deg=1)
        trend = xr.polyval(da.time, fit['polyfit_coefficients'])
        return (da - trend + da.mean('time')).assign_attrs(da.attrs)

    precip_detrended = detrend_add_mean(precip)
    print(precip_detrended)

    prc1w, prc2w, prc1s, prc2s = [], [], [], []
    for i in year:
        path_tp='/precip/tp.'+str(i)+'.nc'
        precip_i = precip_detrended.sel(time=str(i))
        tmp, tmp_tp = [], []
        for j in range(150,250):
            tmp.append(z700.sel(time=str(i)).sel(longitude=slice(lon1,lon2)).groupby('time.dayofyear')[j].mean(('latitude','longitude','time')).values)
            tmp_tp.append(precip_i.groupby('time.dayofyear')[j].mean('time').values * 24000)
        tmp, tmp_tp = np.array(tmp), np.array(tmp_tp)
        for j in range(len(tmp)):
            if(np.mean(tmp[j:j+5])<0 and np.mean(tmp[j:j+5])>cut1 and tmp[j-1]>0):
                if(i<=1999):
                    prc1w.append(np.mean(tmp_tp[j:j+5]))
                else:
                    prc2w.append(np.mean(tmp_tp[j:j+5]))
            elif(np.mean(tmp[j:j+5])<cut2):
                if(i<=1999):
                    prc1s.append(np.mean(tmp_tp[j:j+5]))
                else:
                    prc2s.append(np.mean(tmp_tp[j:j+5]))
    prc1w, prc2w, prc1s, prc2s = map(np.array, [prc1w, prc2w, prc1s, prc2s])

    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial'],
        'font.size': 7,
        'axes.labelsize': 7,
        'axes.titlesize': 7,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'figure.dpi': 900,
        'savefig.dpi': 900,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'axes.linewidth': 1.2,
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        'hatch.linewidth': 0.6
    })
    fig = plt.figure(figsize=(7, 3.2))

    ax_z = fig.add_axes([0.08, 0.15, 0.5, 0.7])  
    ax_z.bar(edges[:-1], counts, width=np.diff(edges), color=colors_z, align='edge', edgecolor='none')
    ax_z.set_xlabel(r'$Z_{\mathrm{daily}}^{\prime}$', fontsize=7)
    ax_z.set_ylabel('Counts', fontsize=7)
    ax_z.set_title('(a)', fontsize=10,fontweight='bold',loc='left')
    ax_z.spines['top'].set_visible(False)
    ax_z.spines['right'].set_visible(False)
    ax_z.set_ylim([0, 250])

    ax_pw = fig.add_axes([0.68, 0.60, 0.3, 0.3])
    min_val = min(prc1w.min(), prc2w.min())
    max_val = max(prc1w.max(), prc2w.max())
    bins_pw = np.linspace(min_val, max_val, 20)
    sns.histplot(prc1w, bins=bins_pw, stat="density", color='#1E90FF', alpha=0.25, edgecolor='none', linewidth=0.01, ax=ax_pw)
    sns.histplot(prc2w, bins=bins_pw, stat="density", color='#004687', alpha=0.25, edgecolor='none', linewidth=0.01, ax=ax_pw)
    sns.kdeplot(prc1w, color='#1E90FF', linewidth=1.2, ax=ax_pw, label='1979-1999', bw_adjust=1.5)
    sns.kdeplot(prc2w, color='#004687', linewidth=1.2, ax=ax_pw, label='2000-2020', bw_adjust=1.5)
    median1 = np.nanmedian(prc1w)
    median2 = np.nanmedian(prc2w)

    ax_pw.set_xlabel('mm/day', fontsize=7)
    ax_pw.set_ylabel('Density', fontsize=7)
    ax_pw.set_title('(b)', fontsize=10, fontweight='bold',loc='left')
    ax_pw.legend(fontsize=7)
    ax_pw.grid(False)

    ax_ps = fig.add_axes([0.68, 0.15, 0.3, 0.3])
    min_val = min(prc1s.min(), prc2s.min())
    max_val = max(prc1s.max(), prc2s.max())
    bins_ps = np.linspace(min_val, max_val, 20)
    sns.histplot(prc1s, bins=bins_ps, stat="density", color='#FFAC1C', alpha=0.25, edgecolor='none', linewidth=0.01, ax=ax_ps)
    sns.histplot(prc2s, bins=bins_ps, stat="density", color='#CC5500', alpha=0.25, edgecolor='none', linewidth=0.01, ax=ax_ps)
    sns.kdeplot(prc1s, color='#FFAC1C', linewidth=1.2, ax=ax_ps, label='1979-1999')
    sns.kdeplot(prc2s, color='#CC5500', linewidth=1.2, ax=ax_ps, label='2000-2020')
    median1 = np.nanmedian(prc1s)
    median2 = np.nanmedian(prc2s)

    ax_ps.set_xlabel('mm/day', fontsize=7)
    ax_ps.set_ylabel('Density', fontsize=7)
    ax_ps.set_title('(c)', fontsize=10, fontweight='bold',loc='left')
    ax_ps.legend(fontsize=7)
    ax_ps.grid(False)
    
    plt.tight_layout()
    plt.savefig('ght_prec_distribution_IND.pdf',dpi=900)
    plt.show() 
    plt.close()

    statistical_test(prc1w,prc2w)
    statistical_test(prc1s,prc2s)
    
    return 0    

def statistical_test(X1,X2):
    p_value = ks_2samp(X1,X2).pvalue
    statistic = ks_2samp(X1,X2).statistic
    print(len(X1),len(X2),p_value,statistic)
    return 0

ght_distribution()