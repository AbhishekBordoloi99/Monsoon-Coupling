import numpy as np
import xarray as xr
import matplotlib.pyplot as plt 
from mpl_toolkits.basemap import Basemap
import cmaps
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Polygon
import math
import pandas as pd
from matplotlib.colors import ListedColormap
import cartopy.crs as ccrs
import cartopy.mpl.ticker as cticker
from cartopy.util import add_cyclic_point
import matplotlib.patches as mpatches
import matplotlib
from sklearn.decomposition import PCA
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import signal
from eofs.standard import Eof
from windspharm.standard import VectorWind
from scipy.ndimage import rotate
from scipy.ndimage import label
import random
from scipy.stats import norm
import scipy.stats as stats
from sklearn.metrics import r2_score
from scipy.stats import linregress
from scipy.interpolate import make_interp_spline
from scipy.stats import gaussian_kde
from scipy.stats import ks_2samp
import itertools
from scipy.signal import detrend
from scipy.signal import periodogram
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import ConnectionPatch
from collections import defaultdict
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import cartopy.feature as cfeature
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib import transforms
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import pearsonr
from scipy.signal import butter, filtfilt
import geopandas as gpd
from matplotlib import gridspec
import csv
import matplotlib.patches as patches
import os
import pyproj
import matplotlib as mpl
from scipy.stats import mannwhitneyu

# Ensure pyproj uses the conda‑installed PROJ data only
proj_data_dir = "/home/abhishek/anaconda3/envs/conda_env/share/proj"

# Clear any conflicting PROJ_LIB from outside
if "PROJ_LIB" in os.environ:
    del os.environ["PROJ_LIB"]

# Force pyproj to use the conda data directory
pyproj.datadir.set_data_dir(proj_data_dir)

def station_analysis():
    path='DATA/station_data/tpw_station_time_nc4.nc'
    tpw=xr.open_dataset(path)
    ## Climatology
    AFC_clm=[]
    IND_clm=[]
    AMS_clm=[]
    MC_clm=[]

    year=np.linspace(1980,2023,44,dtype=int)

    for i in year:
        for j in range(0,2):
            tmp=[]
            tmp.append(tpw.sel(time=str(i)).tpw.values[j,0:730])
            AFC_clm.append(np.where((np.array(tmp, dtype=float) > 400) | (np.array(tmp, dtype=float) <=0), np.nan, np.array(tmp)))
        for j in range(2,10):
            tmp=[]
            tmp.append(tpw.sel(time=str(i)).tpw.values[j,0:730])
            IND_clm.append(np.where((np.array(tmp, dtype=float) > 400) | (np.array(tmp, dtype=float) <=0), np.nan, np.array(tmp)))
        for j in range(10,23):
            tmp=[]
            tmp.append(tpw.sel(time=str(i)).tpw.values[j,0:730])
            AMS_clm.append(np.where((np.array(tmp, dtype=float) > 400) | (np.array(tmp, dtype=float) <=0), np.nan, np.array(tmp)))
        for j in range(23,26):
            tmp=[]
            tmp.append(tpw.sel(time=str(i)).tpw.values[j,0:730])
            MC_clm.append(np.where((np.array(tmp, dtype=float) > 400) | (np.array(tmp, dtype=float) <=0), np.nan, np.array(tmp)))

    IND_clm=np.nanmean(np.array(IND_clm),axis=0)
    AFC_clm=np.nanmean(np.array(AFC_clm),axis=0)
    AMS_clm=np.nanmean(np.array(AMS_clm),axis=0)
    MC_clm=np.nanmean(np.array(MC_clm),axis=0)

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
    AFC_anom=[]
    IND_anom=[]
    AMS_anom=[]
    MC_anom=[]
    
    for i in year:
        tmp1=[]
        for j in range(0,2):
            t=np.where((tpw.sel(time=str(i)).tpw.values[j,0:730]>100) | (tpw.sel(time=str(i)).tpw.values[j,0:730] <0) , np.nan, tpw.sel(time=str(i)).tpw.values[j,0:730])
            tmp1.append(t)
        tmp1=np.nanmean(np.array(tmp1),axis=0)
        AFC_anom.append(tmp1-AFC_clm[0])

        tmp2=[]
        for j in range(2,10):
            t=np.where((tpw.sel(time=str(i)).tpw.values[j,0:730]>100) | (tpw.sel(time=str(i)).tpw.values[j,0:730] <0) , np.nan, tpw.sel(time=str(i)).tpw.values[j,0:730])
            tmp2.append(t)
        tmp2=np.nanmean(np.array(tmp2),axis=0)
        IND_anom.append(tmp2-IND_clm[0])

        tmp3=[]
        for j in range(10,23):
            t=np.where((tpw.sel(time=str(i)).tpw.values[j,0:730]>100) | (tpw.sel(time=str(i)).tpw.values[j,0:730] <0) , np.nan, tpw.sel(time=str(i)).tpw.values[j,0:730])
            tmp3.append(t)
        tmp3=np.nanmean(np.array(tmp3),axis=0)
        AMS_anom.append(tmp3-AMS_clm[0])

        tmp4=[]
        for j in range(23,26):
            t=np.where((tpw.sel(time=str(i)).tpw.values[j,0:730]>100) | (tpw.sel(time=str(i)).tpw.values[j,0:730] <0) , np.nan, tpw.sel(time=str(i)).tpw.values[j,0:730])
            tmp4.append(t)
        tmp4=np.nanmean(np.array(tmp4),axis=0)
        MC_anom.append(tmp4-MC_clm[0])

    AFC_anom=np.array(AFC_anom)[:,300:500]
    IND_anom=np.array(IND_anom)[:,300:500]
    AMS_anom=np.array(AMS_anom)[:,300:500]
    MC_anom=np.array(MC_anom)[:,300:500]

        
    frac_afc=[]
    avg_afc=[]

    frac_ams=[]
    avg_ams=[]

    frac_ams_afc=[]
    avg_ams_afc=[]

    idx_list=defaultdict(list)
    events_dates=[]
    for i in range(0,len(IND_anom)):
        std_ind=np.nanstd(IND_anom[i,:])
        std_afc=np.nanstd(AFC_anom[i,:])
        count=0
        count_pos_afc=0
        count_pos_ams=0

        count_pos_ams_afc=0

        av_afc=[]
        av_ams=[]

        av_ams_afc=[]
        
        for j in range(5,len(IND_anom[i,:])):
            if(IND_anom[i,j]>std_ind and np.nanmean(IND_anom[i,j-1:j])<std_ind and np.nanmean(IND_anom[i,j-2:j-1])<std_ind):
                count+=1
                idx_list[year[i]].append((300+j)/2)
                if(np.nanmean(AFC_anom[i,j+20:j+60])>np.nanmean(AFC_anom[i,j+10:j+20]) and np.nanmean(AFC_anom[i,j+20:j+60])>0):
                    events_dates.append((i,j))
                    count_pos_afc+=1
                    av_afc.append(np.nanmean(AFC_anom[i,j+20:j+60]))
                if(np.nanmean(AMS_anom[i,j+60:j+80])>np.nanmean(AMS_anom[i,j+40:j+60]) and np.nanmean(AMS_anom[i,j+60:j+80])>0):
                    count_pos_ams+=1
                    av_ams.append(np.nanmean(AMS_anom[i,j+60:j+80]))
        if(count!=0):
            frac_afc.append(count_pos_afc/count)
            frac_ams.append(count_pos_ams/count)
            avg_afc.append(np.mean(np.array(av_afc)))
            avg_ams.append(np.mean(np.array(av_ams)))
        else:
            frac_afc.append(np.nan)
            frac_ams.append(np.nan)
            avg_afc.append(np.nan)
            avg_ams.append(np.nan)

        count=0
        for j in range(5,len(AFC_anom[i,:])):
            if (AFC_anom[i, j] > std_afc and np.nanmean(AFC_anom[i, j-1:j]) < std_afc and np.nanmean(AFC_anom[i, j-2:j-1]) < std_afc):
                count+=1
                idx_list[year[i]].append((300+j)/2)
                if (np.nanmean(AMS_anom[i, j+40:j+80]) > np.nanmean(AMS_anom[i, j+0:j+40]) and np.nanmean(AMS_anom[i, j+40:j+80]) > 0):
                    count_pos_ams_afc+=1
                    av_ams_afc.append(np.nanmean(AMS_anom[i,j+40:j+80]))
        if(count!=0):
            frac_ams_afc.append(count_pos_ams_afc/count)
            avg_ams_afc.append(np.mean(np.array(av_ams_afc)))
        else:
            frac_ams_afc.append(np.nan)
            avg_ams_afc.append(np.nan)
    
    def split_and_clean(data):
        group1 = [x for x in data[:20] if not np.isnan(x)]
        group2 = [x for x in data[20:40] if not np.isnan(x)]
        return [group1, group2]

    afc_groups = split_and_clean(frac_afc)
    ams_groups = split_and_clean(frac_ams)
    ams_afc_groups = split_and_clean(frac_ams_afc)

    av_afc_groups = split_and_clean(avg_afc)
    av_ams_groups = split_and_clean(avg_ams)
    av_ams_afc_groups = split_and_clean(avg_ams_afc)

    colors = ["#E57373", "#6699CC"]

    def plot_boxplot(ax, data, labels, title, colors):
        bplot = ax.boxplot(data, patch_artist=True, labels=labels, showfliers=False, widths=0.25)
        
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor('lightgray')
            patch.set_edgecolor('black')
            patch.set_linewidth(0.25)
        
        for median in bplot['medians']:
            median.set_color('black')
            median.set_linewidth(0.35)
        
        for whisker in bplot['whiskers']:
            whisker.set_color('gray')
            whisker.set_linewidth(0.3)
        for cap in bplot['caps']:
            cap.set_color('gray')
            cap.set_linewidth(0.3)
        
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        for spine in ax.spines.values():
            spine.set_linewidth(0.25)
        
        if data == afc_groups:

            ax.set_ylim(-0.05, 1)
            ax.set_ylabel('Fraction', fontsize=7)
            ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            ax.tick_params(axis='y', which='major', labelsize=7)

            rect1 = patches.Rectangle((0.0, 0.85), 0.5, 0.15,transform=ax.transAxes,facecolor='#f7c183',edgecolor='none',linewidth=0.25,clip_on=False)
            rect2 = patches.Rectangle((0.5, 0.85), 0.5, 0.15,transform=ax.transAxes,facecolor='#b5dcff',edgecolor='none',linewidth=0.25,clip_on=False)

        elif data == ams_groups:

            ax.set_ylim(-0.05, 1)
            ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            ax.tick_params(axis='y', which='major', labelsize=7)

        elif data == ams_afc_groups:

            ax.set_ylim(-0.05, 1)
            ax.set_ylabel('', fontsize=16)
            ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            ax.tick_params(axis='y', which='major', labelsize=7)

            rect1 = patches.Rectangle((0.0, 0.85), 0.5, 0.15,transform=ax.transAxes,facecolor='#b5dcff',edgecolor='none',linewidth=0.25,clip_on=False)
            rect2 = patches.Rectangle((0.5, 0.85), 0.5, 0.15,transform=ax.transAxes,facecolor='#b5f7c2',edgecolor='none',linewidth=0.25,clip_on=False)

        elif data == av_afc_groups:
            ax.set_ylim(0, 9)
            ax.set_ylabel('Moisture anomaly $(kg/m^2)$', fontsize=7)
            ax.tick_params(axis='both', which='major', labelsize=7)

            rect1 = patches.Rectangle((0.0, 0.85), 1, 0.15,transform=ax.transAxes,facecolor='#b5dcff',edgecolor='none',linewidth=0.25,clip_on=False)
        elif data == av_ams_groups or data == av_ams_afc_groups:
            ax.set_ylim(0, 9)
            ax.tick_params(axis='both', which='major', labelsize=7)
            rect1 = patches.Rectangle((0.0, 0.85), 1, 0.15,transform=ax.transAxes,facecolor='#b5f7c2',edgecolor='none',linewidth=0.25,clip_on=False)
        
        ax.text(0.01, 1.1, title, fontsize=10, ha='left', va='top', transform=ax.transAxes,fontweight='bold')
        ax.tick_params(axis='both', length=1.2, width=0.2, direction='out', which='major')
        ax.tick_params(axis='both', length=1, width=0.2, direction='out', which='minor')

    locations = [
        {"lat": 13.480, "lon": 2.160, "name": "1. NIAMEY-AERO, NIGER","Index":"1"},
        {"lat": 12.350, "lon": -1.510, "name": "2. BURKINA-FASO","Index":"2"},
        {"lat": 26.300, "lon": 73.010, "name": "3. JODHPUR","Index":"3"},
        {"lat": 23.060, "lon": 72.630, "name": "4. AHMADABAD","Index":"4"},
        {"lat": 28.580 , "lon": 77.200, "name": "5. NEW-DELHI","Index":"5"},
        {"lat": 23.280 , "lon": 77.350, "name": "6. BHOPAL/BAIRAGARH","Index":"6"},
        {"lat": 19.110 , "lon": 72.850, "name": "7. BOMBAY / SANTACRUZ","Index":"7"},
        {"lat": 18.530 , "lon": 73.850, "name": "8. POONA","Index":"8"},
        {"lat": 21.100 , "lon": 79.050, "name": "9. NAGPUR SONEGAON","Index":"9"},
        {"lat": 21.230 , "lon": 81.650, "name": "10. PBO RAIPUR","Index":"10"},
        {"lat": 27.779 , "lon": -97.505, "name": "11. CORPUS CHRISTI/INT","Index":"11"},
        {"lat": 33.180 , "lon": -86.783, "name": "12. BIRMINGHAM, AL, USA","Index":"12"},
        {"lat": 32.835 , "lon": -97.297, "name": "13. FORT WORTH, TX., USA","Index":"13"},
        {"lat": 35.233 , "lon": -101.709, "name": "14. AMARILLO/INTL., TX., USA","Index":"14"},
        {"lat": 30.338 , "lon": -89.825, "name": "15. SLIDELL/MUN. LA, USA","Index":"15"},
        {"lat": 34.836 , "lon": -92.260, "name": "16. LITTLE ROCK/ADAMS FLD, AR., USA","Index":"16"},
        {"lat": 19.400 , "lon": -99.200, "name": "17. AEROP. INTERNACIONAL MEXICO","Index":"17"},
        {"lat": 19.050 , "lon": -104.320, "name": "18. MANZANILLO, COL., MEXICO","Index":"18"},
        {"lat": 17.500 , "lon": -88.330, "name": "19. BELIZE/PHILLIP GOLDSTON INTL.","Index":"19"},
        {"lat": 21.020 , "lon": -86.850, "name": "20. CANCUN, MEXICO","Index":"20"},
        {"lat": 17.939 , "lon": -76.781, "name": "21. KINGSTON/NORMAN MANLEY","Index":"21"},
        {"lat": 24.170 , "lon": -110.300, "name": "22. LA PAZ, BCS, MEXICO","Index":"22"},
        {"lat": 25.916 , "lon": -97.421, "name": "23. BROWNSVILLE/INT., TX, USA","Index":"23"}
    ]

    gs = gridspec.GridSpec(3, 4, height_ratios=[2, 3, 3], width_ratios=[1, 1, 1, 0.3], hspace=0.5, wspace=0.3)
    gdf = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(
            [loc["lon"] for loc in locations],
            [loc["lat"] for loc in locations]
        )
    )

    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))

    fig = plt.figure(figsize=(7, 7), dpi=900)

    ax_map = fig.add_axes([0.01, 0.55, 0.95, 0.25])
    ax_map.text(-130,40.1,'(a)',fontsize=10,fontweight='bold')
    world.plot(ax=ax_map, color="white", edgecolor="none", linewidth=0.2)
    ax_map.set_facecolor("#D9ECF9")

    for x, y, loc in zip(
        [loc["lon"] for loc in locations],
        [loc["lat"] for loc in locations],
        locations
    ):
        idx = int(loc["Index"])

        if idx in [1, 2]:
            mcolor = "#5B9BD5"  
        elif 3 <= idx <= 10:
            mcolor = "#E6953A"  
        else:
            mcolor = "#5AA469"  

        ax_map.scatter(x, y,s=20,marker='*',color=mcolor,edgecolor="none",linewidth=0.25,zorder=5)
        ax_map.text(x + 1, y + 0.5, loc["Index"], fontsize=7, ha='left', va='bottom')

    ax_map.spines['top'].set_visible(False)
    ax_map.spines['right'].set_visible(False)
    ax_map.spines['left'].set_visible(False)
    ax_map.spines['bottom'].set_visible(False)
    ax_map.set_xticks([])
    ax_map.set_yticks([])

    ax_map.set_xlim([-130, 130])
    ax_map.set_ylim([-5, 40])

    ax_map.set_title("", fontsize=16, fontweight='bold', pad=2)

    axs_box0 = fig.add_axes([0.16, 0.38, 0.18, 0.18])
    plot_boxplot(axs_box0, afc_groups, labels=['1980-2000', '2000-2020'], title='(b)', colors=colors)
    print(afc_groups)
    axs_box1 = fig.add_axes([0.44, 0.38, 0.18, 0.18])
    plot_boxplot(axs_box1, ams_afc_groups, labels=['1980-2000', '2000-2020'], title='(c)', colors=colors)
    print(ams_afc_groups)
    axs_box2 = fig.add_axes([0.16, 0.14, 0.18, 0.18])
    plot_boxplot(axs_box2, av_afc_groups, labels=['1980-2000', '2000-2020'], title='(d)', colors=colors)
    print(av_afc_groups)
    axs_box3 = fig.add_axes([0.44, 0.14, 0.18, 0.18])
    plot_boxplot(axs_box3, av_ams_afc_groups, labels=['1980-2000', '2000-2020'], title='(e)', colors=colors)
    print(av_ams_afc_groups)

    ax_names = fig.add_subplot(gs[1:, 3])
    ax_names.axis('off')
    ax_names.set_xlim(0, 1)
    ax_names.set_ylim(0, 1)

    station_text = "\n".join([loc["name"] for loc in locations])
    ax_names.text(-3,0.875,'(f)',fontsize=10,fontweight='bold')
    ax_names.text(
        -3, 0.85,
        station_text,
        fontsize=7,
        va='top', ha='left',
        linespacing=1.65,
        bbox=dict(
            facecolor='white',
            edgecolor='black',
            boxstyle='square,pad=0.65',
            linewidth=0.55,
            alpha=0.8
        )
    )
    plt.show()
    return 0
station_analysis()
