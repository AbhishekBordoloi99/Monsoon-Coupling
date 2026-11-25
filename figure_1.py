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
from matplotlib.ticker import AutoMinorLocator

def corr_bars():
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Nimbus Sans L', 'Arial'],
        'font.size': 7,
        'axes.labelsize': 7,
        'axes.titlesize': 7,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'figure.dpi': 300,
        'savefig.dpi': 900,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'axes.linewidth': 1.2,
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        'hatch.linewidth': 0.6
    })
    fig = plt.figure(figsize=(7.2, 5.4))
    ax_n = fig.add_axes([-0.01, 0.55, 0.55, 0.50], projection=ccrs.Mollweide(central_longitude=0))
    ax_s = fig.add_axes([0.54, 0.55, 0.55, 0.50], projection=ccrs.Mollweide(central_longitude=0))

    ax_north = fig.add_axes([0.01, 0.10, 0.50, 0.40])
    ax_south = fig.add_axes([0.58, 0.10, 0.50, 0.40])

    path = 'cru_ts4.09.1901.2024.pre.dat.nc'
    P = xr.open_dataset(path)

    year_map = np.arange(1975, 2025)
    PRC_NH, PRC_SH = [], []

    for i in year_map:
        for j in range(6, 9):
            data = P.sel(time=str(i), lat=slice(-20, 90)).pre
            PRC_NH.append(data.isel(time=j).values)
    PRC_NH = np.mean(np.array(PRC_NH), axis=0) / 30

    for i in year_map:
        for j in [0, 1]:
            data = P.sel(time=str(i), lat=slice(-90, 20)).pre
            PRC_SH.append(data.isel(time=j).values)
    PRC_SH = np.mean(np.array(PRC_SH), axis=0) / 30

    lat_nh = P.lat.sel(lat=slice(-20, 90)).values
    lat_sh = P.lat.sel(lat=slice(-90, 20)).values
    lon_full = P.lon.values

    year_nh = np.linspace(1975, 2024, 50, dtype=int)
    nams = P.sel(lat=slice(10, 25), lon=slice(-100, -80)).mean(('lat', 'lon'))
    waf = P.sel(lat=slice(10, 15), lon=slice(-20, 20)).mean(('lat', 'lon'))
    ismr = P.sel(lat=slice(18, 28), lon=slice(70, 90)).mean(('lat', 'lon'))
    mc_nh = P.sel(lat=slice(-10, 10), lon=slice(95, 150)).mean(('lat', 'lon'))

    NAMS, WAF, ISMR, MC_nh = [], [], [], []
    for i in year_nh:
        tmp1, tmp2, tmp3, tmp4 = 0, 0, 0, 0
        for j in range(6, 9):
            tmp1 += nams.sel(time=str(i)).groupby('time.month')[j].mean('time').pre.values
            tmp2 += waf.sel(time=str(i)).groupby('time.month')[j].mean('time').pre.values
            tmp3 += ismr.sel(time=str(i)).groupby('time.month')[j].mean('time').pre.values
            tmp4 += mc_nh.sel(time=str(i)).groupby('time.month')[j].mean('time').pre.values
        NAMS.append(tmp1/4)
        WAF.append(tmp2/4)
        ISMR.append(tmp3/4)
        MC_nh.append(tmp4/4)

    NAMS, WAF, ISMR, MC_nh = np.array(NAMS), np.array(WAF), np.array(ISMR), np.array(MC_nh)
    NAMS[0:25]=detrend(NAMS[0:25])
    WAF[0:25]=detrend(WAF[0:25])
    ISMR[0:25]=detrend(ISMR[0:25])
    MC_nh[0:25]=detrend(MC_nh[0:25])

    NAMS[25:]=detrend(NAMS[25:])
    WAF[25:]=detrend(WAF[25:])
    ISMR[25:]=detrend(ISMR[25:])
    MC_nh[25:]=detrend(MC_nh[25:])

    variables_nh = [NAMS, WAF, ISMR, MC_nh]
    var_names_nh = ['CAM', 'WAM', 'ISM', 'MC']

    NH3 = []
    NH4 = []
    for i in range(len(variables_nh)):
        for j in range(i+1, len(variables_nh)):
            corr3 = np.corrcoef(variables_nh[i][0:25], variables_nh[j][0:25])[0, 1]
            corr4 = np.corrcoef(variables_nh[i][25:], variables_nh[j][25:])[0, 1]
            NH3.append(corr3)
            NH4.append(corr4)

    NH3 = np.array(NH3)
    NH4 = np.array(NH4)
    print(NH3)
    print(NH4)
    corr_labels_nh = ['CAM-WAM', 'CAM-ISM', 'CAM-MC', 'WAM-ISM', 'WAM-MC', 'ISM-MC']
    colors_nh = ['#66C2A5', '#91BFDB', '#A6D96A', '#FDAE61', '#EF8A62', '#C2A5CF']

    year_sh = np.linspace(1974, 2023, 50, dtype=int)
    aus = P.sel(lat=slice(-25, -10), lon=slice(120, 160)).mean(('lat', 'lon'))
    sam = P.sel(lat=slice(-25, -10), lon=slice(-80, -40)).mean(('lat', 'lon'))
    safc = P.sel(lat=slice(-25, -10), lon=slice(10, 40)).mean(('lat', 'lon'))
    mc_sh = P.sel(lat=slice(-10, 10), lon=slice(95, 150)).mean(('lat', 'lon'))

    AUS, SAM, SAFC, MC_sh = [], [], [], []
    for i in year_sh:
        tmp1, tmp2, tmp3, tmp4 = 0, 0, 0, 0
        for j in range(12, 13):
            tmp1 += aus.sel(time=str(i)).groupby('time.month')[j].mean('time').pre.values
            tmp2 += sam.sel(time=str(i)).groupby('time.month')[j].mean('time').pre.values
            tmp3 += safc.sel(time=str(i)).groupby('time.month')[j].mean('time').pre.values
            tmp4 += mc_sh.sel(time=str(i)).groupby('time.month')[j].mean('time').pre.values
        for j in range(1, 3):
            tmp1 += aus.sel(time=str(i+1)).groupby('time.month')[j].mean('time').pre.values
            tmp2 += sam.sel(time=str(i+1)).groupby('time.month')[j].mean('time').pre.values
            tmp3 += safc.sel(time=str(i+1)).groupby('time.month')[j].mean('time').pre.values
            tmp4 += mc_sh.sel(time=str(i+1)).groupby('time.month')[j].mean('time').pre.values
        if i == 2020:
            break
        AUS.append(tmp1/3)
        SAM.append(tmp2/3)
        SAFC.append(tmp3/3)
        MC_sh.append(tmp4/3)

    AUS, SAM, SAFC, MC_sh = np.array(AUS), np.array(SAM), np.array(SAFC), np.array(MC_sh)
    AUS[0:25]=detrend(AUS[0:25])
    SAM[0:25]=detrend(SAM[0:25])
    SAFC[0:25]=detrend(SAFC[0:25])
    MC_sh[0:25]=detrend(MC_sh[0:25])

    AUS[25:]=detrend(AUS[25:])
    SAM[25:]=detrend(SAM[25:])
    SAFC[25:]=detrend(SAFC[25:])
    MC_sh[25:]=detrend(MC_sh[25:])

    variables_sh = [AUS, SAM, SAFC]
    var_names_sh = ['AUS', 'SAM', 'SAFC']

    NH3_sh = []
    NH4_sh = []
    for i in range(len(variables_sh)):
        for j in range(i+1, len(variables_sh)):
            corr3 = np.corrcoef(variables_sh[i][0:25], variables_sh[j][0:25])[0, 1]
            corr4 = np.corrcoef(variables_sh[i][25:], variables_sh[j][25:])[0, 1]
            NH3_sh.append(corr3)
            NH4_sh.append(corr4)

    NH3_sh = np.array(NH3_sh)
    NH4_sh = np.array(NH4_sh)

    corr_labels_sh = ['AUS-SAM', 'AUS-SAFC', 'SAM-SAFC']
    colors_sh = ['#91BFDB', '#FDAE61', '#C2A5CF']
    
    print(NH3_sh)
    print(NH4_sh)

    gs = gridspec.GridSpec(
        2, 2,
        figure=fig,
        height_ratios=[1.2, 1],  
        width_ratios=[1, 1],
        hspace=0.25,            
        wspace=0.25,
        top=0.95,
        bottom=0.10,
        left=0.08,
        right=0.98
    )

    proj = ccrs.Mollweide(central_longitude=0)
    levels = np.linspace(0, 16, 17)

    mesh_nh = ax_n.pcolormesh(
        lon_full, lat_nh, PRC_NH,
        transform=ccrs.PlateCarree(),
        cmap=cmaps.gsltod,
        shading='auto',
        vmin=0, vmax=12
    )

    mesh_sh = ax_s.pcolormesh(
        lon_full, lat_sh, PRC_SH,
        transform=ccrs.PlateCarree(),
        cmap=cmaps.gsltod,
        shading='auto',
        vmin=0, vmax=12
    )

    for ax in [ax_n, ax_s]:
        ax.add_feature(cfeature.OCEAN, facecolor='none')
        ax.add_feature(cfeature.COASTLINE, linewidth=0.01)
        ax.gridlines(linewidth=0.5, linestyle='--', color='gray', alpha=0.1)
    ax_n.outline_patch.set_linewidth(0.5)
    ax_s.outline_patch.set_linewidth(0.5)

    ax_n.set_title("(a)", fontsize=10, fontweight='bold', loc='left')
    ax_s.set_title("(b)", fontsize=10, fontweight='bold', loc='left')

    ax_n.text(0.62, 0.65, 'R1=0.06\nR2=0.41', fontsize=7, transform=ax_n.transAxes,fontweight='bold')
    ax_n.text(0.35, 0.55, 'R1=-0.01\nR2=0.41', fontsize=7, transform=ax_n.transAxes,fontweight='bold')
    ax_n.text(0.56, 0.40, 'R1=0.33\nR2=0.65', fontsize=7, transform=ax_n.transAxes,fontweight='bold')
    ax_n.text(0.85, 0.45, 'R1=0.18\nR2=0.43', fontsize=7, transform=ax_n.transAxes,fontweight='bold')
    ax_n.text(0.66, 0.15, 'R1=0.19\nR2=0.40', fontsize=7, transform=ax_n.transAxes,fontweight='bold')
    ax_n.text(0.40, 0.13, 'R1=0.40\nR2=0.31', fontsize=7, transform=ax_n.transAxes,fontweight='bold')

    ax_s.text(0.40, 0.63, 'R1=-0.34\nR2=0.17', fontsize=7, transform=ax_s.transAxes,fontweight='bold')
    ax_s.text(0.65, 0.60, 'R1=0.35\nR2=0.48', fontsize=7, transform=ax_s.transAxes,fontweight='bold')
    ax_s.text(0.55, 0.20, 'R1=-0.14\nR2=-0.08', fontsize=7, transform=ax_s.transAxes,fontweight='bold')

    def add_box(ax, r):
        ax.add_patch(
            mpatches.Rectangle(
                r["xy"], r["w"], r["h"],
                linewidth=1,
                edgecolor=r["c"],
                facecolor='none',
                transform=ccrs.PlateCarree(),
                zorder=2
            )
        )

    regions_nh = {
        "Central America": {"xy": (-100, 10), "w": 40, "h": 10, "c": "black"},
        "West Africa":     {"xy": (-20, 10),  "w": 40, "h": 10, "c": "black"},
        "India":           {"xy": (70, 18),   "w": 20, "h": 10, "c": "black"},
        "Maritime Cont":   {"xy": (95, -10),  "w": 55, "h": 20, "c": "black"},
    }

    regions_sh = {
        "Australia":     {"xy": (120, -25), "w": 40, "h": 15, "c": "black"},
        "South America": {"xy": (-80, -25), "w": 40, "h": 15, "c": "black"},
        "South Africa":  {"xy": (10, -25),  "w": 30, "h": 15, "c": "black"},
    }


    centers_nh = {
        "Central America": (-90, 15),
        "West Africa": (-5, 15),
        "India": (80, 19),
        "Maritime Cont": (122.5, 0),
    }


    centers_sh = {
        "Australia": (140, -15),
        "South America": (-60, -15),
        "South Africa": (25, -15),
    }

    def curved_double_arrow(ax, start, end, rad, c1, c2):
        base = FancyArrowPatch(
            start, end,
            arrowstyle='-',
            connectionstyle=f'arc3,rad={rad}',
            linewidth=1.2,
            color='none',
            transform=ccrs.PlateCarree(),
            zorder=3
        )
        ax.add_patch(base)
        ax.figure.canvas.draw()


        path = base.get_path()
        verts = path.vertices
        codes = path.codes


        start_arr = np.array(start)
        if np.linalg.norm(verts[-1] - start_arr) < np.linalg.norm(verts[0] - start_arr):
            verts = verts[::-1]
            codes = codes[::-1]


        d = np.sqrt(np.sum(np.diff(verts, axis=0)**2, axis=1))
        s = np.concatenate([[0], np.cumsum(d)])
        s /= s[-1]
        mid = np.searchsorted(s, 0.5)


        codes1 = codes[:mid+1].copy()
        codes1[0] = Path.MOVETO
        ax.add_patch(FancyArrowPatch(path=Path(verts[:mid+1], codes1),arrowstyle='-',linewidth=1.2,color=c1,transform=ccrs.PlateCarree(),zorder=3))


        codes2 = codes[mid:].copy()
        codes2[0] = Path.MOVETO
        ax.add_patch(FancyArrowPatch(path=Path(verts[mid:], codes2),arrowstyle='-',linewidth=1.2,color=c2,transform=ccrs.PlateCarree(),zorder=3))

        ax.add_patch(FancyArrowPatch(verts[-2], verts[-1],arrowstyle='->',linewidth=1.2,color=c2,mutation_scale=15,transform=ccrs.PlateCarree(),zorder=3))
        ax.add_patch(FancyArrowPatch(verts[1], verts[0],arrowstyle='->',linewidth=1.2,color=c1,mutation_scale=15,transform=ccrs.PlateCarree(),zorder=3))


    connections_nh = [
        ("Central America", "West Africa", -0.4, "#66C2A5", "#1B9E77"),
        ("West Africa", "India", 0.20, "#FDAE61", "#D95F02"),
        ("Central America", "India", -0.65, "#91BFDB", "#4575B4"),
        ("India", "Maritime Cont",-1.5, "#C2A5CF", "#7B3294"),
        ("West Africa", "Maritime Cont", 0.25, "#EF8A62", "#B2182B"),
        ("Central America", "Maritime Cont", 0.25, "#A6D96A", "#4D9221"),
    ]


    connections_sh = [
        ("Australia", "South America", -0.3, "#4575B4", "#91BFDB"),
        ("Australia", "South Africa", 0.3, "#D95F02", "#FDAE61"),
        ("South America", "South Africa", 0.25, "#C2A5CF", "#7B3294"),
    ]


    for a, b, rad, c1, c2 in connections_nh:
        curved_double_arrow(ax_n, centers_nh[a], centers_nh[b], rad, c1, c2)
    for a, b, rad, c1, c2 in connections_sh:
        curved_double_arrow(ax_s, centers_sh[a], centers_sh[b], rad, c1, c2)

    for r in regions_nh.values():
        add_box(ax_n, r)
    for r in regions_sh.values():
        add_box(ax_s, r)

    cbar_ax = fig.add_axes([0.35, 0.63, 0.35, 0.02])
    cbar = fig.colorbar(mesh_nh, cax=cbar_ax, orientation='horizontal', extend='max', drawedges=True)
    cbar.set_label('Precipitation (mm/day)', fontsize=7)
    cbar.ax.tick_params(labelsize=7)
    cbar.outline.set_edgecolor('black')
    cbar.outline.set_linewidth(0.75)

    x_nh = np.arange(len(corr_labels_nh))
    width_nh = 0.25

    bars3_nh = ax_north.bar(x_nh, NH3, width_nh, label='NH3', color=colors_nh, linewidth=0.5,
                        hatch='\\\\', alpha=0.95, edgecolor='black', clip_on=False)
    bars4_nh = ax_north.bar(x_nh + width_nh, NH4, width_nh, label='NH4', color=colors_nh, linewidth=0.5,
                        hatch='..', alpha=0.95, edgecolor='black', clip_on=False)

    ax_north.set_xlabel('Region Pairs', fontsize=7, fontweight='bold', labelpad=8)
    ax_north.set_ylabel('Correlation Coefficient', fontsize=7, fontweight='bold', labelpad=8)
    ax_north.set_title('(c)', fontsize=10, fontweight='bold', loc='left')
    ax_north.set_xticks(x_nh)
    ax_north.set_xticklabels(corr_labels_nh, rotation=0, ha='center', rotation_mode='anchor')
    ax_north.tick_params(axis='x', pad=8)

    legend_elements_nh = [
        plt.Rectangle((0,0), 1, 1, facecolor='#f0f0f0', hatch='\\\\', 
                    edgecolor='black', linewidth=1.0, label='1975-1999'),
        plt.Rectangle((0,0), 1, 1, facecolor='#f0f0f0', hatch='....', 
                    edgecolor='black', linewidth=1.0, label='2000-2024')
    ]
    ax_north.legend(handles=legend_elements_nh, frameon=True, fancybox=False, shadow=True, 
                framealpha=1, edgecolor='black', ncol=1, loc='upper left')

    ax_north.grid(True, axis='y', alpha=0.4, linestyle='-', linewidth=0.5)
    ax_north.set_ylim(-0.05, 0.75)
    ax_north.spines['top'].set_visible(False)
    ax_north.spines['right'].set_visible(False)
    ax_north.spines['left'].set_linewidth(1.0)
    ax_north.spines['bottom'].set_linewidth(1.0)

    x_sh = np.arange(len(corr_labels_sh))
    width_sh = 0.15

    bars3_sh = ax_south.bar(x_sh, NH3_sh, width_sh, label='NH3', color=colors_sh, linewidth=0.5,
                        hatch='\\\\', alpha=0.95, edgecolor='black', clip_on=False)
    bars4_sh = ax_south.bar(x_sh + width_sh, NH4_sh, width_sh, label='NH4', color=colors_sh, linewidth=0.5,
                        hatch='..', alpha=0.95, edgecolor='black', clip_on=False)

    ax_south.set_xlabel('Region Pairs', fontsize=7, fontweight='bold', labelpad=8)
    #ax_south.set_ylabel('Correlation Coefficient', fontsize=7, fontweight='bold', labelpad=8)
    ax_south.set_title('(d)', fontsize=10, fontweight='bold', loc='left')
    ax_south.set_xticks(x_sh)
    ax_south.set_xticklabels(corr_labels_sh, rotation=0, ha='center', rotation_mode='anchor')
    ax_south.tick_params(axis='x', pad=8)

    legend_elements_sh = [
        plt.Rectangle((0,0), 1, 1, facecolor='#f0f0f0', hatch='\\\\', 
                    edgecolor='black', linewidth=1.0, label='1975-1999'),
        plt.Rectangle((0,0), 1, 1, facecolor='#f0f0f0', hatch='....', 
                    edgecolor='black', linewidth=1.0, label='2000-2024')
    ]
    ax_south.legend(handles=legend_elements_sh, frameon=True, fancybox=False, shadow=True, 
                framealpha=1, edgecolor='black', ncol=1, loc='upper left')

    ax_south.grid(True, axis='y', alpha=0.4, linestyle='-', linewidth=0.5)
    ax_south.set_ylim(-0.4, 0.6)
    ax_south.spines['top'].set_visible(False)
    ax_south.spines['right'].set_visible(False)
    ax_south.spines['left'].set_linewidth(1.0)
    ax_south.spines['bottom'].set_linewidth(1.0)

    plt.tight_layout(pad=1.5, rect=[0, 0, 1, 0.95])
    plt.show()
    return 0

corr_bars()