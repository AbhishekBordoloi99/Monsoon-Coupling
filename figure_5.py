import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import gaussian_kde
from matplotlib.ticker import ScalarFormatter, MaxNLocator
from matplotlib.lines import Line2D
import seaborn as sns
from scipy.stats import ks_2samp

mpl.rcParams.update({'font.size':10, 'axes.labelsize':11, 'axes.titlesize':11, 'xtick.labelsize':9, 'ytick.labelsize':9, 'axes.linewidth':1.25, 'xtick.direction':'out', 'ytick.direction':'out', 'xtick.major.width':0.9, 'ytick.major.width':0.9, 'legend.fontsize':9, 'font.family':'sans-serif', 'savefig.dpi':600})

BLUE, ORANGE = '#CC6677', '#4477AA'

PATH_GHT = '/ght/700/1979_2020.er_10_30.nc'
PATH_TCWV = '/tcwv/tcwv.1979_2023.era.nc'
PATH_RH = '/rel_hum/rel_hum700.nc'
PATH_PR = '/ERA5/ERA5_daily/precip/tp.{}.nc'

ght_ds = xr.open_dataset(PATH_GHT)
tcwv_ds = xr.open_dataset(PATH_TCWV)
rh_ds = xr.open_dataset(PATH_RH)

regions = [(-100,-80,10,20), (-20,20,10,20), (70,90,8,28)]
region_names = ['Central America', 'West Africa', 'India']


def get_events(lon1,lon2,lat1,lat2,years,variable='tcwv'):
    pr_all, w_all = [], []

    for year in years:
        precip = xr.open_dataset(PATH_PR.format(year)).sel(lat=slice(lat1,lat2),lon=slice(lon1,lon2)).mean(('lat','lon'))
        ght = ght_ds.sel(time=str(year),lat=slice(lat2,lat1),lon=slice(lon1,lon2)).mean(('lat','lon'))

        if variable == 'tcwv':
            moisture = tcwv_ds.sel(time=str(year),lat=slice(lat1,lat2),lon=slice(lon1,lon2)).mean(('lat','lon'))
        else:
            moisture = rh_ds.sel(time=str(year),latitude=slice(lat2,lat1),longitude=slice(lon1,lon2)).mean(('latitude','longitude'))

        ght_ts = np.array([ght.groupby('time.dayofyear')[j].mean('time').er.values for j in range(150,270)])
        ght_ts = -(ght_ts-np.mean(ght_ts))/np.std(ght_ts)

        for j in range(5,len(ght_ts)):
            if ght_ts[j-2]<1 and ght_ts[j-1]<1 and ght_ts[j]>1:
                tmp_pr,tmp_w = [],[]

                for k in range(6):
                    day = 150+j+k
                    tmp_pr.append(precip.groupby('time.dayofyear')[day].mean('time').tp.values)
                    tmp_w.append(moisture.groupby('time.dayofyear')[day].mean('time').tcwv.values if variable=='tcwv' else moisture.groupby('time.dayofyear')[day].mean('time').rh.values)

                pr_all.append(np.mean(tmp_pr)*1000*24)
                w_all.append(np.mean(tmp_w))

    return np.asarray(pr_all),np.asarray(w_all)

def binned_data(X,Y,bin_width):
    edges = np.arange(np.floor(np.nanmin(X)/bin_width)*bin_width,np.ceil(np.nanmax(X)/bin_width)*bin_width+bin_width,bin_width)
    data,centers = [],[]

    for l,r in zip(edges[:-1],edges[1:]):
        values = Y[(X>=l)&(X<r)]
        if len(values):
            data.append(values)
            centers.append((l+r)/2)

    return data,np.asarray(centers)

def plot_boxes(ax,X1,Y1,X2,Y2,bin_width,offset=0,box_scale=0.4):
    d1,c1 = binned_data(X1,Y1,bin_width)
    d2,c2 = binned_data(X2,Y2,bin_width)

    med1,med2 = np.array([np.nanmedian(x) for x in d1]),np.array([np.nanmedian(x) for x in d2])

    ax.boxplot(d1,positions=c1-offset,widths=bin_width*box_scale,patch_artist=True,showfliers=False,medianprops=dict(color=BLUE,lw=1),boxprops=dict(facecolor=BLUE,edgecolor=BLUE,alpha=0.25),whiskerprops=dict(color=BLUE,lw=1),capprops=dict(color=BLUE,lw=1))
    ax.boxplot(d2,positions=c2+offset,widths=bin_width*box_scale,patch_artist=True,showfliers=False,medianprops=dict(color=ORANGE,lw=1),boxprops=dict(facecolor=ORANGE,edgecolor=ORANGE,alpha=0.25),whiskerprops=dict(color=ORANGE,lw=1),capprops=dict(color=ORANGE,lw=1))

    ax.plot(c1-offset,med1,color=BLUE,marker='o',ms=3.5,lw=1.75,zorder=4)
    ax.plot(c2+offset,med2,color=ORANGE,marker='o',ms=3.5,lw=1.75,zorder=4)

def plot_tcwv_distribution(ax,X1,X2):
    X1,X2 = X1[np.isfinite(X1)],X2[np.isfinite(X2)]
    x_grid = np.linspace(min(X1.min(),X2.min()),max(X1.max(),X2.max()),500)

    sns.histplot(X1,bins=10,stat='density',color=BLUE,alpha=0.18,edgecolor=BLUE,linewidth=0.05,ax=ax)
    sns.histplot(X2,bins=10,stat='density',color=ORANGE,alpha=0.18,edgecolor=ORANGE,linewidth=0.05,ax=ax)

    if len(X1)>1 and np.std(X1)>0:
        ax.plot(x_grid,gaussian_kde(X1)(x_grid),color=BLUE,lw=2.2,zorder=5)
    if len(X2)>1 and np.std(X2)>0:
        ax.plot(x_grid,gaussian_kde(X2)(x_grid),color=ORANGE,lw=2.2,zorder=5)

    ax.set_ylabel(None)
    ax.scatter(np.mean(X1),0,color=BLUE,s=25,zorder=6,clip_on=False)
    ax.scatter(np.mean(X2),0,color=ORANGE,s=25,zorder=6,clip_on=False)

def add_pvalue(ax,X1,X2):
    X1 = X1[np.isfinite(X1)]
    X2 = X2[np.isfinite(X2)]
    
    pvalue = ks_2samp(X1,X2).pvalue
    Kstatistic = ks_2samp(X1,X2).statistic
    print(len(X1),len(X2),pvalue,Kstatistic)
    ax.text(0.97,0.95,f'p = {pvalue:.4f}',
            transform=ax.transAxes,
            ha='right',va='top',
            fontsize=8,
            bbox=dict(boxstyle='round,pad=0.25',
                      facecolor='white',
                      edgecolor='gray',
                      linewidth=0.7,
                      alpha=0.9))

    return pvalue

def rh_bin_width(X1,X2):
    X = np.concatenate([X1,X2])
    bin_width = (np.nanmax(X)-np.nanmin(X))/3.5

    while True:
        edges = np.arange(np.floor(np.nanmin(X)/bin_width)*bin_width,np.ceil(np.nanmax(X)/bin_width)*bin_width+bin_width,bin_width)
        n1 = sum(np.any((X1>=l)&(X1<r)) for l,r in zip(edges[:-1],edges[1:]))
        n2 = sum(np.any((X2>=l)&(X2<r)) for l,r in zip(edges[:-1],edges[1:]))

        if n1>=5 and n2>=5:
            break

        bin_width *= 0.8

    return bin_width

def bootstrap_peak_ci(L1,L2,n_boot=10000,seed=42):
    L1 = np.asarray(L1,dtype=float)
    L2 = np.asarray(L2,dtype=float)

    L1 = L1[np.isfinite(L1)]
    L2 = L2[np.isfinite(L2)]

    pooled = np.concatenate([L1,L2])
    h = np.std(pooled,ddof=1)*len(pooled)**(-1/5)
    grid = np.linspace(pooled.min()-2*h,pooled.max()+2*h,5000)

    def mode(data):
        kde = gaussian_kde(data,bw_method=h/np.std(data,ddof=1))
        return grid[np.argmax(kde(grid))]

    rng = np.random.default_rng(seed)
    differences = np.empty(n_boot)

    for i in range(n_boot):
        A = rng.choice(L1,len(L1),replace=True)
        B = rng.choice(L2,len(L2),replace=True)
        differences[i] = mode(B)-mode(A)

    ci = np.percentile(differences,[2.5,97.5])

    print('\nBootstrap peak displacement')
    print(f'Median displacement = {np.median(differences):.2f}')
    print(f'95% CI = ({ci[0]:.2f}, {ci[1]:.2f})')

    return differences,ci

def subsampling_peak_test(L1,L2,n_sub=35,n_iter=10000,seed=42):
    L1 = np.asarray(L1,dtype=float)
    L2 = np.asarray(L2,dtype=float)

    L1 = L1[np.isfinite(L1)]
    L2 = L2[np.isfinite(L2)]

    n_sub = min(n_sub,len(L1),len(L2))

    print(f'Subsample size used = {n_sub}')

    pooled = np.concatenate([L1,L2])
    h = np.std(pooled,ddof=1)*len(pooled)**(-1/5)
    grid = np.linspace(pooled.min()-2*h,pooled.max()+2*h,5000)

    def mode(data):
        kde = gaussian_kde(data,bw_method=h/np.std(data,ddof=1))
        return grid[np.argmax(kde(grid))]

    rng = np.random.default_rng(seed)
    differences = np.empty(n_iter)

    for i in range(n_iter):
        A = rng.choice(L1,n_sub,replace=False)
        B = rng.choice(L2,n_sub,replace=False)
        differences[i] = mode(B)-mode(A)

    ci = np.percentile(differences,[2.5,97.5])

    print('\nSubsampling peak displacement')
    print(f'Median displacement = {np.median(differences):.2f}')
    print(f'95% interval = ({ci[0]:.2f}, {ci[1]:.2f})')

    return differences,ci

def statistical_significance(X1,X2):
    p_value = ks_2samp(X1,X2).pvalue
    statistic = ks_2samp(X1,X2).statistic
    return 0

fig = plt.figure(figsize=(9,11))

gs = fig.add_gridspec(
    4,3,
    height_ratios=[1.05,0.8,1.05,0.8],
    hspace=0.42,
    wspace=0.30
)

axes_top,axes_mid,axes_bottom,axes_rh = [],[],[],[]

for i,(lon1,lon2,lat1,lat2) in enumerate(regions):

    ax1 = fig.add_subplot(gs[0,i])
    ax2 = fig.add_subplot(gs[1,i],sharex=ax1)
    ax3 = fig.add_subplot(gs[2,i])
    ax4 = fig.add_subplot(gs[3,i],sharex=ax3)

    axes_top.append(ax1)
    axes_mid.append(ax2)
    axes_bottom.append(ax3)
    axes_rh.append(ax4)


    pr1,tc1 = get_events(lon1,lon2,lat1,lat2,np.arange(1979,1990),'tcwv')
    pr2,tc2 = get_events(lon1,lon2,lat1,lat2,np.arange(2010,2021),'tcwv')

    plot_boxes(ax1,tc1,pr1,tc2,pr2,5)
    plot_tcwv_distribution(ax2,tc1,tc2)
    p_tcwv = add_pvalue(ax2,tc1,tc2)


    pr1_rh,rh1 = get_events(lon1,lon2,lat1,lat2,np.arange(1979,1990),'rh')
    pr2_rh,rh2 = get_events(lon1,lon2,lat1,lat2,np.arange(2010,2021),'rh')

    bw = rh_bin_width(rh1,rh2)

    plot_boxes(ax3,rh1,pr1_rh,rh2,pr2_rh,bw,offset=bw*0.15,box_scale=0.3)

    plot_tcwv_distribution(ax4,rh1,rh2)
    p_rh = add_pvalue(ax4,rh1,rh2)

    statistical_significance(rh1,rh2)

    ax1.text(0.02,0.94,f'({chr(97+i)})',transform=ax1.transAxes,ha='left',va='top',fontweight='bold')
    ax2.text(0.02,0.94,f'({chr(100+i)})',transform=ax2.transAxes,ha='left',va='top',fontweight='bold')
    ax3.text(0.02,0.94,f'({chr(103+i)})',transform=ax3.transAxes,ha='left',va='top',fontweight='bold')
    ax4.text(0.02,0.94,f'({chr(106+i)})',transform=ax4.transAxes,ha='left',va='top',fontweight='bold')

    ax1.set_title(region_names[i],fontsize=11,fontweight='bold',pad=10)

    tcwv_min = min(np.nanmin(tc1),np.nanmin(tc2))
    tcwv_max = max(np.nanmax(tc1),np.nanmax(tc2))
    tcwv_pad = 0.25*(tcwv_max-tcwv_min)

    ax1.set_xlim(tcwv_min-tcwv_pad,tcwv_max+tcwv_pad)
    ax1.tick_params(labelbottom=False)

    ax2.set_xlabel('TCWV (kg m$^{-2}$)',labelpad=5)
    ax2.set_yticks([])
    ax2.margins(y=0.12)

    [ax2.spines[s].set_visible(False) for s in ['left','right','top']]

    rh_min = min(np.nanmin(rh1),np.nanmin(rh2))
    rh_max = max(np.nanmax(rh1),np.nanmax(rh2))
    rh_pad = (rh_max-rh_min)*0.25

    ax3.set_xlim(rh_min-rh_pad,rh_max+rh_pad)

    ax3.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax3.ticklabel_format(axis='x',style='sci',scilimits=(0,0),useMathText=True)
    ax3.xaxis.set_major_locator(MaxNLocator(nbins=5))

    ax3.tick_params(labelbottom=False)

    ax4.set_xlabel(r'$RH_{700 \ hPa}$',labelpad=5)
    ax4.set_yticks([])
    ax4.margins(y=0.12)

    [ax4.spines[s].set_visible(False) for s in ['left','right','top']]

    for ax in [ax1,ax3]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y',ls='--',lw=0.6,alpha=0.22)
        ax.set_axisbelow(True)

    if i==0:
        ax1.set_ylabel('Precipitation (mm day$^{-1}$)',labelpad=7)
        ax3.set_ylabel('Precipitation (mm day$^{-1}$)',labelpad=7)
    else:
        ax1.tick_params(labelleft=False)
        ax3.tick_params(labelleft=False)

handles = [
    Line2D([0],[0],color=BLUE,lw=2.2,marker='o',ms=6,label='Early'),
    Line2D([0],[0],color=ORANGE,lw=2.2,marker='o',ms=6,label='Late')
]

fig.legend(
    handles=handles,
    loc='upper center',
    bbox_to_anchor=(0.5,0.995),
    ncol=2,
    frameon=True,
    handlelength=2.2,
    columnspacing=2,
    fontsize=12
)

fig.subplots_adjust(left=0.105,right=0.985,bottom=0.07,top=0.93)

plt.savefig('figure_5.png',dpi=600,bbox_inches='tight')
plt.savefig('figure_5.pdf',bbox_inches='tight')

plt.show()
