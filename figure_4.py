import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from scipy.ndimage import label
import cmocean
from matplotlib.patches import Rectangle, FancyArrowPatch
from scipy import stats
import random
from scipy.stats import ks_2samp
from scipy.stats import gaussian_kde

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial'],
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'savefig.dpi': 900,
    'figure.dpi': 150,
    'hatch.linewidth': 0.3
})


def compute_propagation_lengths():
    years = np.arange(1979, 2023)
    results = [[], [], [], [], [], []]

    for y in years:
        path = f'/Eq_Rossby/{y}.er_10_48.nc'

        for idx, subset in enumerate([M, M[:, 0:60], M[:, 100:]]):
            threshold = -1.5 * np.std(subset, axis=0)
            labeled, nclusters = label(subset < threshold)

            for j in range(1, nclusters + 1):
                cols = np.argwhere(labeled == j)[:, 1]
                if cols.size == 0:
                    continue

                length = cols.max() - cols.min()

                if length > 20:
                    results[2 * idx + (y > 1999)].append(length)

    return tuple(np.asarray(x) for x in results)


def plot_hovmoller(fig, ax):
    hov = []

    for year in range(1979, 2021):
        path = f'/Eq_Rossby/{year}.er_10_48.nc'

        try:
            with xr.open_dataset(path) as ds:
                olr_full = ds.groupby('time.dayofyear').mean('time').er.values
                olr_filtered = ds.sel(lon=slice(0, 20)).groupby(
                    'time.dayofyear'
                ).mean(('lon', 'time')).er.values

            olr_filtered = -(olr_filtered - np.mean(olr_filtered)) / np.std(olr_filtered)

            for j in range(10, len(olr_filtered) - 41):
                if olr_filtered[j] > 1 and olr_filtered[j-1] < 1 and olr_filtered[j+1] > 1:
                    hov.append(np.array([olr_full[j+k] for k in range(-40, 41)]))

        except Exception:
            continue

    if hov:
        hov = np.mean(hov, axis=0)
        hov = np.concatenate((hov[:, 72:], hov[:, :72]), axis=1)
    else:
        hov = np.zeros((81, 144))

    times = np.arange(-40, 41)
    lons = np.linspace(-180, 181, 144, endpoint=False)

    im = ax.pcolormesh(
        lons, times, hov,
        cmap=cmocean.cm.amp_r,
        shading='gouraud',
        vmin=-1.5,
        vmax=0,
        rasterized=True
    )

    cbar = fig.colorbar(
        im, ax=ax,
        orientation='horizontal',
        pad=0.08,
        fraction=0.045,
        shrink=0.88,
        extend='both'
    )

    cbar.set_label('OLR (W m$^{-2}$)', fontsize=8, labelpad=3)
    cbar.ax.tick_params(labelsize=7, length=2.5, width=0.7)

    ax.set_xlabel('Longitude (°)', labelpad=4)
    ax.set_ylabel('Day', labelpad=4)
    ax.set_title('(a)', loc='left', fontweight='bold', fontsize=10, pad=6)

    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_yticks(np.arange(-40, 41, 20))
    ax.set_xticklabels([f'{x}°' for x in np.arange(-180, 181, 60)])

    ax.grid(True, linewidth=0.4, alpha=0.3)
    ax.tick_params(direction='out', length=3, width=0.8)
    regions = [
        (-100, 18, 20, 4, 'CENTRAL\nAMERICA'),
        (0, -2, 20, 4, 'WEST\nAFRICA'),
        (70, -12, 20, 4, 'INDIA')
    ]
    
    for lon_start, day_start, lon_w, day_h, label in regions:
        ax.add_patch(
            Rectangle((lon_start, day_start), lon_w, day_h, fill=False, 
                      edgecolor='white', linestyle='--', linewidth=1.2, zorder=6)
        )
        ax.text(lon_start + (lon_w / 2), day_start + day_h + 1, label,
                color='#008080', fontsize=10, fontweight='bold',
                ha='center', va='bottom', zorder=7)

    arrow_lon = 165  
    
    arrow_0_to_neg10 = FancyArrowPatch(
        (arrow_lon, 0), (arrow_lon, -10),
        arrowstyle='<->', mutation_scale=8, color='#333333', lw=1.2, zorder=10
    )
    ax.add_patch(arrow_0_to_neg10)
    
    ax.text(arrow_lon, -5, '10 Days', color='#555555', fontsize=7, fontweight='bold',
            ha='center', va='center', rotation=90, zorder=11,
            bbox=dict(facecolor='white', edgecolor='none', pad=1.0)) 
    
    arrow_0_to_pos20 = FancyArrowPatch(
        (arrow_lon, 0), (arrow_lon, 20),
        arrowstyle='<->', mutation_scale=8, color='#333333', lw=1.2, zorder=10
    )
    ax.add_patch(arrow_0_to_pos20)

    ax.text(arrow_lon, 10, '20 Days', color='#555555', fontsize=7, fontweight='bold',
            ha='center', va='center', rotation=90, zorder=11,
            bbox=dict(facecolor='white', edgecolor='none', pad=1.0))
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def plot_distribution(ax, L1_data, L2_data, title, panel_label):
    L1_data = np.asarray(L1_data, dtype=float) * 2.5
    L2_data = np.asarray(L2_data, dtype=float) * 2.5

    L1_data = L1_data[np.isfinite(L1_data)]
    L2_data = L2_data[np.isfinite(L2_data)]

    colors = ['#CC6677', '#4477AA']

    label1 = f'1979–1999 (N={len(L1_data)})'
    label2 = f'2000–2022 (N={len(L2_data)})'

    combined = np.concatenate((L1_data, L2_data))

    if len(combined) > 0:
        bins = np.histogram_bin_edges(combined, bins=21)
    else:
        bins = np.linspace(0, 1, 22)

    ax.hist(
        L1_data, bins=bins, density=True,
        color=colors[0], alpha=0.22,
        edgecolor=colors[0], linewidth=0.65,
        label=label1
    )

    ax.hist(
        L2_data, bins=bins, density=True,
        color=colors[1], alpha=0.22,
        edgecolor=colors[1], linewidth=0.65,
        label=label2
    )

    if len(L1_data) > 1 and np.std(L1_data) > 0:
        sns.kdeplot(
            x=L1_data, ax=ax,
            color=colors[0], linewidth=1.7,
            fill=False
        )

    if len(L2_data) > 1 and np.std(L2_data) > 0:
        sns.kdeplot(
            x=L2_data, ax=ax,
            color=colors[1], linewidth=1.7,
            fill=False
        )

    ax.set_title(
        f'({panel_label}) {title}',
        loc='left', fontsize=8.5,
        fontweight='bold', pad=7
    )

    ax.set_xlabel('Propagation length (° longitude)', labelpad=4)
    ax.set_ylabel('Probability density', labelpad=4)

    ax.tick_params(
        axis='both', direction='out',
        length=3, width=0.8, pad=3
    )

    ax.grid(
        axis='y', linestyle='--',
        linewidth=0.45, alpha=0.3
    )

    ax.set_axisbelow(True)

    ax.legend(
        loc='upper right',
        frameon=False,
        fontsize=6.8,
        handlelength=1.5,
        handletextpad=0.5,
        borderaxespad=0.5,
        labelspacing=0.4
    )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)


fig = plt.figure(figsize=(7.2, 6.4))

positions = {
    'hovmoller': [0.06, 0.05, 0.27, 0.90],
    'global':    [0.43, 0.55, 0.54, 0.35],
    'ind_1':     [0.43, 0.15, 0.25, 0.29],
    'ind_2':     [0.72, 0.15, 0.25, 0.29]
}

ax_hov = fig.add_axes(positions['hovmoller'])
ax_global = fig.add_axes(positions['global'])
ax_iw = fig.add_axes(positions['ind_1'])
ax_wc = fig.add_axes(positions['ind_2'], sharey=ax_iw)

L1_ful, L2_ful, L1_IW, L2_IW, L1_WC, L2_WC = compute_propagation_lengths()

plot_hovmoller(fig, ax_hov)

plot_distribution(
    ax_global, L1_ful, L2_ful,
    'Global propagation length distribution', 'b'
)

plot_distribution(
    ax_iw, L1_IW, L2_IW,
    'India → West Africa', 'c'
)

plot_distribution(
    ax_wc, L1_WC, L2_WC,
    'West Africa → Central America', 'd'
)

ax_wc.tick_params(axis='y', labelleft=False)
ax_wc.set_ylabel('')

plt.savefig('figure_4.pdf', bbox_inches='tight', dpi=900)
plt.savefig('figure_4.png', bbox_inches='tight', dpi=900)

plt.show()

def bootstrap_peak_ci(L1, L2, n_boot=10000, seed=42):

    L1 = np.asarray(L1, dtype=float)
    L2 = np.asarray(L2, dtype=float)

    L1 = L1[np.isfinite(L1)]
    L2 = L2[np.isfinite(L2)]

    pooled = np.concatenate([L1, L2])
    h = np.std(pooled, ddof=1) * len(pooled)**(-1/5)
    grid = np.linspace(pooled.min()-2*h, pooled.max()+2*h, 5000)

    def mode(data):
        kde = gaussian_kde(data, bw_method=h/np.std(data, ddof=1))
        return grid[np.argmax(kde(grid))]

    rng = np.random.default_rng(seed)
    differences = np.empty(n_boot)

    for i in range(n_boot):
        A = rng.choice(L1, len(L1), replace=True)
        B = rng.choice(L2, len(L2), replace=True)

        differences[i] = mode(B) - mode(A)

    ci = np.percentile(differences, [2.5, 97.5])

    print('\nBootstrap peak displacement')
    print(f'Median displacement = {np.median(differences):.2f}')
    print(f'95% CI = ({ci[0]:.2f}, {ci[1]:.2f})')

    return differences, ci


def subsampling_peak_test(L1, L2, n_sub=40, n_iter=10000, seed=42):

    L1 = np.asarray(L1, dtype=float)
    L2 = np.asarray(L2, dtype=float)

    L1 = L1[np.isfinite(L1)]
    L2 = L2[np.isfinite(L2)]

    pooled = np.concatenate([L1, L2])
    h = np.std(pooled, ddof=1) * len(pooled)**(-1/5)

    grid = np.linspace(pooled.min()-2*h, pooled.max()+2*h, 5000)

    def mode(data):
        kde = gaussian_kde(data, bw_method=h/np.std(data, ddof=1))
        return grid[np.argmax(kde(grid))]

    rng = np.random.default_rng(seed)
    differences = np.empty(n_iter)

    for i in range(n_iter):

        A = rng.choice(L1, n_sub, replace=False)
        B = rng.choice(L2, n_sub, replace=False)

        differences[i] = mode(B) - mode(A)

    ci = np.percentile(differences, [2.5, 97.5])

    print('\nSubsampling peak displacement')
    print(f'Median displacement = {np.median(differences):.2f}')
    print(f'95% interval = ({ci[0]:.2f}, {ci[1]:.2f})')

    return differences, ci


def statistical_test():
    L1_ful, L2_ful, L1_IW, L2_IW, L1_WC, L2_WC = compute_propagation_lengths()
    L1_ful=L1_ful*2.5
    L2_ful=L2_ful*2.5
    L1_IW=L1_IW*2.5
    L2_IW=L2_IW*2.5
    L1_WC=L1_WC*2.5
    L2_WC=L2_WC*2.5
    groups = {
        'Global': (L1_ful, L2_ful),
        'India → West Africa': (L1_IW, L2_IW),
        'West Africa → Central America': (L1_WC, L2_WC)
    }

    for name, (L1, L2) in groups.items():
        L1 = np.asarray(L1, dtype=float)
        L2 = np.asarray(L2, dtype=float)

        L1 = L1[np.isfinite(L1)]
        L2 = L2[np.isfinite(L2)]

        stat, pvalue = ks_2samp(L1, L2, alternative='greater')

        print(f'\n{name}')
        print(f'N (Early) = {len(L1)}, N (Late) = {len(L2)}')
        print(f'Mean (Early, Late) = {np.mean(L1):.2f}, {np.mean(L2):.2f}')
        print(f'Median (Early, Late) = {np.median(L1):.2f}, {np.median(L2):.2f}')
        print(f'Mean difference (Late - Early) = {np.mean(L2) - np.mean(L1):.2f}')
        print(f'Median difference (Late - Early) = {np.median(L2) - np.median(L1):.2f}')
        print(f'KS statistic = {stat:.4f}')
        print(f'p-value = {pvalue:.5f}')
        print('Significant at 5%:', pvalue < 0.05)
    
    L1 = np.asarray(L1_WC, dtype=float)
    L2 = np.asarray(L2_WC, dtype=float)

    L1 = L1[np.isfinite(L1)]
    L2 = L2[np.isfinite(L2)]

    boot_diff, boot_ci = bootstrap_peak_ci(
        L1, L2, n_boot=10000, seed=42
    )

    sub_diff, sub_ci = subsampling_peak_test(
        L1, L2, n_sub=40, n_iter=10000, seed=42
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    results = [
        (boot_diff, boot_ci, 'Bootstrap'),
        (sub_diff, sub_ci, 'Subsampling')
    ]

    for i, (differences, ci, name) in enumerate(results):

        ax = axes[i]

        median = np.median(differences)

        ax.hist(differences,bins=45,density=True,color='#4393C3',alpha=0.75,edgecolor='black',linewidth=0.6)

        ax.axvline(
            0, color='black', linestyle='--',
            linewidth=1.8, label='No difference in Propagation Lengths'
        )

        ax.axvline(
            median, color='#B2182B',
            linewidth=2, label='Median'
        )

        ax.set_title(name, pad=12, fontweight='bold')

        ax.set_xlabel('Peak displacement (degrees)\n(Late − Early)')
        ax.set_ylabel('Probability density')

        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend(loc='upper right', frameon=False)

    fig.suptitle(
        'West Africa–Central America: Peak displacement',
        fontsize=15, fontweight='bold', y=1.03
    )
    fig.savefig('WAM_CAM_statistical_Test.png',dpi=600, bbox_inches='tight')
    fig.savefig('WAM_CAM_statistical_Test.pdf',bbox_inches='tight')
    fig.tight_layout()

    plt.show()

    return 0

statistical_test()