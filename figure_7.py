import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib.ticker import ScalarFormatter, MaxNLocator
from scipy.signal import detrend
from scipy.stats import ks_2samp

sns.set(style="whitegrid", context="talk", font_scale=1.1)
mpl.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Helvetica', 'Arial'],
    'font.size': 7, 'axes.labelsize': 7, 'axes.titlesize': 7,
    'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'legend.fontsize': 6,
    'figure.dpi': 200, 'savefig.dpi': 900,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.linewidth': 0.6,
    'xtick.major.size': 2.0, 'ytick.major.size': 2.0,
    'xtick.minor.size': 1.0, 'ytick.minor.size': 1.0,
    'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
    'xtick.major.pad': 1.5, 'ytick.major.pad': 1.5,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'axes.labelpad': 2,
})

START_BP, HOLO_BP = 22000, 11000
DEC_PER_CHUNK, YR_PER_CHUNK = 10, 100

COL_FULL = {'Low': '#1b9e77', 'Mid': '#d95f02', 'High': '#7570b3'}
COL_HOLO = {'Low': '#56B4E9', 'Mid': '#782167', 'High': '#002244'}
GREY = '0.45'

arr = np.load('/precl_arrays.npz', allow_pickle=True)
pr_ind, pr_afc, pr_ams = arr['IND'], arr['AFC'], arr['AMS']

ds = xr.open_dataset('/TMQ/trace.01-36.22000BP.cam2.TMQ.22000BP_decavgJJA_400BCE.nc')
TC = (ds.sel(lat=slice(5, 25), lon=slice(0, 150)).mean(('lat', 'lon')).TMQ.values
      + ds.sel(lat=slice(5, 25), lon=slice(220, 360)).mean(('lat', 'lon')).TMQ.values) / 2
TC = TC[:len(TC) - 4]

path_sp = '/TraceK/T/static_stability.nc'
dsS = xr.open_dataset(path_sp)
S = (dsS.sel(lat=slice(10, 20), lon=slice(0, 150),   lev=slice(600, 500)).mean(('lat', 'lon', 'lev')).s.values
   + dsS.sel(lat=slice(10, 20), lon=slice(220, 360), lev=slice(600, 500)).mean(('lat', 'lon', 'lev')).s.values) / 2
S = S[:len(TC)]
print(len(TC), len(S))

def analyse(start_bp):
    i0_dec   = (START_BP - start_bp) // 10
    i0_chunk = i0_dec // DEC_PER_CHUNK
    TC_s  = TC[i0_dec:]
    S_s   = S[i0_dec:]
    age   = (START_BP - 10 * np.arange(i0_dec, i0_dec + len(TC_s))) / 1000.0   # ka BP

    n_pr = (min(len(pr_ind), len(pr_afc), len(pr_ams)) // YR_PER_CHUNK) - i0_chunk
    n    = min(len(TC_s) // DEC_PER_CHUNK, n_pr)

    TC_chunk = TC_s[:n * DEC_PER_CHUNK].reshape(n, DEC_PER_CHUNK).mean(axis=1)
    lo, hi = np.percentile(TC_chunk, [33.3, 66.7])
    cls_chunk = np.where(TC_chunk <= lo, 0, np.where(TC_chunk <= hi, 1, 2))

    cls = np.full(len(TC_s), -1)
    cls[:n * DEC_PER_CHUNK] = np.repeat(cls_chunk, DEC_PER_CHUNK)

    def corr(ids):
        out = []
        for k in ids:
            g = k + i0_chunk
            sl = slice(YR_PER_CHUNK * g, YR_PER_CHUNK * (g + 1))
            a, b, c = detrend(pr_ind[sl]), detrend(pr_afc[sl]), detrend(pr_ams[sl])
            out.append(max(np.corrcoef(a, b)[0, 1], np.corrcoef(a, c)[0, 1],
                           np.corrcoef(b, c)[0, 1]))
        return np.array(out)

    groups = {name: corr(np.where(cls_chunk == c)[0]) for c, name in enumerate(['Low', 'Mid', 'High'])}
    pvals = {(a, b): ks_2samp(groups[a], groups[b]).pvalue
             for a, b in [('Low', 'Mid'), ('Low', 'High'), ('Mid', 'High')]}
    print(f"start {start_bp} BP: chunks={n}, n(low/mid/high)="
          f"{len(groups['Low'])}/{len(groups['Mid'])}/{len(groups['High'])}, p={pvals}")
    return dict(TC=TC_s, S=S_s, age=age, cls=cls, groups=groups, pvals=pvals)

full = analyse(START_BP)
holo = analyse(HOLO_BP)

def fmt_p(p):
    return 'p < 0.01' if p < 0.01 else f'p = {p:.2f}'

def pvalue_box(ax, pvals, colors, fig, x0=0.05, y_top=0.96, line_h=0.082, w=0.60):
    h = 3 * line_h + 0.04
    ax.add_patch(Rectangle((x0 - 0.03, y_top - h + 0.01), w, h, transform=ax.transAxes,
                           facecolor='white', edgecolor='0.3', linewidth=0.4, zorder=5))
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    y = y_top - 0.035
    for (a, b), p in pvals.items():
        x = x0
        for txt, col, wt in [(a, colors[a], 'bold'), ('–', 'k', 'normal'),
                             (b, colors[b], 'bold'), (fmt_p(p), 'k', 'bold' if p < 0.05 else 'normal')]:
            t = ax.text(x, y, txt, color=col, fontsize=6, fontweight=wt,
                        transform=ax.transAxes, va='center', ha='left', zorder=6)
            bb = t.get_window_extent(rend)
            x += bb.width / ax.bbox.width + 0.025
        y -= line_h

def draw_row(ax_ts, ax_pdf, res, colors, lab_ts, lab_pdf, xlim_ts, s_ylim=None):
    for c, name in enumerate(['Low', 'Mid', 'High']):
        ax_ts.plot(res['age'], np.where(res['cls'] == c, res['TC'], np.nan),
                   color=colors[name], lw=0.9, label=f'{name} moisture')

    ax_s = ax_ts.twinx()
    ax_s.plot(res['age'], res['S'], color='#B2BEB5', lw=0.7, alpha=0.9, label='Stability')
    if s_ylim is not None:
        ax_s.set_ylim(*s_ylim)
    ax_s.set_ylabel(r'Static stability (K/hPa)', color='#36454F', labelpad=2)  
    ax_s.tick_params(axis='y', colors='#36454F')
    ax_s.grid(False)
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))               
    ax_s.yaxis.set_major_formatter(fmt)
    ax_s.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax_s.yaxis.get_offset_text().set_fontsize(6)
    ax_s.yaxis.get_offset_text().set_color('#36454F')
    ax_s.yaxis.offsetText.set_x(1.0)           
    for s in ax_s.spines.values():
        s.set_visible(True); s.set_linewidth(0.6)
    ax_ts.set_zorder(ax_s.get_zorder() + 1)
    ax_ts.patch.set_visible(False)

    ax_ts.set_xlim(*xlim_ts)
    ax_ts.set_xlabel('Time (ka BP)')
    ax_ts.set_ylabel(r'TCWV (kg m$^{-2}$)')
    ax_ts.set_title(lab_ts, fontsize=9, fontweight='bold', loc='left')
    ax_ts.yaxis.set_major_locator(MaxNLocator(nbins=5))

    h1, l1 = ax_ts.get_legend_handles_labels()
    h2, l2 = ax_s.get_legend_handles_labels()
    ax_ts.legend(h1 + h2, l1 + l2, frameon=True, loc='lower left', ncol=2,
                 handlelength=1.3, columnspacing=0.8, labelspacing=0.25,
                 borderpad=0.35, fontsize=5.5)

    allv = np.concatenate(list(res['groups'].values()))
    bins = np.histogram_bin_edges(allv, bins=21)
    for name in ['Low', 'Mid', 'High']:
        d = res['groups'][name]
        sns.histplot(d, bins=bins, stat='density', color=colors[name], alpha=0.25,
                     edgecolor='#CFCFCF', linewidth=0.3, ax=ax_pdf)
        sns.kdeplot(d, color=colors[name], lw=1.5, ax=ax_pdf)
    ax_pdf.set_xlim(-0.3, 0.6)
    ax_pdf.set_ylim(0, ax_pdf.get_ylim()[1] * 1.5)
    ax_pdf.set_xlabel('Maximum correlation')
    ax_pdf.set_ylabel('Density')
    ax_pdf.set_title(lab_pdf, fontsize=9, fontweight='bold', loc='left')
    ax_pdf.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax_pdf.yaxis.set_major_locator(MaxNLocator(nbins=5))

fig = plt.figure(figsize=(7.2, 5.6))
gs = fig.add_gridspec(2, 2, width_ratios=[1.9, 1], hspace=0.40, wspace=0.40,
                      left=0.075, right=0.985, top=0.955, bottom=0.075)
axs = [[fig.add_subplot(gs[i, j]) for j in range(2)] for i in range(2)]

draw_row(axs[0][0], axs[0][1], full, COL_FULL, '(a)', '(b)', (full['age'].max(), full['age'].min()),  s_ylim=(4.5e-2, 6.6e-2))
draw_row(axs[1][0], axs[1][1], holo, COL_HOLO, '(c)', '(d)', (11, holo['age'].min()),  s_ylim=(5.9e-2, 6.45e-2))

for ax in np.ravel(axs):
    ax.grid(lw=0.2, ls='--', alpha=0.25, color='k')
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(0.6)
    leg = ax.get_legend()
    if leg: leg.get_frame().set_linewidth(0.4)

pvalue_box(axs[0][1], full['pvals'], COL_FULL, fig)
pvalue_box(axs[1][1], holo['pvals'], COL_HOLO, fig)
plt.show()