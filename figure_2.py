import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import cmaps
from scipy.signal import detrend
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

year = np.arange(1975, 2025)
regions = ['nams', 'waf', 'ismr', 'mc']
region_labels = {'nams': 'CAM', 'waf': 'WAM', 'ismr': 'ISM', 'mc': 'MC'}


def regional_mean(ds, boxes, lat='latitude', lon='longitude'):
    return {r: ds.sel({lat: boxes[r][0], lon: boxes[r][1]}).mean((lat, lon)) for r in boxes}


def seasonal_mean(data, variable):
    tmp = 0
    for j in range(6, 9):
        tmp += data.groupby('time.month')[j].mean('time')[variable].values
    return tmp / 4


def detrend_epochs(data):
    return np.concatenate([
        np.mean(data[:25]) + detrend(data[:25]),
        np.mean(data[25:]) + detrend(data[25:])
    ])


def print_trends(data, label):
    print(label)
    for r in regions:
        print(region_labels[r])
        print(np.polyfit(range(25), data[r][:25], 1)[0])
        print(np.polyfit(range(25), data[r][25:], 1)[0])


def precip():
    path = 'cru_ts4.09.1901.2024.pre.dat.nc'
    ds = xr.open_dataset(path)

    boxes = {
        'nams': (slice(10, 25), slice(-100, -80)),
        'waf': (slice(10, 15), slice(-20, 20)),
        'ismr': (slice(8, 28), slice(70, 90)),
        'mc': (slice(-10, 10), slice(95, 150))
    }

    data = regional_mean(ds, boxes, 'lat', 'lon')
    result = {r: [] for r in regions}

    for i in year:
        for r in regions:
            result[r].append(seasonal_mean(data[r].sel(time=str(i)), 'pre'))

    result = {r: detrend_epochs(np.array(result[r]) / 30) for r in regions}
    ds.close()

    return result


def evap():
    boxes = {
        'nams': (slice(25, 10), slice(-100, -80)),
        'waf': (slice(15, 10), slice(-20, 20)),
        'ismr': (slice(28, 8), slice(70, 90)),
        'mc': (slice(10, -10), slice(95, 150))
    }

    result = {r: [] for r in regions}

    for i in year:
        path = 'evap.mnmean.' + str(i) + '.nc'
        ds = xr.open_dataset(path)
        data = regional_mean(ds, boxes)

        for r in regions:
            result[r].append(seasonal_mean(data[r], 'e'))

        ds.close()

    result = {r: np.array(result[r]) for r in regions}
    print_trends(result, 'Evaporation')
    result = {r: detrend_epochs(result[r]) for r in regions}

    return result


def Qnet():
    boxes = {
        'nams': (slice(25, 10), slice(260, 280)),
        'ismr': (slice(28, 8), slice(70, 90)),
        'mc': (slice(10, -10), slice(95, 150))
    }

    result = {r: [] for r in regions}

    for i in year:
        path = 'nei.' + str(i) + '.nc'
        ds = xr.open_dataset(path)
        data = regional_mean(ds, boxes)

        waf1 = ds.sel(latitude=slice(15, 10), longitude=slice(0, 20)).mean(('latitude', 'longitude'))
        waf2 = ds.sel(latitude=slice(15, 10), longitude=slice(340, 360)).mean(('latitude', 'longitude'))
        data['waf'] = (waf1 + waf2) / 2

        for r in regions:
            result[r].append(seasonal_mean(data[r], 'nei'))

        ds.close()

    result = {r: np.array(result[r]) for r in regions}
    print_trends(result, 'Q_net')
    result = {r: detrend_epochs(result[r]) for r in regions}

    return result


def tpwv():
    boxes = {
        'nams': (slice(25, 10), slice(-100, -80)),
        'waf': (slice(15, 10), slice(-20, 20)),
        'ismr': (slice(28, 8), slice(70, 90)),
        'mc': (slice(10, -10), slice(95, 150))
    }

    result = {r: [] for r in regions}

    for i in year:
        path = 'tcwv.mnmean.' + str(i) + '.nc'
        ds = xr.open_dataset(path)
        data = regional_mean(ds, boxes)

        for r in regions:
            result[r].append(seasonal_mean(data[r], 'tcwv'))

        ds.close()

    result = {r: np.array(result[r]) for r in regions}
    print_trends(result, 'Moisture')
    result = {r: detrend_epochs(result[r]) for r in regions}

    return result


def C(P, E, Q, W):
    result = {}

    for r in regions:
        tmp_Pe = P[r] + 1000 * E[r]
        result[r] = W[r] * (Q[r] / (28.94 * tmp_Pe) + 1)

    print_trends(result, 'Stability, C')
    result = {r: detrend_epochs(result[r]) for r in regions}

    return result


def param_precipitation(E, Q, C, W):
    P_E = {}
    P_Q = {}
    P_C = {}
    P_W = {}

    for r in regions:
        P_E[r] = E[r] + np.mean(Q[r]) / ((np.mean(C[r]) / np.mean(W[r])) - 1)
        P_Q[r] = np.mean(E[r]) + Q[r] / ((np.mean(C[r]) / np.mean(W[r])) - 1)
        P_C[r] = np.mean(E[r]) + np.mean(Q[r]) / ((C[r] / np.mean(W[r])) - 1)
        P_W[r] = np.mean(E[r]) + np.mean(Q[r]) / ((np.mean(C[r]) / W[r]) - 1)

    return {'P_E': P_E, 'P_Q': P_Q, 'P_C': P_C, 'P_W': P_W}


def correlation_matrix(data, start, end):
    selected = {r: data[r][start:end] for r in regions}
    frame = pd.DataFrame({
        region_labels[r]: selected[r] for r in regions
    })
    return frame.corr().round(2)


def Early_Late_Plot():
    P = precip()
    E = evap()
    Q = Qnet()
    W = tpwv()
    C_data = C(P, E, Q, W)

    parametric = param_precipitation(E, Q, C_data, W)

    components = ['P_E', 'P_Q', 'P_C', 'P_W']
    heatmaps_data = []

    for component in components:
        heatmaps_data.append(correlation_matrix(parametric[component], 0, 25))
        heatmaps_data.append(correlation_matrix(parametric[component], 25, 50))

    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial']
    plt.rcParams['font.size'] = 7
    plt.rcParams['font.weight'] = 'normal'

    fig = plt.figure(figsize=(5.0, 7.0))

    panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']

    positions = [
        (0.15, 0.72, 0.26, 0.26),
        (0.48, 0.72, 0.26, 0.26),
        (0.15, 0.49, 0.26, 0.26),
        (0.48, 0.49, 0.26, 0.26),
        (0.15, 0.26, 0.26, 0.26),
        (0.48, 0.26, 0.26, 0.26),
        (0.15, 0.03, 0.26, 0.26),
        (0.48, 0.03, 0.26, 0.26)
    ]

    cbar_ax = fig.add_axes([0.80, 0.25, 0.03, 0.5])
    first_heatmap = None

    for i, (left, bottom, width, height) in enumerate(positions):
        ax = fig.add_axes([left, bottom, width, height])
        mask = np.triu(np.ones_like(heatmaps_data[i], dtype=bool), k=1)

        hm = sns.heatmap(heatmaps_data[i], annot=True, cmap=cmaps.GMT_panoply, fmt=".2f", linewidths=0.5, cbar=None, vmin=-0.75, vmax=0.75, mask=mask, annot_kws={"size": 8, "fontweight": "bold"}, ax=ax, square=True)

        if first_heatmap is None:
            first_heatmap = hm

        ax.tick_params(labelsize=7)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.text(-0.15, 1.10, panel_labels[i], transform=ax.transAxes, fontsize=10, fontweight='bold', fontfamily='sans-serif', verticalalignment='top')

        if i % 2 == 1:
            ax.set_yticklabels([])

        if i < 6:
            ax.set_xticklabels([])

        if i == 0:
            ax.set_title('Early (1975–1999)', fontsize=8)
            ax.set_ylabel(r'$P_E$', fontsize=8)
        if i == 1:
            ax.set_title('Late (2000–2024)', fontsize=8)
        if i == 2:
            ax.set_ylabel(r'$P_Q$', fontsize=8)
        if i == 4:
            ax.set_ylabel(r'$P_C$', fontsize=8)
        if i == 6:
            ax.set_ylabel(r'$P_W$', fontsize=8)

    norm = Normalize(vmin=-0.75, vmax=0.75)
    cmap = first_heatmap.collections[0].get_cmap()
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical', extend='both')
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label('Correlation', rotation=90, fontsize=10)
    plt.show()
    
    return 0