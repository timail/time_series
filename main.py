#!/usr/bin/env python3
# UTF-8, *nix

### СКАВРя

### 
#
# Tim Danilov
# timail69@gmail.com
# t.me/timail
#
# SPbSU,
# Mathematics and Mechanics faculty,
# Department of Astronomy.
#
# April, 2018
# 
###


import numpy as np

from numpy.fft import rfft  as Fourier
from numpy.fft import irfft as Fourier_inv


import matplotlib.pyplot as graph


from lib import func

from lib import default_const as data


from lib import fnames


from lib.conf import(
    a_Tukey,
    N_Tukey,

    N_ratio,
)


EXTRA_PLOTS = False


def main():
    
# getting time series and time nodes

    series = func.gen_series()
    nodes = data.nodes


### plotting the time series

    plot_raw_series = graph.figure()
    graph.title('Time series (raw)', fontsize=12)

    graph.xlabel('Time, (sec)')

    graph.plot(
        nodes, 
        series,
        color = 'black',
    )

    plot_raw_series.savefig(fnames.plot('raw_series'))

    graph.clf()


### exclusion of the trend, centering the series 

    series = series - np.mean(series)

    # linear regression
    nodes_mod = np.array([nodes, np.ones(nodes.size)])
    res = np.linalg.lstsq(nodes_mod.T, series, rcond=None)[0]

    linear_trend = res[0]*nodes + res[1]

    series = series - linear_trend
    

### plotting the centred series

    plot_centred_series = graph.figure()
    graph.title('Time series (centred)', fontsize=12)

    graph.xlabel('Time, (sec)')

    graph.plot(
        nodes, 
        series,
        color = 'black',
    )

    plot_centred_series.savefig(fnames.plot('centred_series'))

    graph.clf()


### calculation of the periodogram 

    # adding zeros to series (for FFT)
    N1 = int(2 ** (np.ceil(np.log2(data.N))))
    N2 = N1 * 2

    zeros = np.zeros (N2 - (data.N + 1))
    series_with_zeros = np.concatenate((series, zeros), axis=0)

    # computing freq system
    d_nu = 1 / (N2 * data.dt)
    freq_nodes = [d_nu * j for j in range(N1)]


    X = Fourier(series_with_zeros)
    D = np.array([x.real**2 + x.imag**2 for x in X]) / data.N**2

### estimation of dispersion
    
    sigma_2_null = np.sum(series**2) / (data.N - 1)

    treshold = (sigma_2_null * data.X1) / data.N

### plotting the periodogram and detection treshold

    plot_periodogram = graph.figure()
    graph.title('Periodogram', fontsize=12)

    graph.xlabel('Frequency, (Hz)')

    graph.plot(
        freq_nodes, 
        D,
        color = 'black',
    )

    graph.plot(
        freq_nodes,
        [treshold for i in freq_nodes],
        color = 'black',
        linestyle = '--',
        label = 'detection treshold'
    )

    graph.legend()

    plot_periodogram.savefig(fnames.plot('periodogram'))

    graph.clf()


### computing the correlogram (with FFT)
    
    correlorgam = Fourier_inv(np.array([np.abs(x)**2 for x in X])) / data.N 


### plotting the correlogram
    
    plot_correlogram = graph.figure()
    graph.title('Correlogram', fontsize=12)

    graph.xlabel('Time, (sec)')

    graph.plot(
        nodes[:data.N], 
        correlorgam[:data.N],
        color = 'black',
    )

    plot_correlogram.savefig(fnames.plot('correlogram'))

    graph.clf()


### weighted correlogram

    # Tukey window (parameters in lib/conf)
    W = lambda m: (1 - 2*a_Tukey) + 2*a_Tukey * np.cos(np.pi * m/N_Tukey)

    corr_weighted = np.array([correlorgam[m] * W(m) for m in range(N_Tukey)])

    # adding zeros to fit N2 size
    zeros = np.zeros (N2 - (N_Tukey + 1))
    corr_weighted = np.concatenate((corr_weighted, zeros), axis=0)

    if EXTRA_PLOTS:
        plot_extra = graph.figure()

        graph.suptitle(
            'Cut correlogram', 
            fontsize=12,
        )

        graph.title(
            "Tukey window parameters: "
            "a = " + str(a_Tukey) + ", " + 
            "N* = " + str(N_ratio) +"N",

            fontsize=8,
        )

        graph.xlabel('Time, (sec)')

        graph.plot(
            nodes[:N_Tukey], 
            correlorgam[:N_Tukey],
            color = 'black',
        )

        plot_extra.savefig(fnames.plot('cut_correlogram'))

        graph.clf()


        graph.suptitle(
            'Weighted correlogram', 
            fontsize=12,
        )

        graph.title(
            "Tukey window parameters: "
            "a = " + str(a_Tukey) + ", " + 
            "N* = " + str(N_ratio) +"N",

            fontsize=8,
        )
        graph.xlabel('Time, (sec)')

        graph.plot(
            nodes[:N_Tukey], 
            corr_weighted[:N_Tukey],
            color = 'black',
        )

        plot_extra.savefig(fnames.plot('weighted_correlogram'))

        graph.clf()


### computing smoothed correloram
    
    C_0 = corr_weighted[0]

    D_smooth = np.array([2 * x.real - C_0 for x in Fourier(corr_weighted)])

    D_smooth /= N_Tukey


### plotting smoothed periodogram

    plot_smooth_period = graph.figure()
    
    graph.suptitle(
        'Smoothed periodogram', 
        fontsize=12,
    )

    graph.title(
        "Tukey window parameters: "
        "a = " + str(a_Tukey) + ", " + 
        "N* = " + str(N_ratio) +"N",

        fontsize=8,
    )

    graph.xlabel('Frequency, (Hz)')

    graph.plot(
        freq_nodes, 
        D_smooth,
        color = 'black',
    )


    plot_smooth_period.savefig(fnames.plot('smoothed_periodogram'))

    graph.clf()


if __name__ == '__main__':
    main()
