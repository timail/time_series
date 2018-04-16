#!/usr/bin/env python3
# UTF-8, *nix

import numpy as np

from numpy.fft import rfft  as Fourier
from numpy.fft import irfft as Fourier_inv


import matplotlib.pyplot as graph


from lib import func

from lib import default_const as data


import fnames


def gen_number():
    n = 0
    while True:
        yield n
        yield n + 1 



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
    
    # adding zeros to series
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
    
    c_m = Fourier_inv(np.array([np.abs(x)**2 for x in X])) / data.N 
    
### plotting the correlogram
    
    plot_correlogram = graph.figure()
    graph.title('Correlogram', fontsize=12)

    graph.xlabel('Time, (sec)')

    graph.plot(
        nodes, 
        c_m[:230],
        color = 'black',
    )#!/usr/bin/env python3
# UTF-8, *nix

import numpy as np

from numpy.fft import rfft  as Fourier
from numpy.fft import irfft as Fourier_inv


import matplotlib.pyplot as graph


from lib import func

from lib import default_const as data


import fnames



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
    
    # adding zeros to series
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
    
    c_m = Fourier_inv(np.array([np.abs(x)**2 for x in X])) / data.N 
    
### plotting the correlogram
    
    plot_correlogram = graph.figure()
    graph.title('Correlogram', fontsize=12)

    graph.xlabel('Time, (sec)')

    graph.plot(
        nodes[:data.N], 
        c_m[:data.N],
        color = 'black',
    )

    plot_correlogram.savefig(fnames.plot('correlogram'))

    graph.clf()

### weighted correlogram
    
    N_Tukey = int(0.5 * data.N)
    a = 0.25 ## Tukey window parameter

    W = lambda m: (1 - 2*a) + 2*a * np.cos(np.pi * m/N_Tukey)

    c_m_weighted = np.array([c_m[m] * W(m) for m in range(N_Tukey)])

    zeros = np.zeros (N2 - (N_Tukey + 1))
    C_m = np.concatenate((c_m_weighted, zeros), axis=0)

### computing smoothed correloram
    
    FFT_corr_w = Fourier(C_m)

    C_0 = C_m[0]

    D_j = np.array([2 * x.real - C_0 for x in FFT_corr_w])

    D_j /= N_Tukey


### plotting smoothed correlogram

    plot_smooth_period = graph.figure()
    graph.title('Smoothed periodogram', fontsize=12)

    graph.xlabel('Frequency, (Hz)')

    graph.plot(
        freq_nodes, 
        D_j,
        color = 'black',
    )

    plot_smooth_period.savefig(fnames.plot('smoothed_periodogram'))

    graph.clf()



if __name__ == '__main__':
    main()
