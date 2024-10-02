import h5py
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, rfft
from scipy.signal import detrend

'''Perform Fast Fourier Transform (FFT) analysis on EEGs or LFPs'''
'''Adapted from Chapter 3 and 4 in "Case Studies in Neural Data Analysis", by Mark Kramer and Uri T. Eden'''

def get_FFT_per_mouse(mouse_dir, dsf):
    file = mouse_dir
    EEG = np.array( h5py.File(file,'r').get('EEG1'))[0].reshape(-1)    

    x = EEG # Relabel the data variable
    sampling_rate = int(len(x)/86400) # 60*60*24
    annot_freq = 10 # state annotation every 10 secs

    ''' down sample '''
    ds_x = x[0::dsf]
    FFTsampling_rate = sampling_rate/dsf
    FFT_N = annot_freq * FFTsampling_rate
    FFT_dt = 1/FFTsampling_rate
    FFT_T = FFT_N * FFT_dt

    '''loop over downsampled data, do FFT'''
    fft_result =[]
    for i in range(0, len(ds_x), int(FFT_N)):
        x_window = ds_x[i:int(i+FFT_N)]
        assert len(x_window) == FFT_N, 'wrong number of data point'
        x_window  = np.hanning(FFT_N) * x_window
        x_window = detrend(x_window)
        xf = fft(x_window - x_window.mean()) # Compute Fourier transform of x
        Sxx = 2 * FFT_dt ** 2 / FFT_T * (xf * xf.conj()) # Compute spectrum
        Sxx = Sxx[:int(len(x_window) / 2)] # Ignore negative frequencies
        fft_result.append(Sxx)

    return fft_result # list with 8640 elements, each element length is 1250 (Nyquist, 10000 down sample 2500)
