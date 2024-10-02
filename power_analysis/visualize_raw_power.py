import argparse
import os
import pickle
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats

'''
visualize raw power for each state's median FFT.
requires FFT result from run_FFT_for_states.py
'''

parser = argparse.ArgumentParser()
parser.add_argument("--median_FFT_dir", type=str, default='save_var/mouse_median_dict.pickle', 
                        help = 'copy paste directory for median FFT result from run_FFT_for_states.py')
parser.add_argument("--img_save_dir", default='save_img_FFT',
                        help = 'dir for saving raw power visualization')
parser.add_argument("--a_label", type=str, default='A', help = 'label for A group')
parser.add_argument("--b_label", type=str, default='B', help = 'label for B group')
args = parser.parse_args()

fft_median_dir = args.median_FFT_dir
save_dir = args.img_save_dir
A_label = args.a_label
B_label = args.b_label

if not os.path.exists(save_dir):
  Path(save_dir).mkdir(parents=True)

with open(fft_median_dir, 'rb') as handle:
    whole_mouse_dict = pickle.load(handle)

def plot_time_series_data(whole_mouse_dict, phase):
    A_one = []
    A_two = []
    A_three = []

    B_one = []
    B_two = []
    B_three = []
    if phase == 'dark':
        for key, value in whole_mouse_dict.items():
            if 'A' in key:
                A_one.append(whole_mouse_dict[key][0])
                A_two.append(whole_mouse_dict[key][1])
                A_three.append(whole_mouse_dict[key][2])
            else:
                B_one.append(whole_mouse_dict[key][0])
                B_two.append(whole_mouse_dict[key][1])
                B_three.append(whole_mouse_dict[key][2])
    else: # light phase
        for key, value in whole_mouse_dict.items():
            if 'A' in key:
                A_one.append(whole_mouse_dict[key][3])
                A_two.append(whole_mouse_dict[key][4])
                A_three.append(whole_mouse_dict[key][5])
            else:
                B_one.append(whole_mouse_dict[key][3])
                B_two.append(whole_mouse_dict[key][4])
                B_three.append(whole_mouse_dict[key][5])

    def visualize(a_list, b_list, phase = phase, _type = None, scale = 100):
        a_mean = np.mean(a_list, axis=0)
        b_mean = np.mean(b_list, axis=0)
        a_sem = stats.sem(a_list)
        b_sem = stats.sem(b_list)

        fig = plt.figure()
        
        annot_freq = 10
        FFTsampling_rate = 250
        FFT_N = annot_freq * FFTsampling_rate # 2500
        FFT_dt = 1/FFTsampling_rate # 0.004
        FFT_T = FFT_N * FFT_dt # 10
        df = 1 / FFT_T # Determine frequency resolution
        fNQ = 1 / FFT_dt / 2 # Determine Nyquist frequency
        faxis = np.arange(0.5,fNQ,df) # Construct frequency axis
        
        plt.xlim([0.5, 20])
        plt.plot(faxis, a_mean, ls = '-', color = 'green', lw = 1.2, alpha = 0.5)
        plt.fill_between(faxis, a_mean-a_sem, a_mean+a_sem,
            alpha=0.2, edgecolor='#000000', facecolor='green',
            linewidth=0, label = A_label)

        plt.plot(faxis, b_mean, ls = '-', color = 'blue', lw = 1.2, alpha = 0.5)
        plt.fill_between(faxis, b_mean-b_sem, b_mean+b_sem,
            alpha=0.2, edgecolor='#000000', facecolor='blue',
            linewidth=0, label = B_label)

        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Power [$\mu V^2$/Hz]')
        if _type == 'Wake':
            plt.title(f'Wake EEG Spectra ({phase} phase)')
        elif _type == 'REM':
            plt.title(f'REM EEG Spectra ({phase} phase)')
        elif _type == 'NREM':
            plt.title(f'NREM EEG Spectra ({phase} phase)')
        else:
            raise ('Enter proper state information')
        plt.ylim(bottom=0)
        plt.xticks([0.5,5,10,15,20], [0.5,5,10,15,20])
        plt.legend()
        fig.savefig(f'{save_dir}/{_type} EEG Spectra ({phase} phase).png', dpi=fig.dpi)
        plt.show()
    
    visualize(A_one, B_one, _type = 'Wake')
    visualize(A_two, B_two, _type = 'NREM')
    visualize(A_three, B_three, _type = 'REM')

plot_time_series_data(whole_mouse_dict, 'dark')
plot_time_series_data(whole_mouse_dict, 'light')
