import argparse
import os
import pickle
import numpy as np
from scipy import stats
from pathlib import Path
import matplotlib.pyplot as plt

'''
visualize *normalized* power for each state's median FFT.
requires FFT result from run_FFT_for_states.py
'''

parser = argparse.ArgumentParser()
parser.add_argument("--median_FFT_dir", type=str, default='save_var/mouse_median_dict.pickle', 
                        help = 'copy paste directory for median FFT result from run_FFT_for_states.py')
parser.add_argument("--img_save_dir", default='save_imgs/FFT',
                        help = 'dir for saving normalized power visualization')
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

def norm_FFT(_list):
    cut_list = _list[5:501]
    denom = sum(cut_list)
    norm_list = [item/denom*100 for item in cut_list]
    return norm_list

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
                A_one.append(norm_FFT(whole_mouse_dict[key][0]))
                A_two.append(norm_FFT(whole_mouse_dict[key][1]))
                A_three.append(norm_FFT(whole_mouse_dict[key][2]))
            else:
                B_one.append(norm_FFT(whole_mouse_dict[key][0]))
                B_two.append(norm_FFT(whole_mouse_dict[key][1]))
                B_three.append(norm_FFT(whole_mouse_dict[key][2]))
    else: # light phase
        for key, value in whole_mouse_dict.items():
            if 'A' in key:
                A_one.append(norm_FFT(whole_mouse_dict[key][3]))
                A_two.append(norm_FFT(whole_mouse_dict[key][4]))
                A_three.append(norm_FFT(whole_mouse_dict[key][5]))
            else:
                B_one.append(norm_FFT(whole_mouse_dict[key][3]))
                B_two.append(norm_FFT(whole_mouse_dict[key][4]))
                B_three.append(norm_FFT(whole_mouse_dict[key][5]))

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
        df = 1 / FFT_T # Determine frequency resolution, 0.1
        fNQ = 1 / FFT_dt / 2 # Determine Nyquist frequency, #125
        faxis = np.arange(0.5,50.1,df) # Construct frequency axis # 496 data point
        
        plt.xlim([0.5, 20])
        plt.ylim([0, 2.5])
        plt.plot(faxis, a_mean, ls = '-', color = 'green', lw = 1.2, alpha = 0.5)
        plt.fill_between(faxis, a_mean-a_sem, a_mean+a_sem,
            alpha=0.2, edgecolor='#000000', facecolor='green',
            linewidth=0, label = A_label)

        plt.plot(faxis, b_mean, ls = '-', color = 'blue', lw = 1.2, alpha = 0.5)
        plt.fill_between(faxis, b_mean-b_sem, b_mean+b_sem,
            alpha=0.2, edgecolor='#000000', facecolor='blue',
            linewidth=0, label = B_label)

        plt.xlabel('Frequency [Hz]')
        plt.ylabel(r'% of Total Power')
        if _type == 'Wake':
            plt.title(f'Wake EEG Spectra ({phase} phase)')
        elif _type == 'REM':
            plt.title(f'REM EEG Spectra ({phase} phase)')
        elif _type == 'NREM':
            plt.title(f'NREM EEG Spectra ({phase} phase)')
        else:
            raise ('Enter proper state information')
        plt.ylim(bottom=0)
        plt.xticks([5,10,15,20], [5,10,15,20])
        plt.legend()
        fig.savefig(f'{save_dir}/normalized {_type} EEG Spectra ({phase} phase).png', dpi=fig.dpi)
        plt.show()

    
    visualize(A_one, B_one, _type = 'Wake')
    visualize(A_two, B_two, _type = 'NREM')
    visualize(A_three, B_three, _type = 'REM')

plot_time_series_data(whole_mouse_dict, 'dark')
plot_time_series_data(whole_mouse_dict, 'light')
