import argparse
import pickle
import os
import FFT
import statistics
import numpy as np
import read_annot_get_state as annot
from scipy import stats
from tqdm import tqdm
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--annot_dir", type=str, default=None, 
                        required=True, help = 'copy paste directory for sleep state annotation files')
parser.add_argument("--trace_dir", type=str, default=None, 
                        required=True, help = 'copy paste directory for eeg recording files')
parser.add_argument("--save_dir",  type=str, default='save_var',
                        help = 'directory to save median FFT for each mouse')
parser.add_argument("--verbose", default = True, help = 'If true, FFT progress bar will be shown')
parser.add_argument("--down_sampling_rate", default=4, help = 'down sampling rate for FFT')
args = parser.parse_args()

trace_base_dir = args.trace_dir
annot_base_dir = args.annot_dir
pickle_save_dir = args.save_dir
dsf = args.down_sampling_rate
verbose = args.verbose
if not os.path.exists(pickle_save_dir):
  Path(pickle_save_dir).mkdir(parents=True)


def get_median_FFT_for_mouse(trace_dir, annot_dir):
    fft_result = FFT.get_FFT_per_mouse(trace_dir, dsf)
    d_state1, d_state2, d_state3, l_state1, l_state2, l_state3 = annot.get_annot(annot_dir)
    fft_cut_idx = int(len(fft_result)/2)
    dark_fft_result = fft_result[:fft_cut_idx]
    light_fft_result = fft_result[fft_cut_idx:]

    d_fft1 = [dark_fft_result[i] for i in d_state1]
    d_median1 = np.median(d_fft1,axis=0)
    d_fft2 = [dark_fft_result[i] for i in d_state2]
    d_median2 = np.median(d_fft2,axis=0)
    d_fft3 = [dark_fft_result[i] for i in d_state3]
    d_median3 = np.median(d_fft3,axis=0)
    
    l_fft1 = [light_fft_result[i] for i in l_state1]
    l_median1 = np.median(l_fft1,axis=0)
    l_fft2 = [light_fft_result[i] for i in l_state2]
    l_median2 = np.median(l_fft2,axis=0)
    l_fft3 = [light_fft_result[i] for i in l_state3]
    l_median3 = np.median(l_fft3,axis=0)

    return d_median1, d_median2, d_median3, l_median1, l_median2, l_median3

def get_mouse_names(annot_base_dir):
    mouse_name = []
    for filename in os.listdir(annot_base_dir):
        mouse_name.append(filename[:2])
    return mouse_name

mouse_name = get_mouse_names(annot_base_dir)

whole_mouse_dict = {}
if verbose:
    for m_id in tqdm(mouse_name):
        trace_dir = f'{trace_base_dir}/{m_id}_export_24.mat'
        annot_dir = f'{annot_base_dir}/{m_id}_scores.tsv'
        d_med1, d_med2, d_med3, l_med1, l_med2, l_med3 = get_median_FFT_for_mouse(trace_dir, annot_dir)
        whole_mouse_dict[m_id] = [d_med1, d_med2, d_med3, l_med1, l_med2, l_med3]
else:
    for m_id in mouse_name:
        trace_dir = f'{trace_base_dir}/{m_id}_export_24.mat'
        annot_dir = f'{annot_base_dir}/{m_id}_scores.tsv'
        d_med1, d_med2, d_med3, l_med1, l_med2, l_med3 = get_median_FFT_for_mouse(trace_dir, annot_dir)
        whole_mouse_dict[m_id] = [d_med1, d_med2, d_med3, l_med1, l_med2, l_med3]

with open(f'{pickle_save_dir}/mouse_median_dict.pickle', 'wb') as handle:
    pickle.dump(whole_mouse_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
