#calculate each wave occurance from normalized median FFT, save output df as csv.
import argparse
import os
import pickle
import pandas as pd
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--median_dir", type=str, default=None, required=True, 
                    help = 'copy paste directory for median FFT results')
parser.add_argument("--a_label", type=str, default='A', help = 'label for A group')
parser.add_argument("--b_label", type=str, default='B', help = 'label for B group')
parser.add_argument("--save_dir", type=str, default='save_var/', help = 'directory to save median fft')  
args = parser.parse_args()
median_dir = args.median_dir
A = args.a_label
B = args.b_label
save_dir = args.save_dir

if not os.path.exists(save_dir):
  Path(save_dir).mkdir(parents=True)

with open(f'{median_dir}', 'rb') as handle:
    whole_mouse_dict = pickle.load(handle)

freq_dict = {}
freq_dict['delta'] = [*range(5-5,41-5,1)]
freq_dict['theta'] = [*range(60-5, 100-5, 1)]
freq_dict['alpha'] = [*range(100-5, 150-5,1)]
freq_dict['beta'] = [*range(150-5, 300-5,1)]
freq_dict['gamma'] = [*range(300-5, 500-5,1)]

def norm_FFT(_list): # normaluze FFT result. % conversion until 50Hz.
    cut_list = _list[5:501]
    denom = sum(cut_list)
    norm_list = [item/denom*100 for item in cut_list]
    return norm_list

def freq_sum(FFT_result): # 해당 쥐의 median fft 결과를 토대로 각 주파수 합 구함
    _sum_list = []
    for key, value in freq_dict.items():
        _sum_list.append(sum(FFT_result[i] for i in value))
    return _sum_list

def label_group(row):
  if 'A' in row['ID']:
    return A
  else:
    return B

def make_row_per_mouse(mouse_id):
    mouse_list = []
    dark_wake = norm_FFT(whole_mouse_dict[mouse_id][0])      
    dark_nrem = norm_FFT(whole_mouse_dict[mouse_id][1])
    dark_rem = norm_FFT(whole_mouse_dict[mouse_id][2])
    light_wake = norm_FFT(whole_mouse_dict[mouse_id][3])
    light_nrem = norm_FFT(whole_mouse_dict[mouse_id][4])
    light_rem = norm_FFT(whole_mouse_dict[mouse_id][5])
    
    dark_wake_freq = freq_sum(dark_wake)
    dark_wake_freq.extend(['dark', 'wake', mouse_id])
    mouse_list.append(dark_wake_freq)
    
    dark_nrem_freq = freq_sum(dark_nrem)
    dark_nrem_freq.extend(['dark', 'nrem', mouse_id])
    mouse_list.append(dark_nrem_freq)

    dark_rem_freq = freq_sum(dark_rem)
    dark_rem_freq.extend(['dark', 'rem', mouse_id])
    mouse_list.append(dark_rem_freq)
    
    light_wake_freq = freq_sum(light_wake)
    light_wake_freq.extend(['light', 'wake', mouse_id])
    mouse_list.append(light_wake_freq)

    light_nrem_freq = freq_sum(light_nrem)
    light_nrem_freq.extend(['light', 'nrem', mouse_id])
    mouse_list.append(light_nrem_freq)
    
    light_rem_freq = freq_sum(light_rem)
    light_rem_freq.extend(['light', 'rem', mouse_id])
    mouse_list.append(light_rem_freq)

    return mouse_list

total_list = []
for key, value in whole_mouse_dict.items():
    total_list.extend(make_row_per_mouse(key))

freq_df = pd.DataFrame(total_list, columns=['delta', 'theta', 'alpha', 'beta', 'gamma', 'phase', 'state','ID'])

freq_df['group'] = freq_df.apply(label_group, axis=1)

df = freq_df[['group', 'ID', 'phase', 'state', 'delta', 'theta', 'alpha', 'beta', 'gamma']]
df.to_csv(f'{save_dir}/delta_df.csv', index=False) 
