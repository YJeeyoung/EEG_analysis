import argparse
import csv
import os
import re
import statistics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
from logging import raiseExceptions
from helper_function import get_count_mat_dictionary, convert_six_dict

'''
plot time series information for wake, nrem, rem state
requires sleep state annotation files
'''

parser = argparse.ArgumentParser()
parser.add_argument("--annot_dir", type=str, default=None, 
                        required=True, help = 'copy paste directory for sleep state annotation files')    
parser.add_argument("--a_label", type=str, default='A', help = 'label for A group')
parser.add_argument("--b_label", type=str, default='B', help = 'label for B group')
parser.add_argument("--bin_count", type=int, default = 2, help = 'bin for smoothing time')
parser.add_argument("--save_dir", type=str, default='save_imgs/', help = 'directory to save output figures')
args = parser.parse_args()

dir = args.annot_dir
A = args.a_label
B = args.b_label
bin_count = args.bin_count
save_dir = args.save_dir
if not os.path.exists(save_dir):
  Path(save_dir).mkdir(parents=True)

def get_bin_percent_list(dir):
  mat_dict = get_count_mat_dictionary(dir)
  mat_dict = convert_six_dict(mat_dict)
  def bin_transform(mat_dict, bin=bin_count):
    bin_dict = {}
    keysList = list(mat_dict.keys())

    for item in keysList:
        su=[sum(i) for i in mat_dict[item]]
        assert len(su) == 24, 'one day should be 24 hours'
        assert all(x == 360 for x in su), 'each hour should have 360 data points'
        emp = []
        for i in range(0, len(mat_dict[item]), bin):
          emp.append((mat_dict[item][i] + mat_dict[item][i+1]).tolist())
          bin_dict[item] = emp
    return bin_dict

  def conv_percentag(bin = bin_count):
    bin_mat = bin_transform(mat_dict)
    new_dict = {}
    for item in list(bin_mat.keys()):
        emp = []
        mat = [[(item / (360*bin)) * 100 for item in subl] for subl in bin_mat[item]]
        emp.extend(mat)
        new_dict[item] = emp
    return new_dict
  bin_per_mat = conv_percentag()
  return bin_per_mat

per_mat = get_bin_percent_list(dir)
per_mat = dict(sorted(per_mat.items()))

wake_list = []
nrem_list = []
rem_list = []

for key in list(per_mat.keys()):
  wake = [row[0] for row in per_mat[key]] 
  nrem = [row[1] for row in per_mat[key]] 
  rem = [row[2] for row in per_mat[key]]
  
  wake.insert(0, key)
  nrem.insert(0, key)
  rem.insert(0, key)
  
  wake_list.append(wake)
  nrem_list.append(nrem)
  rem_list.append(rem)

wake_df = pd.DataFrame(wake_list, columns=['ID', *range(0,23,2)]) 
nrem_df = pd.DataFrame(nrem_list, columns=['ID', *range(0,23,2)]) 
rem_df = pd.DataFrame(rem_list, columns=['ID', *range(0,23,2)]) 


def plot_time_series_data(df, _type, scale = 100, _mode = 'shade'):
  '''visualize each state's time series information'''
  a_df = df[df['ID'].str.contains("A")]
  b_df = df[df['ID'].str.contains("B")]
  a_df_list = a_df.values.tolist()
  b_df_list = b_df.values.tolist()
  a_data = [item[1:] for item in a_df_list]
  b_data = [item[1:] for item in b_df_list]

  a_mean = []
  a_st_err = stats.sem(a_data)
  for i in range(len(a_data[0])):
    a_col_val = [row[i] for row in a_data]
    a_mean.append(sum(a_col_val)/len(a_col_val))

  b_mean = []
  b_st_err = stats.sem(b_data)
  for i in range(len(b_data[0])):
    b_col_val = [row[i] for row in b_data]
    b_mean.append(sum(b_col_val)/len(b_col_val))

  fig = plt.figure()
  a_yerr = [a_st_err, a_st_err]  # specify 'down' error and 'up' error
  x = range(0, 24, 2)
  plt.ylim(-3 , scale)

  if scale < 100:
    plt.ylim(-0.5 , scale)
    yint = range(0, scale+1, 2)
    plt.yticks(yint)


  dark_light_a_mean = a_mean
  dark_light_b_mean = b_mean


  if _mode == 'error_bar':
    plt.axvline(x = 11, color = 'black', ls = '--', lw = 0.5)
    plt.errorbar(x, dark_light_a_mean, yerr=a_yerr, capsize=3, fmt="b--o", ecolor = "blue", label = A)

    b_yerr = [b_st_err, b_st_err]  # specify 'down' error and 'up' error
    plt.errorbar(x, dark_light_b_mean, yerr=b_yerr, capsize=3, fmt="r--o", ecolor = "red", label = B)
    plt.xlabel("Time (hour)")
    plt.ylabel("% of time")
    if _type == 'Wake':
        plt.title(' Wake across 24 hours')
    elif _type == 'REM':
        plt.title('REM across 24 hours')
    elif _type == 'NREM':
        plt.title('NREM across 24 hours')
    else:
        raise ('Enter proper state information')

    if scale == 100:
        plt.axhline(y=0, color='black', linestyle='-')
        plt.axhline(y=-1.8, color='black', linestyle='-')
        plt.axhline(y=-1, xmin = 0, xmax = 0.5, color='black', linestyle='-')
        plt.axhline(y=-0.5, xmin = 0, xmax = 0.5, color='black', linestyle='-')
        plt.axhline(y=-1.5, xmin = 0, xmax = 0.5, color='black', linestyle='-')

    else:
        plt.axhline(y=0, color='black', linestyle='-')
        plt.axhline(y=-0.3, color='black', linestyle='-')
        plt.axhline(y=-0.1, xmin = 0, xmax = 0.5, color='black', linestyle='-')
        plt.axhline(y=-0.2, xmin = 0, xmax = 0.5, color='black', linestyle='-')

    plt.legend()
    plt.savefig(f'{save_dir}{A}{B}_{_type}_{_mode}.png', bbox_inches='tight')
    plt.show()
  else:
    plt.xlim(0 , 22)
    plt.xticks([0,5,10,15,20], [0,5,10,15,20])
        
    plt.axvline(x = 11, color = 'black', ls = '--', lw = 0.5)
    plt.plot(x, dark_light_a_mean, ls = '-', color = 'black', lw = 0.8)
    plt.fill_between(x, dark_light_a_mean-a_st_err, dark_light_a_mean+a_st_err,
        alpha=0.8, edgecolor='#000000', facecolor='#91cb98',
        linewidth=0, label = A)


    b_yerr = [b_st_err, # 'down' error
            b_st_err]  # 'up' error
    plt.plot(x, dark_light_b_mean, ls = '-', color = 'black', lw = 0.8)
    plt.fill_between(x, dark_light_b_mean-b_st_err, dark_light_b_mean+b_st_err,
        alpha=0.8, edgecolor='#000000', facecolor='#afdee8',
        linewidth=0, label = B)

    plt.xlabel("Time (hour)")
    plt.ylabel("% of time")
    if _type == 'Wake':
        plt.title(' Wake across 24 hours')
    elif _type == 'REM':
        plt.title('REM across 24 hours')
    elif _type == 'NREM':
        plt.title('NREM across 24 hours')
    else:
        raise ('Enter proper state information')

    if scale == 100:
        plt.axhline(y=0, color='black', linestyle='-')
        plt.axhline(y=-1.8, color='black', linestyle='-')
        plt.axhline(y=-1, xmin = 0, xmax = 0.5, color='black', linestyle='-')
        plt.axhline(y=-0.5, xmin = 0, xmax = 0.5, color='black', linestyle='-')
        plt.axhline(y=-1.5, xmin = 0, xmax = 0.5, color='black', linestyle='-')

    else:
        plt.axhline(y=0, color='black', linestyle='-')
        plt.axhline(y=-0.3, color='black', linestyle='-')
        plt.axhline(y=-0.1, xmin = 0, xmax = 0.5, color='black', linestyle='-')
        plt.axhline(y=-0.2, xmin = 0, xmax = 0.5, color='black', linestyle='-')


    plt.legend()
    plt.savefig(f'{save_dir}{A}{B}_{_type}_{_mode}.png', bbox_inches='tight')
    plt.show()
     
plot_time_series_data(wake_df, 'Wake', _mode = 'error_bar')
plot_time_series_data(nrem_df, 'NREM', _mode = 'error_bar')
plot_time_series_data(rem_df, 'REM', scale = 18, _mode = 'error_bar')

plot_time_series_data(wake_df, 'Wake')
plot_time_series_data(nrem_df, 'NREM')
plot_time_series_data(rem_df, 'REM', scale = 18)