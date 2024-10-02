import argparse
import csv
import re
import os
import itertools
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
import matplotlib.pylab as plt
from collections import Counter
from helper_function import *

'''
plot sleep bout count and duration for nrem, rem sleep during dark / light phase
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

directory = args.annot_dir
A_label = args.a_label
B_label = args.b_label
save_dir = args.save_dir
if not os.path.exists(save_dir):
  Path(save_dir).mkdir(parents=True)

def get_bout_info(mouse_dir):
  with open(mouse_dir) as f:
    reader = csv.reader(f, delimiter = '\t')
    data = list(reader)

  def get_starting_point(data):
    for i in range(len(data)):
      try:
        if bool(re.search(r'\d', data[i][0])):
          return(i)
          break
      except:
        pass

  starting_point = get_starting_point(data) # get non-header index
  remove_params = data[starting_point:]
  data_only  = [[x.strip() for x in y] for y in  remove_params] # remove white space

  # convert annotation to 1, 2, 3 (int)
  for i in range(len(data_only)):
    data_only[i][4] = data_only[i][4][:4] # remove e and decimal details

  for i in range(len(data_only)):
    if data_only[i][4] == '1.29':
      data_only[i][4] = 1
    if data_only[i][4] == '1.30':
      data_only[i][4] = 2
    if data_only[i][4] == '1.31':
      data_only[i][4] = 3

  for i in range(len(data_only)):
    if data_only[i][4] == '1.00':
      data_only[i][4] = 1
    if data_only[i][4] == '2.00':
      data_only[i][4] = 2
    if data_only[i][4] == '3.00':
      data_only[i][4] = 3

  def get_six_idx(data):
    for i in range(len(data)):
      try:
        if bool(re.search(r'18:00', data[i][1])):
          return(i)
          break
      except:
        pass

  six_idx = get_six_idx(data_only)
  from_six = data_only[six_idx:]
  twenty_four_H_data =  from_six[:8640]

  # split into dark phase, light phase
  def get_light_time(data):
    for i in range(len(data)):
      try:
        if bool(re.search(r'06:00', data[i][1])):
          return(i)
          break
      except:
        pass

  light_idx = get_light_time(twenty_four_H_data)
  dark_phase = twenty_four_H_data[:light_idx]
  light_phase = twenty_four_H_data[light_idx:]

  dark_annotation = [row[4] for row in dark_phase]
  light_annotation = [row[4] for row in light_phase]

  def get_bout_table(input_list):
    groups = []
    col_str = ''.join(str(x) for x in input_list)
    for _, group in itertools.groupby(col_str):
      groups.append(''.join(group))

    _counter = Counter(groups)
    nrem_counter_list = [(key, value) for key, value in _counter.items() if '2' in key] # nrem
    rem_counter_list = [(key, value) for key, value in _counter.items() if '3' in key] # rem

    def get_freq_table(counter_list):
      freq_table = []
      for item in counter_list:
        a = len(item[0])
        b = item[1]
        freq_table.append([a,b])
      return freq_table

    nrem_freq_table = get_freq_table(nrem_counter_list)
    rem_freq_table = get_freq_table(rem_counter_list)
    return nrem_freq_table, rem_freq_table

  d_nrem, d_rem = get_bout_table(dark_annotation)
  l_nrem, l_rem = get_bout_table(light_annotation)

  def get_bout_count(input_list):
    bout_number = sum(row[1] for row in input_list)
    _unpack = sum(([val,]*freq for val, freq in input_list), [])
    bout_duration = sum(_unpack) / len(_unpack)
    return bout_duration, bout_number

  dark_nrem_b_dur, dark_nrem_b_number = get_bout_count(d_nrem)
  dark_rem_b_dur, dark_rem_b_number = get_bout_count(d_rem)
  light_nrem_b_dur, light_nrem_b_number = get_bout_count(l_nrem)
  light_rem_b_dur, light_rem_b_number = get_bout_count(l_rem)

  return {'D_nrem_d_dur': dark_nrem_b_dur, 'D_nrem_d_num': dark_nrem_b_number,
          'D_rem_d_dur': dark_rem_b_dur, 'D_rem_d_num': dark_rem_b_number,
          'L_nrem_d_dur': light_nrem_b_dur, 'L_nrem_d_num': light_nrem_b_number,
          'L_rem_d_dur': light_rem_b_dur, 'L_rem_d_num': light_rem_b_number,}


def get_mat_dictionary(directory = directory):
  bout_list = []
  for filename in os.listdir(directory):
      f = os.path.join(directory, filename)
      if os.path.isfile(f):
          ele = get_bout_info(f)
          ele_list = list(ele.values())
          ele_list.insert(0, filename[:2])
          bout_list.append(ele_list) # fill dictionary with format {mouse number : state count matrix}
      else:
        raise ValueError(f'{f} is not a dir')
  return bout_list

bout_list = get_mat_dictionary()
bout_df = pd.DataFrame(bout_list, columns=['ID', 'd_nrem_dur', 'd_nrem_num', 'd_rem_dur', 'd_rem_num',
                                           'l_nrem_dur', 'l_nrem_num', 'l_rem_dur', 'l_rem_num'])
def label_group(row):
  if 'A' in row['ID']:
    return A_label
  else:
    return B_label

bout_df['group'] = bout_df.apply(label_group, axis=1)


def plot_figure(_mode, _palette, save = False):
  def get_p_val(data_frame, mode):
    if mode == 'duration':
      A = list(data_frame[data_frame['group'] == A_label]['Bout Duration'])
      B = list(data_frame[data_frame['group'] == B_label]['Bout Duration'])
      _stats =  pg.ttest(A, B, paired = True)
      p_val = float(_stats['p-val'])
    elif mode == 'number':
      A = list(data_frame[data_frame['group'] == A_label]['Bout Number'])
      B = list(data_frame[data_frame['group'] == B_label]['Bout Number'])
      _stats =  pg.ttest(A, B, paired = True)
      p_val = float(_stats['p-val'])
    else:
      pass
    return data_frame, p_val

  if _mode == 'NREM':
    dark_dur_merge = bout_df[['group', 'd_nrem_dur']]
    dark_dur_merge.rename(columns={"d_nrem_dur": "Bout Duration"}, inplace=True)
    light_dur_merge = bout_df[['group', 'l_nrem_dur']]
    light_dur_merge.rename(columns={"l_nrem_dur": "Bout Duration"}, inplace=True)
    dark_num_merge = bout_df[['group', 'd_nrem_num']]
    dark_num_merge.rename(columns={"d_nrem_num": "Bout Number"}, inplace=True)
    light_num_merge = bout_df[['group', 'l_nrem_num']]
    light_num_merge.rename(columns={"l_nrem_num": "Bout Number"}, inplace=True)
    
  elif _mode == 'REM':
    dark_dur_merge = bout_df[['group', 'd_rem_dur']]
    dark_dur_merge.rename(columns={"d_rem_dur": "Bout Duration"}, inplace=True)
    light_dur_merge = bout_df[['group', 'l_rem_dur']]
    light_dur_merge.rename(columns={"l_rem_dur": "Bout Duration"}, inplace=True)
    dark_num_merge = bout_df[['group', 'd_rem_num']]
    dark_num_merge.rename(columns={"d_rem_num": "Bout Number"}, inplace=True)
    light_num_merge = bout_df[['group', 'l_rem_num']]
    light_num_merge.rename(columns={"l_rem_num": "Bout Number"}, inplace=True)

  dark_dur_merge['Bout Duration'] = dark_dur_merge['Bout Duration'].apply(lambda x: x*10)
  light_dur_merge['Bout Duration'] = light_dur_merge['Bout Duration'].apply(lambda x: x*10)
  dark_dur_df, dark_dur_P = get_p_val(dark_dur_merge, 'duration')
  light_dur_df, light_dur_P = get_p_val(light_dur_merge, 'duration')
  dark_num_df, dark_num_P = get_p_val(dark_num_merge, 'number')
  light_num_df, light_num_P = get_p_val(light_num_merge, 'number')

  f, axes = plt.subplots(nrows = 2, ncols = 2, figsize=(5, 9))

  if _palette == 'red':
    pal = ['blue', 'red']
  elif _palette == 'green':
    pal = ['#91cb98', '#afdee8']
  else:
    pass

  axes[0,0].set_ylim([0, 220])
  if _mode == 'REM':
    axes[0,0].set_ylim([0, 100])
  sns.barplot(dark_dur_df, x ='group', y = 'Bout Duration', errorbar='ci',
              ax = axes[0,0], palette = pal, alpha = 0.9)

  sns.stripplot(x="group", y="Bout Duration", data=dark_dur_df, dodge=True, 
                alpha=0.8, ax= axes[0,0] , color = 'black', s = 10,  marker="$\circ$")
  axes[0,0].set(xlabel=None)
  axes[0,0].set(ylabel = f'{_mode} Bout Duration(s)')
  axes[0,0].title.set_text(f'{_mode} Bout Duration (Dark)')


  axes[0, 1].set_ylim([0, 250])
  if _mode == 'REM':
    axes[0,1].set_ylim([0, 100])
  sns.barplot(light_dur_df, x ='group', y = 'Bout Duration', errorbar='ci', 
                ax = axes[0, 1], palette = pal, alpha = 0.9)
  sns.stripplot(x="group", y="Bout Duration", data=light_dur_df, dodge=True, 
                alpha=0.8, ax= axes[0, 1] , color = 'black', s = 10,  marker="$\circ$")
  axes[0,1].set(xlabel=None)
  axes[0,1].set(ylabel = f'{_mode} Bout Duration(s)')
  axes[0, 1].title.set_text(f'{_mode} Bout Duration (Light)')

  axes[1, 0].set_ylim([0, 400])
  if _mode == 'REM':
    axes[1,0].set_ylim([0, 60])
  sns.barplot(dark_num_df, x ='group', y = 'Bout Number', errorbar='ci', ax = axes[1,0], palette = pal, alpha = 0.9)
  sns.stripplot(x="group", y="Bout Number", data=dark_num_df, 
                dodge=True, alpha=0.8, ax= axes[1,0] , color = 'black', s = 10,  marker="$\circ$")
  axes[1,0].set(xlabel=None)
  axes[1,0].set(ylabel = f'{_mode} Bout Number')
  axes[1,0].title.set_text(f'{_mode} Bout Number (Dark)')

  axes[1,1].set_ylim([0, 400])
  if _mode == 'REM':
    axes[1,1].set_ylim([0, 90])
  sns.barplot(light_num_df, x ='group', y = 'Bout Number', errorbar='ci', 
              ax = axes[1,1], palette = pal, alpha = 0.9)
  sns.stripplot(x="group", y="Bout Number", data=light_num_df, dodge=True, 
                alpha=0.8, ax= axes[1,1] , color = 'black', s = 10,  marker="$\circ$")
  axes[1,1].set(xlabel=None)
  axes[1,1].set(ylabel = f'{_mode} Bout Number')
  axes[1,1].title.set_text(f'{_mode} Bout Number (Light)')


  def annot_stat(star, x1, x2, y, h, col='k', ax=None):
      if star != 'ns':
        ax = plt.gca() if ax is None else ax
        ax.plot([x1, x1, x2, x2], [y, y, y, y], lw=1.5, c=col)
        ax.text((x1+x2)*.5, y+h, star, ha='center', va='bottom', color=col, size = 10)

  dark_dur_star = convert_pvalue_to_asterisks(dark_dur_P)
  light_dur_star = convert_pvalue_to_asterisks(light_dur_P)
  dark_num_star = convert_pvalue_to_asterisks(dark_num_P)
  light_num_star = convert_pvalue_to_asterisks(light_num_P)

  if _mode == 'NREM':
    annot_stat(dark_dur_star, 0, 1, 175, 2, ax=axes[0,0])
    annot_stat(light_dur_star, 0, 1, 200, 2, ax=axes[0,1])
    annot_stat(dark_num_star, 0, 1, 325, 2, ax=axes[1,0])
    annot_stat(light_num_star, 0, 1, 325, 2, ax=axes[1,1])
  elif _mode == 'REM':
    annot_stat(dark_dur_star, 0, 1, 90, 2, ax=axes[0,0])
    annot_stat(light_dur_star, 0, 1, 90, 2, ax=axes[0,1])
    annot_stat(dark_num_star, 0, 1, 48, 2, ax=axes[1,0])
    annot_stat(light_num_star, 0, 1, 75, 2, ax=axes[1,1])
  plt.tight_layout()
  if save == True:
    plt.savefig(f'{save_dir}{A_label}{B_label}_bout_{_palette}_{_mode}.png', bbox_inches='tight')
  plt.show()

plot_figure('REM', 'red', save = True)
plot_figure('REM', 'green', save = True)
plot_figure('NREM', 'red', save = True)
plot_figure('NREM', 'green', save = True)
