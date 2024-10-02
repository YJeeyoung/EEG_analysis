import argparse
import csv
import os
import re
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
import matplotlib.pylab as plt
from pathlib import Path
from helper_function import get_count_mat_dictionary

'''
plot percent time spent in wake, nrem, rem state during dark / light phase
requires sleep state annotation files
'''

parser = argparse.ArgumentParser()
parser.add_argument("--annot_dir", type=str, default=None,
                      required=True, help= 'copy paste directory for sleep annotation files.tsv')
parser.add_argument("--a_label", type=str, default='A', help = 'label for A group')
parser.add_argument("--b_label", type=str, default='B', help = 'label for B group')
parser.add_argument("--save_dir", type=str, default='save_imgs/', help = 'directory to save output figures')
args = parser.parse_args()

directory = args.annot_dir
A_label = args.a_label
B_label = args.b_label
save_dir = args.save_dir
if not os.path.exists(save_dir):
  Path(save_dir).mkdir(parents=True)

def convert_dark_light(mat_dict):
  dark_dict = {}
  light_dict = {}
  for item in list(mat_dict.keys()):
    dark_list = mat_dict[item][-6:].tolist() + mat_dict[item][:6].tolist()
    light_list = mat_dict[item][6:-6].tolist()
    dark_dict[item] = dark_list
    light_dict[item] = light_list
  return dark_dict, light_dict
mat_dict = get_count_mat_dictionary(directory)
dark_dict, light_dict = convert_dark_light(mat_dict)

def get_col_sums(_dict):
  sum_dict = {}
  for item in list(_dict.keys()):
    m_list = _dict[item]
    col_totals = [ sum(x) for x in zip(*m_list) ]
    sum_dict[item] = col_totals
  return sum_dict
dark_col_sum = get_col_sums(dark_dict)
light_col_sum = get_col_sums(light_dict)

def get_state_percentage(sum_dict):
  percentage_dict = {}
  for item in list(sum_dict.keys()):
    m_list = sum_dict[item]
    divd = sum(m_list)
    percentage_dict[item] = [x/divd for x in sum_dict[item]]
  return percentage_dict
dark_sum_dict = get_state_percentage(dark_col_sum)
light_sum_dict = get_state_percentage(light_col_sum)

def split_a_b(_dict):
  a_dict = {}
  b_dict = {}
  for item in list(_dict.keys()):
    if 'A' in item:
      a_dict[item] = _dict[item]
    elif 'B' in item:
      b_dict[item] = _dict[item]
    else:
      raise ValueError('slip')
  return a_dict, b_dict

a_dark_sum, b_dark_sum = split_a_b(dark_sum_dict)
a_light_sum, b_light_sum = split_a_b(light_sum_dict)

a_dark_wake = ([row[0] for row in list(a_dark_sum.values())])
a_dark_NREM = ([row[1] for row in list(a_dark_sum.values())])
a_dark_REM = ([row[2] for row in list(a_dark_sum.values())])

b_dark_wake = ([row[0] for row in list(b_dark_sum.values())])
b_dark_NREM = ([row[1] for row in list(b_dark_sum.values())])
b_dark_REM = ([row[2] for row in list(b_dark_sum.values())])

a_light_wake = ([row[0] for row in list(a_light_sum.values())])
a_light_NREM = ([row[1] for row in list(a_light_sum.values())])
a_light_REM = ([row[2] for row in list(a_light_sum.values())])

b_light_wake = ([row[0] for row in list(b_light_sum.values())])
b_light_NREM = ([row[1] for row in list(b_light_sum.values())])
b_light_REM = ([row[2] for row in list(b_light_sum.values())])

dark_per_df = pd.DataFrame.from_dict(dict(sorted(dark_sum_dict.items())), orient='index', columns = ['wake', 'nrem', 'rem'])
dark_per_df = dark_per_df.apply(lambda x: x*100)

light_per_df = pd.DataFrame.from_dict(dict(sorted(light_sum_dict.items())), orient='index', columns = ['wake', 'nrem', 'rem'])
light_per_df = light_per_df.apply(lambda x: x*100)

def label_group(row):
  if 'A' in row['ID']:
    return A_label
  else:
    return B_label

def convert_pvalue_to_asterisks(pvalue):
    if pvalue <= 0.0001:
        return "****"
    elif pvalue <= 0.001:
        return "***"
    elif pvalue <= 0.01:
        return "**"
    elif pvalue <= 0.05:
        return "*"
    return "ns"

def plot_figure(data_frame, phase, palette_info):
  def get_data_and_p_val(data_frame, mode):
    subset_df = data_frame[['group',mode]]
    subset_df.columns = ['group', 'value']
    A = list(subset_df[subset_df['group'] == A_label]['value'])
    B = list(subset_df[subset_df['group'] == B_label]['value'])
    _stats =  pg.mwu(A, B, alternative='two-sided')
    p_val = float(_stats['p-val'])
    return subset_df, p_val
  
  data_frame['ID'] = data_frame.index
  data_frame['group'] = data_frame.apply(label_group, axis=1)
  
  w_df, w_pval = get_data_and_p_val(data_frame, 'wake')
  rem_df, rem_pval = get_data_and_p_val(data_frame, 'rem')
  nrem_df, nrem_pval = get_data_and_p_val(data_frame, 'nrem')

  f, axes = plt.subplots(1, 3, figsize=(6, 5))
  f.tight_layout()

  if palette_info == 'green':
    pal = ['#91cb98', '#afdee8']
  elif palette_info == 'red':
    pal = ['blue', 'red']
  else:
    raise Exception('enter proper palette information')

  if phase == 'dark':
    axes[0].set_ylim([0, 100])
  elif phase == 'light':
    axes[0].set_ylim([0, 65])
  else:
    raise Exception('enter proper phase information')  
  
  sns.barplot(w_df, x ='group', y = 'value', errorbar='ci', ax = axes[0], palette = pal, alpha = 1)
  sns.stripplot(x="group",y="value",data=w_df, dodge=True, 
                alpha=0.8, ax= axes[0] , color = 'black', s = 10,  marker="$\circ$")
  axes[0].title.set_text(f'Wake ({phase} phase)')
  axes[0].set(xlabel=None)
  axes[0].set(ylabel = '% of time')

  if phase == 'dark':
    axes[1].set_ylim([0, 60])
  elif phase == 'light':
    axes[1].set_ylim([0, 90])
  else:
    raise Exception('enter proper phase information')  

  sns.barplot(nrem_df, x ='group', y = 'value', errorbar='ci', ax = axes[1], palette = pal, alpha = 1)
  sns.stripplot(x="group", y="value", data=nrem_df, dodge=True, 
                alpha=0.8, ax= axes[1] , color = 'black', s = 10,  marker="$\circ$")
  axes[1].title.set_text(f'NREM ({phase} phase)')
  axes[1].set(xlabel=None)
  axes[1].set(ylabel = '% of time')

  if phase == 'dark':
    axes[2].set_ylim([0, 7])
  elif phase == 'light':
    axes[2].set_ylim([0, 15])
  else:
    raise Exception('enter proper phase information')  

  sns.barplot(rem_df, x ='group', y = 'value', errorbar='ci', ax = axes[2], palette = pal, alpha = 1)
  sns.stripplot(x="group", y="value", data=rem_df, dodge=True,
                alpha=0.8, ax= axes[2] , color = 'black', s = 10,  marker="$\circ$")
  axes[2].title.set_text(f'REM ({phase} phase)')
  axes[2].set(xlabel=None)
  axes[2].set(ylabel = '% of time')

  def annot_stat(star, x1, x2, y, h, col='k', ax=None):
      if star != 'ns':
        ax = plt.gca() if ax is None else ax
        ax.plot([x1, x1, x2, x2], [y, y, y, y], lw=1.5, c=col)
        ax.text((x1+x2)*.5, y+h, star, ha='center', va='bottom', color=col, size = 10)

  w_star = convert_pvalue_to_asterisks(w_pval)
  rem_star = convert_pvalue_to_asterisks(rem_pval)
  nrem_star = convert_pvalue_to_asterisks(nrem_pval)

  if phase == 'dark':
    annot_stat(w_star, 0, 1, 83, 2, ax=axes[0])
    annot_stat(rem_star, 0, 1, 5.5, 0.1, ax=axes[2])
    annot_stat(nrem_star, 0, 1, 50, 1, ax=axes[1])
  else:
    annot_stat(w_star, 0, 1, 83, 2, ax=axes[0])
    annot_stat(rem_star, 0, 1, 11, 0.1, ax=axes[2])
    annot_stat(nrem_star, 0, 1, 53, 1, ax=axes[1])
  plt.tight_layout()
  plt.savefig(f'{save_dir}/{A_label}{B_label}_{phase}_{palette_info}.png', bbox_inches='tight')
  plt.show()

plot_figure(dark_per_df, 'dark', 'red')
plot_figure(light_per_df, 'light', 'red')

plot_figure(dark_per_df, 'dark', 'green')
plot_figure(light_per_df, 'light', 'green')
