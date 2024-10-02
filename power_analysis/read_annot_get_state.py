import os
import csv
import pandas as pd
import numpy as np
import re
from collections import Counter
import itertools

'''get annotation index for wake, nrem, rem state'''

def get_annot(mouse_dir):
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
    if data_only[i][4] == '1.00':
      data_only[i][4] = 1
    if data_only[i][4] == '2.00':
      data_only[i][4] = 2
    if data_only[i][4] == '3.00':
      data_only[i][4] = 3

  twentyfour_H_data =  data_only[:8640] # first 24H data.
  phase_idx = int(len(twentyfour_H_data)/2)
  dark_phase = twentyfour_H_data[:phase_idx]
  light_phase = twentyfour_H_data[phase_idx:]
  assert len(dark_phase) == len(light_phase), 'unequal phase len'
  
  d_annotations = [row[4] for row in dark_phase]
  d_indices_1 = [i for i, x in enumerate(d_annotations) if x == 1]
  d_indices_2 = [i for i, x in enumerate(d_annotations) if x == 2]
  d_indices_3 = [i for i, x in enumerate(d_annotations) if x == 3]

  l_annotations = [row[4] for row in light_phase]
  l_indices_1 = [i for i, x in enumerate(l_annotations) if x == 1]
  l_indices_2 = [i for i, x in enumerate(l_annotations) if x == 2]
  l_indices_3 = [i for i, x in enumerate(l_annotations) if x == 3]
  
  
  return d_indices_1, d_indices_2, d_indices_3, l_indices_1, l_indices_2, l_indices_3