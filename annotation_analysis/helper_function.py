import csv
import os
import re
import numpy as np
import pandas as pd

'''helper functions for data preprocessing and statistical analysis'''

def get_count_mat_dictionary(dir):
  def get_count_matrix(file):
    # read data
    with open(file) as f:
        reader = csv.reader(f, delimiter = '\t')
        data = list(reader)

    # add hourly information (will be used for bin count)
    for i in range(len(data)):
        try:
            if data[i][0][0].isdigit() == True:
                data[i][-1] = re.findall("\S+",data[i][1])[0][:2]
        except: # pass initial parameter rows
            pass

    def get_starting_point(data):
        for i in range(len(data)):
            try:
                if bool(re.search(r'\d', data[i][0])):
                    return(i)
            except:
                pass

    starting_point = get_starting_point(data)
    remove_params = data[starting_point:]
    data_only  = [[x.strip() for x in y] for y in  remove_params] # remove white space and indentation

    # clean and rewrite annotations
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
            except:
                pass
    
    six_idx = get_six_idx(data_only)
    from_six = data_only[six_idx:]
    twentyfour_H_data =  from_six[:8640] # from 6pm to next day 6pm. 24h data.

    # change data type (to use for matrix indexing later)
    for i in range(len(twentyfour_H_data)):
        twentyfour_H_data[i][-1] = int(twentyfour_H_data[i][-1])

    cols = data[10] # get column information
    score_name = cols[-1]

    cols.append('time_hour')
    df = pd.DataFrame(twentyfour_H_data, columns=data[10])
    dictionary =  dict(df.value_counts(subset=['time_hour', score_name]))
    

    count_matrix = np.zeros((24, 3))
    for i in range(24):
        for j in [1, 2, 3]:
            try:
                val = dictionary[(i, j)]
                count_matrix[i,j-1] = val
            except:
                pass

    return count_matrix

  mat_dict = {}
  for filename in os.listdir(dir):
    f = os.path.join(dir, filename)
    if os.path.isfile(f):
        ele = get_count_matrix(f)
        mat_dict[filename[:2]] = ele # fill dictionary with format {mouse number : state count matrix}
    else:
        raise ValueError(f'{f} is not a dir')
  return mat_dict

def convert_six_dict(mat_dict):
    '''mat dict counts from 0am. convert it to start from 6pm.'''
    six_dict = {}
    for key, item in mat_dict.items():
        _list = mat_dict[key]
        #print('before :', _list)
        six_list = np.array(_list[-6:].tolist() + _list[:-6].tolist())
        #print('after :', six_list)
        assert len(six_list) == len(_list), 'indexing error'
        six_dict[key] = six_list
    return six_dict

def convert_pvalue_to_asterisks(pvalue):
    #print('p value :', pvalue)
    if pvalue <= 0.0001:
        return "****"
    elif pvalue <= 0.001:
        return "***"
    elif pvalue <= 0.01:
        return "**"
    elif pvalue <= 0.05:
        return "*"
    return "ns"