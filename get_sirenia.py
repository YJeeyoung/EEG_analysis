import mne
import numpy as np
import h5py

'''codes for extracting EEG trace from sirenia export file. File format should be .edf or .mat'''

file = "dir/to/your/edf/export/from/sirenia.edf"
data = mne.io.read_raw_edf(file)
raw_data = data.get_data()[0].reshape(-1)

file = "dir/to/your/matlab/file.mat"
EEG = np.array( h5py.File(file,'r').get('EEG1'))[0].reshape(-1)  
