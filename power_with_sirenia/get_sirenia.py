import mne
import numpy as np
import h5py
file = "C:/Users/jeeyoung/data_from_LF/data_after_241125/241121_scoring_from_LF/0_JY_Trap_Mice_Data_Finalize/DataToJeeyoung/A1_export.edf"
data = mne.io.read_raw_edf(file)
raw_data = data.get_data()[0].reshape(-1)
print('###raw data###', raw_data)
print(len(raw_data)) # 이거 86400000 len 으로 맞춰서 잘라줘야함

file = "C:/Users/jeeyoung/data_from_LF/data_after_241125/old_mouse_gfp_vegfc/old_mouse_n_8/Raw_Trace/A1_export_24.mat"
EEG = np.array( h5py.File(file,'r').get('EEG1'))[0].reshape(-1)    
print('EEG', EEG)
print(len(EEG))
# # you can get the metadata included in the file and a list of all channels:
# info = data.info
# print('###info###', info)
# channels = data.ch_names
# #print(channels[0])
# channel_name = channels[0]
# print(channel_name)
# print(data[channel_name])

# print(data[channel_name][0])
# print(data[channel_name][1])
