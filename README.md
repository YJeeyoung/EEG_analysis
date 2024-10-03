# EEG trace and annotation analysis
This repo includes codes for analyzing EEG recordings, including 1) fast Fourier transform on raw trace and 2) annotation analysis.
Required inputs are  
```
1. EEG trace file
2. Annotation (wake, nrem, rem) file
```

## 1. Power Analysis
### Running FFT on EEG trace 
At the following command to conduct FFT for each state (wake, nrem, rem), and save median FFT result for each mouse..   
```python power_analysis/read_annot_get_state.py --annot_dir dir_to_state_annotations --trace_dir dir_to_raw_trace```


Following arguments are optional:
```
-h, --help : show help message
--save_dir : directory to save median FFT output. Default setting will automatically make save_var folder and save it there.  
--verbose : If true, tqdm progress par will be shown for while running FFT.  
--down_sampling_rate : Down sampling rate before conducting FFT. Defualt setting is 4, so 1000Hz recoding will be downsampled to 250Hz.
```
### Visualizing power analysis result
Run the following command to visualize the raw power.
```python power_analysis/visualize_raw_power.py```

Although there is no required argument, saving directory and label name for each group can be adjusted.
```
--median_FFT_dir : directory for median FFT result from run_FFT_for_states.py
--img_save_dir : dir for saving raw power visualization
--a_label : label for A group  
--b_label : label for B group
```
