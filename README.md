# EEG trace and annotation analysis
This repo includes code for analyzing EEG recordings, including 1) fast Fourier transform on raw traces and 2) annotation analysis.
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
--verbose : If true, tqdm progress bar will be shown for while running FFT.  
--down_sampling_rate : Down sampling rate before conducting FFT. Default setting is 4, so 1000Hz recording will be downsampled to 250Hz.
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

## 2. Annotation Analysis
Based on sleep state annotation information, following visualizations can be conducted:
```
- time series visualization
- percent time spent in each state
- bout frequency and bout duration
```

### Time series visualization
Run the following command:
```
python annotation_analysis/time_series_analysis.py --annot_dir dir_to_state_annotations
```

Following arguments are optional:
```
-h, --help : show help message
--save_dir : directory to save visualization result.
--a_label : label for A group  
--b_label : label for B group
--bin_count : bin count for smoothing. default is 2
```

### Percent time spent in each state
Run the following command:
```
python annotation_analysis/percent_time_analysis.py --annot_dir dir_to_state_annotations
```

Following arguments are optional:
```
-h, --help : show help message
--save_dir : directory to save visualization result.
--a_label : label for A group  
--b_label : label for B group
```

### Bout analysis
Run the following command:
```
python annotation_analysis/bout_analysis_analysis.py --annot_dir dir_to_state_annotations
```

Following arguments are optional:
```
-h, --help : show help message
--save_dir : directory to save visualization result.
--a_label : label for A group  
--b_label : label for B group
--bin_count : bin count for smoothing. default is 2
```

## Example usages
Example data and visualization result will be available after the manuscript becomes public.
