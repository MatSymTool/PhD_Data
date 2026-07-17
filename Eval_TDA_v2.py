#**************************************************************************************************************
# This program is designed to automatically calculate hydrogen content from Bruker G8 Galileo TDA measurements.
# Results are evaluated from .ERG files that are created when performung TDA measurements. The algorithm
# operates by constructing a linear function between the first and last point of the peak region. Uncertainty 
# is then determined as the difference between this data and the maximum possible integral. 
#**************************************************************************************************************
#AUTHOR: Philipp V Schulz (ucempvs@ucl.ac.uk)
#DATE:   30/03/2026
#**************************************************************************************************************
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tkinter import filedialog as fd
import numpy
import datetime
import csv

axis_fontsize=15
label_fontsize=18

matplotlib.rc('xtick', labelsize=axis_fontsize)
matplotlib.rc('ytick', labelsize=axis_fontsize)

print('Enter folder path of TDA .erg data')
file_path = fd.askdirectory()
#file_path = r"C:\Users\ucempvs.AD\OneDrive - University College London\Desktop"
ext = '.erg'    #.erg file contains all relevant data for analysis

Name_Str = ['Name','10']    #the number at the second index refers to the code in the .erg file -> change if needed
Comment_Str = ['Comment','14']
Time_Str = ['Testing Time','20 ']
Date_Str = ['Test Date','4 ']
weight_Str = ['Sample Weight','34']

TDA_ppm_Str = ['TDA ppm','730']
Factor_Str = ['Factor','748']
Integral_Str = ['Integral','750']

H_Data_Str = ['Uptick','192']
H_Data_High_Str = ['Uptick','193']  #H high detector

H_cont_ind = []     # H content per second |        Size: Number of Time Steps per File
H_cont_peak = []    # H content per seconds for peak region only
H_cont_tot = []     # total hydrogen mass in ppm |  Size: Number of Files
H_cont_mass = []    # total hydrogen mass in g |    Size: Number of Files
Comment = [] #                                      Size: Number of Files
Name = [] #                                         Size: Number of Files
Time = [] #                                         Size: Number of Files
Datee = [] #                                         Size: Number of Files
weight = [] #                                       Size: Number of Files
TDA_ppm = [] #                                      Size: Number of Files
Bases = []          # list of where the peak values start and end for each file
Factor = []         # integration factor from calibrated TDA |   Size: Number of Files
Integral_TDA = []   # integral from manual TDA calculation |   Size: Number of Files
uncertainty=[]        # maximum uncertainty between linear base function and constant base function between start and finish |   Size: Number of Files
Min_n = []          # list containing the minimum integral value of either first or last n items, which ever is smaller |   Size: Number of Files

#===========================================================================================================================================================
# USER-DEFINED VARIABLES
#============================================================================================================================================================
avg_window = 30         # window for moving average
TDA_step = 0.2          # 1 data point every 0.2 s
time_0_grad = 10        # number of seconds gradient has to be 0 to be counted as end of peak region
beginn_STDEV = 2        # multiplier of standard deviation that raw values need to exceed to trigger beginning of peak region
beginning_region = 10   # number of seconds for which standard deviation is calculated as criterium for finding beginning of peak region
look_ahead_time = 50    # seconds to look ahead after a candidate end to check for continued descent
descent_fraction = 0.05  # if signal drops more than this fraction of peak amplitude in look-ahead window, candidate is a mid-peak plateau (not true end)
debug = False            # set to True to show per-file diagnostic plot with all baseline regions and peak detection points


def movingaverage(data, window_size):
    window= numpy.ones(int(window_size))/float(window_size)
    return numpy.convolve(data, window, 'same')

def find_consecutive_starts(nums,x,peak):   #find where a consecutive number of elements of length X starts -> for finding end of peak region
    if x < 1: return []
    
    return next((nums[i] for i in range(len(nums) - x + 1) 
            if i > peak and all(nums[i+j] == nums[i] + j for j in range(x))),None)  #only triggers after the peak

def Basefunc(x,y_init,y_end,length):    #linear function for calculating baseline which measured values are subtracted from
    return(y_init+(y_end-y_init)/(length-1)*x)

f = open(os.path.join(file_path,"TDA Evaluation Log.txt"), "w+")
f.write('MAXIMUM AMOUNT OF HYDROGEN IN SAMPLE:\n')

print('==============================================================================================')
print('MAXIMUM AMOUNT OF HYDROGEN IN SAMPLE')
print('----------------------------------------------------------------------------------------------')

for files in os.listdir(file_path): #loops through all files in folder that have .erg extension
    if files.endswith(ext):
        file1 = open(os.path.join(file_path,files), 'r')
        Lines = file1.read().splitlines()   #opens file and reads all lines into list

        for idx,item in enumerate(Lines):   #goes through content of .erg file to extract information such as weight, manual H content, Factor, etc.
            if item.startswith(Name_Str[1]) and ' ' in item:  # checks if line contains a space -> if not, it is likely a data line and therefore a false pos
                Name_i = item.split(' ', 1)[1] #splits off number from string but leaves rest of string unchanged, since name might contain a space
                Name.append(Name_i)
            elif item.startswith(Comment_Str[1]) and ' ' in item:
                Comment_i = item.split(' ', 1)[1]
                Comment.append(Comment_i)
            elif item.startswith(Time_Str[1]) and ' ' in item:
                Time_i = float(item.split()[1])
                Time.append(Time_i)   #time will not contain a space and therefore the string can be split regularly
            elif item.startswith(weight_Str[1]) and ' ' in item:
                weight_i = float(item.split()[1])
                weight.append(weight_i)
            elif item.startswith(Date_Str[1]) and ' ' in item:
                Date_i = item.split()[1]
                Datee.append(Date_i)
            elif item.startswith(TDA_ppm_Str[1]) and ' ' in item:
                TDA_ppm_i = float(item.split()[1])
                TDA_ppm.append(TDA_ppm_i)
            elif item.startswith(Factor_Str[1]) and ' ' in item:
                Factor_i = float(item.split()[1])
                Factor.append(Factor_i)
            elif item.startswith(Integral_Str[1]) and ' ' in item:
                Integral_TDA_i = item.split()[1]
                Integral_TDA.append(Integral_TDA_i)
            if item.startswith(H_Data_Str[1])  and ' ' in item or item.startswith(H_Data_High_Str[1]) and ' ' in item:  # following this line, H content data is given; also checks if high H detector picked up something
                H_cont_col_raw = [] # List that contains raw integral data from H measurement
                H_cont_col = []     # List that contains integral data after base is subtracted from raw data
                #H_cont_col_avg = [] # List that contains raw integral data from H measurements as moving average
                i = idx + 1         # variable that goes through data
                num_vals = float(item.split()[1]) + i    #second half of that items correlates with number of lines which contain H content data
                while i < num_vals: # extracts H content data for the number of lines specifiec by num_vals
                    H_cont_col_raw.append(float(Lines[i]))
                    i = i+1
                
                #H_cont_col_avg = movingaverage(H_cont_col_raw,avg_window) # moving average of raw H content data with window size of 10; this is used for plotting and not for calculations to avoid smoothing out peaks too much
                #H_cont_col_grad = np.gradient(H_cont_col_avg)
                H_cont_col_grad = np.gradient(H_cont_col_raw)

                indices_in_range = [    #find where the gradient of the H release curve is 0
                    index for index, value in enumerate(H_cont_col_grad)
                    #if abs(value  <= 0.025) #-> something where the graph is offset in y-direction by some amount
                    if value == 0
                ]
                peak_val = H_cont_col_raw.index(max(H_cont_col_raw))  #find index where peak H release is
                beginning_std = np.std(H_cont_col_raw[:int(beginning_region/TDA_step)],ddof=1)    #first x seconds are used to calculate Standard dev
                baseline_mean = np.mean(H_cont_col_raw[:int(beginning_region/TDA_step)])
                peak_amplitude = max(H_cont_col_raw) - baseline_mean

                # Find end: iterate over consecutive zero-gradient runs after the peak.
                # After each candidate, look ahead look_ahead_time seconds: if the signal
                # continues to drop by more than descent_fraction * peak_amplitude the
                # candidate is a mid-peak plateau (Campaign 2 style) — skip past it and
                # keep searching. A flat or noisy tail (Campaign 3 style) is accepted
                # immediately. Falls back to last data point if no run is found at all.
                x = int(time_0_grad / TDA_step)
                la = int(look_ahead_time / TDA_step)
                search_from = peak_val
                end = None
                while end is None:
                    candidate = None
                    for k in range(len(indices_in_range) - x + 1):
                        if indices_in_range[k] > search_from:
                            if all(indices_in_range[k+j] == indices_in_range[k] + j for j in range(x)):
                                candidate = indices_in_range[k]
                                break
                    if candidate is None:
                        end = len(H_cont_col_raw) - 1  # no run found; use last point
                        break
                    look_end = min(candidate + la, len(H_cont_col_raw) - 1)
                    delta = H_cont_col_raw[look_end] - H_cont_col_raw[candidate]
                    if delta > -descent_fraction * peak_amplitude:
                        end = candidate  # signal flat/noisy after candidate → true end
                    else:
                        search_from = candidate + x  # signal still descending → plateau, skip

                beginning = int(beginning_region/TDA_step)  # fallback: start of search window
                for j in range(len(H_cont_col_raw[int(beginning_region/TDA_step):])): #beginning of peak is where running avg is higher than n * standard deviation of mean in first x s
                    if H_cont_col_raw[int(beginning_region/TDA_step):][j] > np.mean(H_cont_col_raw[:int(beginning_region/TDA_step)])+beginning_std*beginn_STDEV:
                        beginning = j+int(beginning_region/TDA_step)
                        break
                                
                H_cont_col_peak = H_cont_col_raw[beginning:end+1] #list of overall TDA data cropped to peak

                for idx,i in enumerate(H_cont_col_raw):
                    if idx < beginning:
                        Base_i = Basefunc(idx,H_cont_col_raw[0],H_cont_col_raw[beginning-1],len(H_cont_col_raw[:beginning]))
                        H_cont_col.append(i - Base_i )

                    elif idx >= beginning and idx <= end+1:    
                        Base_i = Basefunc(idx,H_cont_col_peak[0],H_cont_col_peak[-1],len(H_cont_col_peak))
                        H_cont_col.append(i - Base_i )    # difference between actual integral value and calculated baseline function.
                    else:
                        Base_i = Basefunc(idx,H_cont_col_raw[end+2],H_cont_col_raw[-1],len(H_cont_col_raw[end+2:]))
                        H_cont_col.append(i - Base_i )

                H_cont_ind.append(H_cont_col)
                H_cont_peak.append(H_cont_col[beginning:end+1])
                Bases.append([beginning,end])

        H_cont_peak[-1] = [i * Factor[-1]/weight[-1]*1e-6*0.05 for i in H_cont_peak[-1]]  #calculates H content per time step (0.2 s) from raw value by using factor
        H_cont_peak[-1] = [value / TDA_step for value in H_cont_peak[-1]]                 #converts H content from ppm/(0.2 s) to ppm/s
        H_cont_mass.append(sum(H_cont_peak[-1])*TDA_step*weight[-1])      #calculates total hydrogen mass in g
        H_cont_tot.append(H_cont_mass[-1]/weight[-1])           #calculates total hydrogen mass in ppm by dividing sample weight

#===========================================================================================================================================================
  #calculates uncertainty by making constant baseline function based on 1) first TDA value, 2) last TDA value
  # then determines maximum error by comparing following 2 scenarios:
  # (1): using beginning value of peak region as CONSTANT baseline across the peak region
  # (2): using value of end of peak region as CONSTANT baseline across the peak region
  #for maximum of these scenarios, total H content is then calculated
  #This is then used to determine uncertainty from lin base

        #case 1: using horizontal baseline from beginning
        Base_i = H_cont_col_raw[Bases[-1][0]]   # value of raw data at beginning of peak region
        Lines_min = [a-Base_i for a in H_cont_col_raw[Bases[-1][0]:Bases[-1][-1]+1]]
        Lines_min = [i *Factor[-1]/weight[-1]*1e-6*0.05 for i in Lines_min]
        Hmass_min = sum(Lines_min)*weight[-1]
        H_ppm_1 = Hmass_min/weight[-1]

        #case 2: using horizontal baseline from beginning
        Base_i = H_cont_col_raw[Bases[-1][-1]]   # value of raw data at beginning of peak region
        Lines_min = [a-Base_i for a in H_cont_col_raw[Bases[-1][0]:Bases[-1][-1]+1]]
        Lines_min = [i *Factor[-1]/weight[-1]*1e-6*0.05 for i in Lines_min]
        Hmass_min = sum(Lines_min)*weight[-1]
        H_ppm_2 = Hmass_min/weight[-1]

        uncertainty.append(abs(H_ppm_1-H_cont_tot[-1]) if H_ppm_1 > H_ppm_2 else abs(H_ppm_2-H_cont_tot[-1]))

#===========================================================================================================================================================
# DEBUG PLOT
#===========================================================================================================================================================
        if debug:
            time_axis_raw = [k * TDA_step for k in range(len(H_cont_col_raw))]

            # Reconstruct the actual baseline that was subtracted (raw - corrected = baseline)
            baseline_actual = [r - c for r, c in zip(H_cont_col_raw, H_cont_col)]

            # Uncertainty horizontal baselines over peak region
            peak_time = [k * TDA_step for k in range(beginning, end + 1)]
            unc_base1 = [H_cont_col_raw[Bases[-1][0]]] * (end - beginning + 1)   # constant at peak start
            unc_base2 = [H_cont_col_raw[Bases[-1][-1]]] * (end - beginning + 1)  # constant at peak end

            # Mean and threshold used for peak-start detection
            mean_begin = np.mean(H_cont_col_raw[:int(beginning_region / TDA_step)])
            threshold = mean_begin + beginning_std * beginn_STDEV

            # Consecutive-zero region that triggered end detection (in time)
            consec_start_t = (end - int(time_0_grad / TDA_step) + 1) * TDA_step
            consec_end_t   = end * TDA_step

            fig_d, (ax1, ax2) = plt.subplots(
                2, 1, figsize=(14, 8), sharex=True,
                gridspec_kw={'height_ratios': [3, 1]}
            )
            fig_d.suptitle(
                f'Debug — {Name[-1]} {Comment[-1]}  |  '
                f'{H_cont_tot[-1]:.3f} ± {uncertainty[-1]:.3f} ppm',
                fontsize=label_fontsize
            )

            # ── Top panel: raw data + all baselines ─────────────────────────────────
            ax1.plot(time_axis_raw, H_cont_col_raw,
                     color='black', linewidth=1.2, label='Raw data', zorder=5)

            # Actual subtracted baseline (three segments combined)
            ax1.plot(time_axis_raw, baseline_actual,
                     color='gray', linewidth=1.2, linestyle='--',
                     label='Subtracted baseline (actual)', zorder=4)

            # Shade the beginning region used for std dev
            ax1.axvspan(0, beginning_region, alpha=0.12, color='royalblue',
                        label=f'Std dev window (0 – {beginning_region} s)')

            # Mean and threshold of beginning region
            ax1.axhline(mean_begin, color='royalblue', linestyle=':', linewidth=1.0, alpha=0.7,
                        label=f'Mean of begin. region ({mean_begin:.3f})')
            ax1.axhline(threshold, color='royalblue', linestyle='--', linewidth=1.2,
                        label=f'Peak-start threshold (mean + {beginn_STDEV}σ = {threshold:.3f})')

            # Uncertainty baselines (horizontal) over peak region
            ax1.plot(peak_time, unc_base1,
                     color='firebrick', linestyle=':', linewidth=1.3,
                     label=f'Unc. baseline 1 — const. at peak start ({H_cont_col_raw[beginning]:.3f})', zorder=3)
            ax1.plot(peak_time, unc_base2,
                     color='darkred', linestyle=':', linewidth=1.3,
                     label=f'Unc. baseline 2 — const. at peak end ({H_cont_col_raw[end]:.3f})', zorder=3)

            # Vertical lines for peak start / end / maximum
            ax1.axvline(beginning * TDA_step, color='limegreen', linewidth=1.8, linestyle='-',
                        label=f'Peak start  t = {beginning * TDA_step:.1f} s  (idx {beginning})')
            ax1.axvline(end * TDA_step, color='tomato', linewidth=1.8, linestyle='-',
                        label=f'Peak end    t = {end * TDA_step:.1f} s  (idx {end})')
            ax1.axvline(peak_val * TDA_step, color='darkorange', linewidth=1.4, linestyle='-.',
                        label=f'Peak maximum  t = {peak_val * TDA_step:.1f} s  (idx {peak_val})')

            # Scatter markers on the raw curve at key points
            ax1.scatter([beginning * TDA_step], [H_cont_col_raw[beginning]],
                        color='limegreen', s=70, zorder=6, marker='o')
            ax1.scatter([end * TDA_step], [H_cont_col_raw[end]],
                        color='tomato', s=70, zorder=6, marker='o')
            ax1.scatter([peak_val * TDA_step], [H_cont_col_raw[peak_val]],
                        color='darkorange', s=120, zorder=6, marker='*')

            ax1.set_ylabel('Raw H signal [a.u.]', fontsize=label_fontsize - 2)
            ax1.legend(fontsize=7.5, loc='upper right', ncol=2).set_draggable(True)

            # ── Bottom panel: gradient ───────────────────────────────────────────────
            ax2.plot(time_axis_raw, H_cont_col_grad,
                     color='steelblue', linewidth=1.0, label='d(raw)/dt (gradient)', zorder=3)
            ax2.axhline(0, color='black', linewidth=0.8, linestyle='--', label='Zero')

            # Shade the consecutive-zero block that marked the end of the peak
            ax2.axvspan(consec_start_t, consec_end_t, alpha=0.25, color='tomato',
                        label=f'Consecutive-zero trigger ({time_0_grad} s window)')

            ax2.axvline(beginning * TDA_step, color='limegreen', linewidth=1.8, linestyle='-')
            ax2.axvline(end * TDA_step,       color='tomato',    linewidth=1.8, linestyle='-')
            ax2.axvline(peak_val * TDA_step,  color='darkorange', linewidth=1.4, linestyle='-.')

            ax2.set_xlabel('Time [s]', fontsize=label_fontsize - 2)
            ax2.set_ylabel('Gradient [a.u./s]', fontsize=label_fontsize - 2)
            ax2.legend(fontsize=7.5).set_draggable(True)

            plt.tight_layout()
            plt.show()

#===========================================================================================================================================================
        outputt = Name[-1]+' '+Comment[-1]+': '+str(TDA_ppm[-1])+' (manual) -> '+str(np.round(H_cont_tot[-1],3))+' \u00B1 '+str(np.round(uncertainty[-1],2))+' ppm (automatic) ('+Datee[-1]+')'
        f.write(outputt+'\n')
        print(outputt)

print('==============================================================================================')

f.write('------------------------------------------')
f.write('\nName\n')
f.write(str(Name)+'\n')
f.write('Comment\n')
f.write(str(Comment)+'\n')
f.write('Date\n')
f.write(str(Datee)+'\n')
f.write('Weight [g]:\n')
f.write(str(weight)+'\n')
f.write('Factor:\n')
f.write(str(Factor)+'\n')
f.write('Manual H Content [ppm]\n')
f.write(str(TDA_ppm)+'\n')
f.write('H mass [µg]:\n')
f.write(str(H_cont_mass)+'\n')
f.write('H content [ppm]:\n')
f.write(str(H_cont_tot)+'\n')
f.write('Uncertainty [ppm]\n')
f.write(str(uncertainty)+'\n')
f.write('\nDate and time of evaluation:\n')
f.write(str(datetime.datetime.now()))
f.close()
fig, ax = plt.subplots()

fig.set_figwidth(10)
fig.set_figheight(5)

for i in range(len(weight)):
    x = [x for x in range(len(H_cont_ind[i]))]
    x = [l * TDA_step for l in x]
    
    #Optional: export ppm data to CSV
    
    #with open(os.path.join(file_path,Name[i]+'_'+Comment[i]+'_data'+'.csv'),'w',newline='') as csvfile:
    #    writer = csv.writer(csvfile)
    #    writer.writerow(('Time (s)','H Release (ppm/s)'))
     #   writer.writerows(zip(x,H_cont_ind[i]))
    
    plt.plot(x,H_cont_ind[i],label = Name[i]+ ' ' +Comment[i]+' ('+f'{H_cont_tot[i]:.3f}' +' \u00B1 '+str(np.round(uncertainty[i],2))+ ' ppm)',linewidth = 1.5)
    #plt.axvline(Bases[i][0]*TDA_step,color="orange",label = "Peak beginning",linestyle = '--')
    #plt.axvline(Bases[i][-1]*TDA_step,color="red",label = "Peak end",linestyle = '--')
    
    plt.legend(fontsize=label_fontsize-5).set_draggable(True)
    plt.xlabel('Time [s]',fontsize=label_fontsize-2)
    plt.ylabel('Hydrogen [ppm/s]',fontsize=label_fontsize-2)
#plt.savefig(os.path.join(file_path,'TDA Plot.pdf'),bbox_inches='tight')
plt.show()


