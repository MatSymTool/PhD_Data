#**************************************************************************************************************
# This program is designed to automatically calculate hydrogen content from Bruker G8 Galileo TDA measurements.
# Results are evaluated from .ERG files that are created when performung TDA measurements. The algorithm
# operates by averageing the first n elements and last n elements of the raw data and then plotting a linear
# function between them. This is then used as a reference point from which the raw integral data is subtracted
# for each point. Uncertainty is then determined as the difference between this data and the maximum possible
# integral. To achieve this, the absolute min value is found in the first and last n elements of the raw data.
# n can be specified by the user through the avg_var variable.
#**************************************************************************************************************
#AUTHOR: Philipp V Schulz (philipp.v.schulz@gmail.com); Based on idea by Dr. Dominik Dziedzic (UCL)
#DATE:   22/04/2024
#**************************************************************************************************************
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tkinter import filedialog as fd
import statistics
import datetime
import pathlib

axis_fontsize=15
label_fontsize=18

matplotlib.rc('xtick', labelsize=axis_fontsize)
matplotlib.rc('ytick', labelsize=axis_fontsize)


print('Select ERG file of ACTUAL test')
H_file = fd.askopenfilename()
print('(Optional) Select ERG file of REFERENCE run')
H_file_ref = fd.askopenfilename()
#file_path = fd.askdirectory(initialdir = os.path.dirname(os.path.realpath(__file__)))

ext = '.erg'    #.erg file contains all relevant data for analysis

Name_Str = ['Name','10']    #the number at the second index refers to the code in the .erg file -> change if needed
Comment_Str = ['Comment','14']
Time_Str = ['Testing Time','20 ']
weight_Str = ['Sample Weight','34']

TDA_ppm_Str = ['TDA ppm','730']
Factor_Str = ['Factor','748']
Integral_Str = ['Integral','750']

H_Data_Str = ['Uptick','192']
T_Data_Str = ['Temperature','190']

T_cont_ind = []     # Temperature of TDA device during test |   Size: Number of Time Steps per File
Med_T=[]            # Median TDA temperature per run |          Size: Number of Files
H_cont_ind = []     # H content per second |        Size: Number of Time Steps per File
H_cont_tot = []     # total hydrogen mass in ppm |  Size: Number of Files
H_cont_mass = []    # total hydrogen mass in g |    Size: Number of Files

ref_H_cont_ind = []     # H content per second reference run |        Size: Number of Time Steps per File
ref_H_cont_tot = []     # total hydrogen mass in ppm reference run |  Size: Number of Files
ref_H_cont_mass = []    # total hydrogen mass in g reference run |    Size: Number of Files

Comment = [] #                                      Size: Number of Files
Name = [] #                                         Size: Number of Files
Time = [] #                                         Size: Number of Files
weight = [] #                                       Size: Number of Files
TDA_ppm = [] #                                      Size: Number of Files
Factor = []         # integration factor from calibrated TDA |   Size: Number of Files
Integral_TDA = []   # integral from manual TDA calculation |   Size: Number of Files
uncertainty=[]        # maximum uncertainty between linear base function and constant base function between start and finish |   Size: Number of Files
Min_n = []          # list containing the minimum integral value of either first or last n items, which ever is smaller |   Size: Number of Files
ref_Min_n = []          # list containing the minimum integral value of either first or last n items, which ever is smaller for ref run |   Size: Number of Files


def Basefunc(x,y_init,y_end,length):    #linear function for calculating baseline which measured values are subtracted from
    return(y_init+(y_end-y_init)/(length-1)*x)


f = open(os.path.join(pathlib.Path(H_file).parent.resolve(),"TDA Evaluation Log.txt"), "w+")
f.write('MAXIMUM AMOUNT OF HYDROGEN IN SAMPLE:\n')

lin_base = 1        #Changes whether to use a linear function as baseline or starting value
avg_var = 75        # CHANGES NUMBER OF AVERAGED ELEMENTS, n AT BEGINNING AND #END OF TDA DATA

print('==============================================================================================')
print('MAXIMUM AMOUNT OF HYDROGEN IN SAMPLE')
print('----------------------------------------------------------------------------------------------')


if H_file.endswith(ext):
    file1 = open(H_file, 'r')
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
        elif item.startswith(TDA_ppm_Str[1]) and ' ' in item:
            TDA_ppm_i = float(item.split()[1])
            TDA_ppm.append(TDA_ppm_i)
        elif item.startswith(Factor_Str[1]) and ' ' in item:
            Factor_i = float(item.split()[1])
            Factor.append(Factor_i)
        elif item.startswith(Integral_Str[1]) and ' ' in item:
            Integral_TDA_i = item.split()[1]
            Integral_TDA.append(Integral_TDA_i)
        elif item.startswith(T_Data_Str[1]) and ' ' in item:  # following this line, H content data is given
            T_cont_col_tot = [] # List that contains raw integral data from H measurement
            i = idx + 1         # variable that goes through data
            num_vals = float(item.split()[1]) + i    #second half of that items correlates with number of lines which contain H content data
            while i < num_vals: # extracts H content data for the number of lines specifiec by num_vals
                T_cont_col_tot.append(float(Lines[i]))
                i = i+1
            T_cont_ind.append(T_cont_col_tot)   #saves list of Temperature in another list
            Med_T.append(np.median(T_cont_col_tot))
        elif item.startswith(H_Data_Str[1]) and ' ' in item:  # following this line, H content data is given
            H_cont_col_tot = [] # List that contains raw integral data from H measurement
            H_cont_col = []     # List that contains integral data after base is subtracted from raw data
            i = idx + 1         # variable that goes through data
            num_vals = float(item.split()[1]) + i    #second half of that items correlates with number of lines which contain H content data
            while i < num_vals: # extracts H content data for the number of lines specifiec by num_vals
                H_cont_col_tot.append(float(Lines[i]))
                i = i+1
            init_avg = statistics.mean(H_cont_col_tot[:+avg_var]) # average of first 50 elements to calculate base function
            end_avg = statistics.mean(H_cont_col_tot[-+avg_var:]) # average of last 50 elements to calculate base function
            Min_n.append(min(min(H_cont_col_tot[:+avg_var]),min(H_cont_col_tot[-+avg_var:])))      # find min value of first n or last n items in H content list
            #init_avg = H_cont_col_tot[0]                   # first element only to calculate base function
            #end_avg = H_cont_col_tot[-1]                   # last element only to calculate base function
            for idx,i in enumerate(H_cont_col_tot):
                Base_i = Basefunc(idx,init_avg,end_avg,len(H_cont_col_tot)) # baseline value is calculated for each H content value
                H_cont_col.append(i - Base_i if (i - Base_i >=0) else 0)    # difference between actual integral value and calculated baseline function. Returns 0 if value is negative

            H_cont_ind.append(H_cont_col)
    H_cont_ind[-1] = [i * Factor[-1]/weight[-1]*1e-6*0.05 for i in H_cont_ind[-1]]  #calculates H content per second from raw value by using factor
    H_cont_mass.append(sum(H_cont_ind[-1])*weight[-1])      #calculates total hydrogen mass in g
    H_cont_tot.append(H_cont_mass[-1]/weight[-1])           #calculates total hydrogen mass in ppm by dividing sample weight

    Lines_min = [a-Min_n[-1] for a in H_cont_col_tot]
    Lines_2_min = [i *Factor[-1]/weight[-1]*1e-6*0.05 for i in Lines_min]
    Hmass_min = sum(Lines_2_min)*weight[-1]
    H_ppm_min = Hmass_min/weight[-1]
    uncertainty.append(abs(H_ppm_min-H_cont_tot[-1]))

if H_file_ref.endswith(ext):
    file1 = open(H_file_ref, 'r')
    Lines = file1.read().splitlines()   #opens file and reads all lines into list

    for idx,item in enumerate(Lines):   #goes through content of .erg file to extract information such as weight, manual H content, Factor, etc.
        if item.startswith(H_Data_Str[1]) and ' ' in item:  # following this line, H content data is given
            ref_H_cont_col_tot = [] # List that contains raw integral data from H measurement
            ref_H_cont_col = []     # List that contains integral data after base is subtracted from raw data
            i = idx + 1         # variable that goes through data
            num_vals = float(item.split()[1]) + i    #second half of that items correlates with number of lines which contain H content data
            while i < num_vals: # extracts H content data for the number of lines specifiec by num_vals
                ref_H_cont_col_tot.append(float(Lines[i]))
                i = i+1
            init_avg = statistics.mean(H_cont_col_tot[:+avg_var]) if lin_base == 1 else H_cont_col_tot[0] # average of first n elements to calculate base function if linear function enabled, else first element
            end_avg = statistics.mean(H_cont_col_tot[-+avg_var:]) if lin_base == 1 else H_cont_col_tot[0] # average of last n elements to calculate base function if linear function enabled, else first element
            ref_Min_n.append(min(min(ref_H_cont_col_tot[:+avg_var]),min(ref_H_cont_col_tot[-+avg_var:])))      # find min value of first n or last n items in H content list
            for idx,i in enumerate(ref_H_cont_col_tot):
                Base_i = Basefunc(idx,init_avg,end_avg,len(ref_H_cont_col_tot)) # baseline value is calculated for each H content value
                ref_H_cont_col.append(i - Base_i if (i - Base_i >=0) else 0)    # difference between actual integral value and calculated baseline function. Returns 0 if value is negative
            ref_H_cont_ind.append(ref_H_cont_col)
    ref_H_cont_ind[-1] = [i * Factor[-1]/weight[-1]*1e-6*0.05 for i in ref_H_cont_ind[-1]]  #calculates H content per second from raw value by using factor
    ref_H_cont_mass.append(sum(ref_H_cont_ind[-1])*weight[-1])      #calculates total hydrogen mass in g
    ref_H_cont_tot.append(ref_H_cont_mass[-1]/weight[-1])           #calculates total hydrogen mass in ppm by dividing sample weight
    H_cont_tot[-1] = H_cont_tot[-1] - ref_H_cont_tot[-1]

#===========================================================================================================================================================
  #calculates uncertainty by making constant baseline function based on 1) first TDA value, 2) last TDA value
  # then determines maximum error by comparing following 2 scenarios:
  #1. maximum of subtracting max of 1st or last value
  #2. minimum of subtracting min of 1st or last value
  #for maximum of these scenarios, total H content is then calculated
  #This is then used to determine uncertainty from lin base

elif H_file_ref =="":   #condition if no reference file is selected
    ref_H_cont_tot.append(0)
    for i in range(len(H_cont_col_tot)):
        ref_H_cont_ind.append(0)
        i+=1
    uncertainty[-1] = np.sqrt(uncertainty[-1]**2 + ref_H_cont_tot[-1]**2)   #uncertainty is RSS of regular run uncertainty and reference run H content

#===========================================================================================================================================================
outputt = Name[-1]+' '+Comment[-1]+': '+str(TDA_ppm[-1])+' (manual) -> '+str(np.round(H_cont_tot[-1],5))+' \u00B1 '+str(np.round(uncertainty[-1],3))+' ppm (automatic)'
f.write(outputt+'\n')
print(outputt)


print('==============================================================================================')

f.write('------------------------------------------')
f.write('\nName\n')
f.write(str(Name)+'\n')
f.write('Comment\n')
f.write(str(Comment)+'\n')
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
f.write('Min Integral Value\n')
f.write(str(Min_n)+'\n')
f.write('Uncertainty [ppm]\n')
f.write(str(uncertainty)+'\n')
f.write('Median TDA Temperature [°C]\n')
f.write(str(Med_T)+'\n')
f.write('\nDate and time of evaluation:\n')
f.write(str(datetime.datetime.now()))
f.close()




#fig, (ax1,ax2) = plt.subplots(2,1)
fig, ax1 = plt.subplots()
fig.set_figwidth(12)
fig.set_figheight(8)

for i in range(len(weight)):
    if H_file_ref !="":
        H_cont_ind[i] = [max(a - b, 0) for a, b in zip(H_cont_ind[i], ref_H_cont_ind[i])]  #subtracts reference run profile from actual run
    x = [x for x in range(len(H_cont_ind[i]))]
    x = [l * 0.2/60 for l in x]
    lns1=ax1.plot(x,H_cont_ind[i],label = Name[i]+ ' ' +Comment[i]+' ('+f'{H_cont_tot[i]:.3f}' +' \u00B1 '+str(np.round(uncertainty[i],3))+ ' ppm)',linewidth = 1.5,color='C0')
    ax1.set_ylabel('Hydrogen [ppm/s]',fontsize=label_fontsize-2)
    ax1.set_xlabel('Time [min]',fontsize=label_fontsize-2)

    ax2 = ax1.twinx()
    lns2 = ax2.plot(x,T_cont_ind[i],label = 'Temperature',color='C1')
    ax2.set_ylabel('TDA Temperature [°C]',fontsize=label_fontsize-2)

    lns = lns1+lns2
    labs = [l.get_label() for l in lns]
    ax2.legend(lns, labs, loc=0,fontsize=label_fontsize-2).set_draggable(True)

#plt.savefig(os.path.join(pathlib.Path(H_file).parent.resolve(),'Diffusible TDA Plot plus T.pdf'),bbox_inches='tight')
plt.show()
