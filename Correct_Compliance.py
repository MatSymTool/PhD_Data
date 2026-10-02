##======================================================================================================================================
#Based on the MATLAB script by Christopher Finfrock
#https://www.researchgate.net/post/How-to-do-compliance-correction-for-a-given-stress-strain-data
##======================================================================================================================================
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, Button
import math
import pandas as pd
from tkinter import filedialog as fd


# Load data
print('Enter CSV File Path (Disp in Col 1, Force in Col 2)')
file_path = fd.askopenfilename(initialdir = os.path.dirname(os.path.realpath(__file__)))

mycols = names=['a', 'b']
datas = pd.read_csv(file_path,names=mycols)

Disp_mm= datas['a']
Force_N= datas['b']

#=================================================================
# USER INPUTS
#=================================================================
theoretical_elastic_modulus_MPa = 200000  # for IN718
initial_area_mm2 = 4 * 0.27
initial_length_mm = 6
#=================================================================


# Create a flag to check if the rectangle is drawn
rectangle_drawn = False

# Plot load-displacement data
fig, ax = plt.subplots()
ax.plot(Disp_mm, Force_N, 'k')
ax.set(xlabel='Displacement [mm]', ylabel='Force [N]',title='Select Elastic Region')

# Create a RectangleSelector
def onselect(eclick, erelease):
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata
    ax.add_patch(plt.Rectangle((min(x1, x2), min(y1, y2)), abs(x2 - x1), abs(y2 - y1),
                               fill=None, edgecolor='r', linestyle='dashed', linewidth=2))
    fig.canvas.draw()
    global rectangle_drawn
    rectangle_drawn = True

rs = RectangleSelector(ax, onselect, useblit=True, button=[1], minspanx=5, minspany=5,
                       spancoords='pixels', interactive=True)

# Press button to continue
def on_button_click(event):
    if rectangle_drawn:
        plt.close()

button_ax = plt.axes([0.85, 0.02, 0.1, 0.04])
button = Button(button_ax, 'Continue')
button.on_clicked(on_button_click)

plt.show()

# Extract data in the selected region
roi_xmin, roi_ymin, roi_width, roi_height = rs.extents
in_roi = (Disp_mm >= roi_xmin) & (Disp_mm <= roi_xmin + roi_width) & (Force_N >= roi_ymin) & (Force_N <= roi_ymin + roi_height)
Disp_mm_elastic = Disp_mm[in_roi]
Force_N_elastic = Force_N[in_roi]

# Calculate effective stiffness based on elastic region drawn in plot
k_effective = np.polyfit(Disp_mm_elastic, Force_N_elastic, 1)

#stiffness is slope of linear plot -> first element of k_effective
k_effective = k_effective[0]

#calculate stiffness of sample through k = E*A/L0
k_sample = (theoretical_elastic_modulus_MPa * initial_area_mm2) / initial_length_mm  # N/mm

# Use springs-in-series reciprocal addition relationship to determine stiffness of frame
k_loadframe = 1 / ((1 / k_effective) - (1 / k_sample))

#correct displacement by subtracting frame displacement from measured displacement
Disp_mm_corr = Disp_mm - (Force_N / k_loadframe)
#sets the initial displacement value of the corrected data to zero
Disp_mm_corr = Disp_mm_corr - Disp_mm_corr[0]

# Center displacement start at zero for the corrected data
start_load = Force_N[0]
start_disp = start_load / k_sample # initial displacement value for the corrected data based on the initial force value and the sample compliance
Disp_mm_corr = Disp_mm_corr + start_disp

#User entry for maximum force where grips are estimated to be elastically deformed (after which no more deformation occurs)-> used to offset stiffness-corrected data by that length
#-> it is necessary to first correct data for stiffness, then by offset, as sample has same stiffness in grips section as in gage section
Force_grips = float(input("Enter last force value where elastic adjustment in grips occured (in N): "))
elast_force = np.where(Force_N < Force_grips)[0] #equates the entered force with all stored force array elements from the input file that are lower than the entered value
corr_disp_max = Disp_mm_corr[elast_force[-1]]   #Determines the corrected displacement corresponding to the entered force value based on the last force array index

corr_elast_grips = np.array([])

for i in range(len(Disp_mm_corr)):
    if i < elast_force[-1]:         #Special routine to  the actual force where it is lower than the maximum elastic grip force -> as the offset from the max force would be too high in that case
        stress = Force_N[i]/initial_area_mm2
        corr_disp = Disp_mm_corr[i] # The displacement in that case is also taken from the already corrected displacement
    else:
        stress = Force_grips/initial_area_mm2
        corr_disp = corr_disp_max
   # eng_str = stress/theoretical_elastic_modulus_MPa    #it is assumed that young's modulus yields engineering strain, not log strain
   # log_disp = math.log(1+eng_str)*initial_length_mm    # log displacement calculated through: delta = epsilon_true * L_0 = ln(1+e)*L_0

    eng_str = (stress + 2*Force_N[i]/(8*0.27))/theoretical_elastic_modulus_MPa
    log_disp = math.log(1+eng_str)*10

    offset = corr_disp-log_disp                         # Offset is difference of corrected displacement and theoretical log displacement
    gage_disp = Disp_mm_corr[i]-offset                  # calculation of displacement only in the gage section by subtracting offset from corrected displacement
    corr_elast_grips =  np.append(corr_elast_grips,gage_disp)

print('====================== MEASURED =====================')
print('max displacement: '+str(np.round(Disp_mm[len(Disp_mm)-1],2))+' mm')
print('max strain: '+str(np.round(np.log(1 + Disp_mm[len(Disp_mm)-1] / initial_length_mm) * 100,2))+' %')
print('=============== STIFFNESS-CORRECTED =================')
print('max displacement: '+str(np.round(Disp_mm_corr[len(Disp_mm_corr)-1],2))+' mm')
print('max strain: '+str(np.round(np.log(1 + Disp_mm_corr[len(Disp_mm_corr)-1] / initial_length_mm) * 100,2))+' %')
print('================= GRIPS-CORRECTED ===================')
print('max displacement: '+str(np.round(corr_elast_grips[len(corr_elast_grips)-1],2))+' mm')
print('max strain: '+str(np.round(np.log(1 + corr_elast_grips[len(corr_elast_grips)-1] / initial_length_mm) * 100,2))+' %')
print("================= YOUNG's MODULUS ===================")
print('Sample (theoretical): '+str(np.round(theoretical_elastic_modulus_MPa/1000,2))+' GPa')
print('Total: '+str(np.round(k_effective*initial_length_mm/initial_area_mm2/1000,2))+' GPa')
print('Rig: '+str(np.round(k_loadframe*initial_length_mm/initial_area_mm2/1000))+' GPa')
print('==================== STIFFNESS ======================')
print('Sample (theoretical): '+str(np.round(k_sample,2))+' N/mm')
print('Total: '+str(np.round(k_effective,2))+' N/mm')
print('Rig: '+str(np.round(k_loadframe))+' N/mm')
print('=====================================================')

# Plot the corrected data on the raw data

#------------------------
PlotStress = 1 # set to 1 to plot stress, set to 0 to plot force
#------------------------


strain_effective = np.log(1 + Disp_mm / initial_length_mm) * 100
strain_corrected = np.log(1 + Disp_mm_corr / initial_length_mm) * 100
strain_grips_corrected = np.log(1 + corr_elast_grips / initial_length_mm) * 100

plt.figure(figsize=(8, 4))
plt.figure(1)
if PlotStress == 1:
    Force_N = Force_N/initial_area_mm2
    Force_N_elastic = Force_N_elastic/initial_area_mm2
    label = 'Stress [MPa]'
else:
    label = 'Force [N]'


result = pd.DataFrame({
    label: Force_N,
    'Raw Elongation [mm]': Disp_mm,
    'Stiffness-Corrected Elongation [mm]': Disp_mm_corr,
    'Grips-Corrected Elongation [mm]': corr_elast_grips,
    'Raw True Strain [%]': strain_effective,
    'Stiffness-Corrected True Strain [%]': strain_corrected,
    'Grips-Corrected True Strain [%]': strain_grips_corrected
})
# Save the result to a new CSV file
result.to_csv(os.path.join(os.path.dirname(file_path), 'Correction Results.csv'), index=False)


plt.plot(Disp_mm, Force_N, 'k',label='Raw Data')
plt.plot(Disp_mm_corr, Force_N, 'b',label='Stiffness-Corrected')
plt.plot(corr_elast_grips, Force_N, 'g',label='Grips-Corrected')
plt.plot(Disp_mm_elastic, Force_N_elastic, 'r',label='Elastic Region')
#plt.plot(Disp_mm[elast_force[-1]],Force_N[elast_force[-1]],'co',label='Max Elastic Grips Force', linewidth=3) # plot entered point where last adjustment of sample in grips occurred
plt.legend(loc='lower right')
plt.xlim([-0.01, max(Disp_mm) + 0.2])
plt.ylim([0, max(Force_N) * 1.05])
plt.xlabel('Displacement [mm]')
plt.ylabel(label)
plt.savefig('Stress-Displacement_corrected.pdf')
plt.show(block=False) #this line and the next allow the code from conitnuing after plotting -> allows two plots to be open at same time -> needs additional plot.show() at very end, i.e., after calling the functions from the main menu
plt.pause(0.001)

# Plot the corrected strain on the raw data
# Set the figure size
plt.figure(figsize=(8, 4))
plt.figure(2)
plt.plot(strain_effective, Force_N, 'k',label='Raw Data')
plt.plot(strain_corrected, Force_N, 'b',label='Stiffness-Corrected')
plt.plot(strain_grips_corrected, Force_N, 'g',label='Grips-Corrected')
#plt.plot(np.log(1 + Disp_mm[elast_force[-1]] / initial_length_mm) * 100,Force_N[elast_force[-1]],'co',label='Max Elastic Grips Force', linewidth=3)    # plot entered point where last adjustment of sample in grips occurred
plt.legend(loc='lower right')
plt.xlim([-0.01, max(strain_effective) * 1.05])
plt.ylim([0, max(Force_N) * 1.05])
plt.xlabel('True Strain [%]')
plt.ylabel(label)
plt.savefig('Stress-Strain_corrected.pdf')
plt.show(block=False) #this line and the next allow the code from conitnuing after plotting -> allows two plots to be open at same time -> needs additional plot.show() at very end, i.e., after calling the functions from the main menu
plt.pause(0.001)
#plt.grid()
plt.show()
