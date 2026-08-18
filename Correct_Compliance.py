##======================================================================================================================================
# Based on the MATLAB script by Christopher Finfrock
# https://www.researchgate.net/post/How-to-do-compliance-correction-for-a-given-stress-strain-data
##======================================================================================================================================
import os
import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, Button
from tkinter import filedialog as fd
from sklearn.linear_model import LinearRegression

# =================================================================
# USER INPUTS
# =================================================================
theoretical_elastic_modulus_MPa = 200000
E_rel_uncertainty = 0.10
initial_area_mm2 = 4 * 0.27
initial_length_mm = 8

FORCE_GRIPS_WINDOW = (90, 130)
REQUIRE_STIFFENING = True
PLASTIC_THRESH = 0.002
PlotStress = 1
# =================================================================

print('Select CSV File (Disp in 2nd Col, Force in 3rd Col, Data starts from row 21)')
file_path = fd.askopenfilename()
if not file_path:
    raise SystemExit('No file selected.')

datas = pd.read_csv(file_path, usecols=[1, 2], skiprows=20, header=None,
                    names=['Disp_mm', 'Force_N'])

Disp_mm = datas['Disp_mm'].to_numpy(dtype=float)
Force_N = datas['Force_N'].to_numpy(dtype=float)

sigma_eng = Force_N / initial_area_mm2
eps_e = sigma_eng / theoretical_elastic_modulus_MPa
k_sample = (theoretical_elastic_modulus_MPa * initial_area_mm2) / initial_length_mm

Disp_full = Disp_mm.copy()

# -----------------------------------------------------------------
# Grip settling threshold
# -----------------------------------------------------------------
low, high = FORCE_GRIPS_WINDOW
mask_win = (Force_N >= low) & (Force_N <= high)
if mask_win.sum() < 8:
    raise RuntimeError('Fewer than 8 points inside FORCE_GRIPS_WINDOW.')

disp_range = Disp_mm[mask_win]
force_range = Force_N[mask_win]

best = None
min_residual_sum = np.inf
for i in range(3, len(force_range) - 3):
    m1 = LinearRegression().fit(disp_range[:i].reshape(-1, 1), force_range[:i])
    m2 = LinearRegression().fit(disp_range[i:].reshape(-1, 1), force_range[i:])
    s1, c1 = float(m1.coef_[0]), float(m1.intercept_)
    s2, c2 = float(m2.coef_[0]), float(m2.intercept_)

    if REQUIRE_STIFFENING and not (s2 > s1):
        continue
    if abs(s1 - s2) < 1e-9:
        continue

    r1 = force_range[:i] - m1.predict(disp_range[:i].reshape(-1, 1))
    r2 = force_range[i:] - m2.predict(disp_range[i:].reshape(-1, 1))
    residual_sum = np.sum(r1 ** 2) + np.sum(r2 ** 2)

    if residual_sum < min_residual_sum:
        min_residual_sum = residual_sum
        best = (s1, c1, s2, c2)

if best is None:
    raise RuntimeError('No valid two-segment split found in FORCE_GRIPS_WINDOW.')

s1, c1, s2, c2 = best
Force_grips = s1 * ((c2 - c1) / (s1 - s2)) + c1

if not (low <= Force_grips <= high):
    raise RuntimeError(f'Fitted Force_grips = {Force_grips:.1f} N lies outside the search window.')

above = np.where(Force_N >= Force_grips)[0]
if above.size == 0:
    raise RuntimeError('No data above Force_grips.')
start_idx = int(above[0])

# -----------------------------------------------------------------
# Manual selection of the elastic region
# -----------------------------------------------------------------
rectangle_drawn = False

fig, ax = plt.subplots()
ax.plot(Disp_mm, Force_N, 'k')
ax.axhline(Force_grips, color='r', linestyle=':', linewidth=1)
ax.plot(Disp_mm[start_idx], Force_N[start_idx], 'ro')
ax.set(xlabel='Displacement [mm]', ylabel='Force [N]',
       title=f'Select elastic region ABOVE the red line (F_grips = {Force_grips:.1f} N)')


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


def on_button_click(event):
    if rectangle_drawn:
        plt.close()


button_ax = plt.axes([0.85, 0.02, 0.1, 0.04])
button = Button(button_ax, 'Continue')
button.on_clicked(on_button_click)
plt.show()

if not rectangle_drawn:
    raise SystemExit('No elastic region selected.')

roi_xmin, roi_ymin, roi_width, roi_height = rs.extents
in_roi = ((Disp_mm >= roi_xmin) & (Disp_mm <= roi_xmin + roi_width)
          & (Force_N >= roi_ymin) & (Force_N <= roi_ymin + roi_height))
in_roi &= (np.arange(len(Force_N)) >= start_idx)

if in_roi.sum() < 10:
    raise RuntimeError('Fewer than 10 selected points lie above Force_grips.')

Disp_mm_elastic = Disp_mm[in_roi]
Force_N_elastic = Force_N[in_roi]

coef = np.polyfit(Disp_mm_elastic, Force_N_elastic, 1)
k_effective = float(coef[0])
intercept = float(coef[1])

if k_effective >= k_sample:
    raise RuntimeError(
        f'Measured system stiffness ({k_effective:.1f} N/mm) is not below the theoretical '
        f'sample stiffness ({k_sample:.1f} N/mm). Check E, area, gauge length and selection.'
    )

k_loadframe = 1 / ((1 / k_effective) - (1 / k_sample))

# -----------------------------------------------------------------
# Corrections
# -----------------------------------------------------------------
Disp_mm_corr = Disp_mm - (Force_N / k_loadframe)

disp_at_grips = (Force_grips - intercept) / k_effective - Force_grips / k_loadframe
elastic_at_grips = (Force_grips * initial_length_mm) / (initial_area_mm2 * theoretical_elastic_modulus_MPa)
offset = disp_at_grips - elastic_at_grips

corr_elast_grips = Disp_mm_corr - offset

keep = np.arange(len(Force_N)) >= start_idx
Disp_mm = Disp_mm[keep]
Force_N = Force_N[keep]
Disp_mm_corr = Disp_mm_corr[keep]
corr_elast_grips = corr_elast_grips[keep]
sigma_eng = sigma_eng[keep]
eps_e = eps_e[keep]

eps_t = np.log1p(corr_elast_grips / initial_length_mm)
eps_p = eps_t - eps_e

offset_yield = None
idx = np.where(eps_p >= PLASTIC_THRESH)[0]
if idx.size > 0:
    j = int(idx[0])
    if j > 0 and (eps_p[j] - eps_p[j - 1]) != 0:
        frac = (PLASTIC_THRESH - eps_p[j - 1]) / (eps_p[j] - eps_p[j - 1])
        offset_yield = float(sigma_eng[j - 1] + frac * (sigma_eng[j] - sigma_eng[j - 1]))
    else:
        offset_yield = float(sigma_eng[j])

# -----------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------
frac_rig = (1 / k_loadframe) / (1 / k_effective)
F_max = float(np.max(Force_N))
elastic_ext_max = F_max * initial_length_mm / (initial_area_mm2 * theoretical_elastic_modulus_MPa)
disp_err = elastic_ext_max * E_rel_uncertainty

f = open(os.path.join(pathlib.Path(file_path).parent.resolve(), 'Compliance_Correction_Data.txt'), 'w+')


def print_to_console_and_file(message, f=f):
    print(message)
    print(message, file=f)


print_to_console_and_file('=================== GRIP SETTLING ===================')
print_to_console_and_file('Force_grips: ' + str(np.round(Force_grips, 2)) + ' N')
print_to_console_and_file('discarded points: ' + str(start_idx) + ' of ' + str(len(Disp_full)))
print_to_console_and_file('offset removed: ' + str(np.round(offset * 1000, 2)) + ' um')
print_to_console_and_file('==================== FIT WINDOW =====================')
print_to_console_and_file('points used: ' + str(int(in_roi.sum())))
print_to_console_and_file('force range: ' + str(np.round(Force_N_elastic.min(), 1)) + ' to '
                          + str(np.round(Force_N_elastic.max(), 1)) + ' N')
print_to_console_and_file('====================== MEASURED =====================')
print_to_console_and_file('max displacement: ' + str(np.round(Disp_full[-1], 3)) + ' mm')
print_to_console_and_file('max strain: ' + str(np.round(np.log1p(Disp_full[-1] / initial_length_mm) * 100, 2)) + ' %')
print_to_console_and_file('=============== STIFFNESS-CORRECTED =================')
print_to_console_and_file('max displacement: ' + str(np.round(Disp_mm_corr[-1], 3)) + ' mm')
print_to_console_and_file('max strain: ' + str(np.round(np.log1p(Disp_mm_corr[-1] / initial_length_mm) * 100, 2)) + ' %')
print_to_console_and_file('================= GRIPS-CORRECTED ===================')
print_to_console_and_file('max displacement: ' + str(np.round(corr_elast_grips[-1], 3)) + ' mm')
print_to_console_and_file('max strain: ' + str(np.round(np.log1p(corr_elast_grips[-1] / initial_length_mm) * 100, 2)) + ' %')
print_to_console_and_file('==================== STIFFNESS ======================')
print_to_console_and_file('Sample (assumed E): ' + str(np.round(k_sample, 2)) + ' N/mm')
print_to_console_and_file('Total (measured): ' + str(np.round(k_effective, 2)) + ' N/mm')
print_to_console_and_file('Rig (derived): ' + str(np.round(k_loadframe, 2)) + ' N/mm')
print_to_console_and_file('Rig share of compliance: ' + str(np.round(frac_rig * 100, 1)) + ' %')
print_to_console_and_file('==================== MECHANICAL =====================')
print_to_console_and_file('UTS: ' + str(np.round(np.max(sigma_eng), 2)) + ' MPa')
if offset_yield is not None:
    print_to_console_and_file('0.2% Offset yield: ' + str(np.round(offset_yield, 2)) + ' MPa')
else:
    print_to_console_and_file('0.2% Offset yield: not reached')
print_to_console_and_file('=================== UNCERTAINTY =====================')
print_to_console_and_file('assumed dE/E: ' + str(np.round(E_rel_uncertainty * 100, 0)) + ' %')
print_to_console_and_file('elastic gauge extension at F_max: ' + str(np.round(elastic_ext_max * 1000, 2)) + ' um')
print_to_console_and_file('resulting elongation error: ' + str(np.round(disp_err * 1000, 2)) + ' um ('
                          + str(np.round(disp_err / initial_length_mm * 100, 4)) + ' % strain)')
print_to_console_and_file('=====================================================')
print_to_console_and_file('NOTE: E is an input to this correction and cannot be reported as a result')
print_to_console_and_file('NOTE: Assumption: section of ' + str(initial_length_mm) + ' mm is being deformed')
print_to_console_and_file('=====================================================')

f.close()

# -----------------------------------------------------------------
# Export and plots
# -----------------------------------------------------------------
strain_effective = np.log1p(Disp_mm / initial_length_mm) * 100
strain_corrected = np.log1p(Disp_mm_corr / initial_length_mm) * 100
strain_grips_corrected = np.log1p(corr_elast_grips / initial_length_mm) * 100

if PlotStress == 1:
    Y_plot = sigma_eng
    Y_elastic = Force_N_elastic / initial_area_mm2
    label = 'Engineering Stress [MPa]'
else:
    Y_plot = Force_N
    Y_elastic = Force_N_elastic
    label = 'Force [N]'

result = pd.DataFrame({
    'Force [N]': Force_N,
    'Engineering Stress [MPa]': sigma_eng,
    'Raw Elongation [mm]': Disp_mm,
    'Stiffness-Corrected Elongation [mm]': Disp_mm_corr,
    'Grips-Corrected Elongation [mm]': corr_elast_grips,
    'Raw True Strain [%]': strain_effective,
    'Stiffness-Corrected True Strain [%]': strain_corrected,
    'Grips-Corrected True Strain [%]': strain_grips_corrected,
    'Plastic True Strain [-]': eps_p,
})
result.to_csv(os.path.join(os.path.dirname(file_path), 'Correction Results.csv'), index=False)

plt.figure(figsize=(8, 4))
plt.figure(1)
plt.plot(Disp_mm, Y_plot, 'k', label='Raw Data')
plt.plot(Disp_mm_corr, Y_plot, 'b', label='Stiffness-Corrected')
plt.plot(corr_elast_grips, Y_plot, 'g', label='Grips-Corrected')
plt.plot(Disp_mm_elastic, Y_elastic, 'r', label='Elastic Region')
plt.legend(loc='lower right')
plt.xlim([-0.01, np.max(Disp_mm) + 0.2])
plt.ylim([0, np.max(Y_plot) * 1.05])
plt.xlabel('Displacement [mm]')
plt.ylabel(label)
plt.show(block=False)
plt.pause(0.001)

plt.figure(figsize=(8, 4))
plt.figure(2)
plt.plot(strain_effective, Y_plot, 'k', label='Raw Data')
plt.plot(strain_corrected, Y_plot, 'b', label='Stiffness-Corrected')
plt.plot(strain_grips_corrected, Y_plot, 'g', label='Grips-Corrected')
if offset_yield is not None:
    plt.plot((PLASTIC_THRESH + offset_yield / theoretical_elastic_modulus_MPa) * 100,
             offset_yield, 'rx', markersize=9, label='0.2% Offset Yield')
plt.legend(loc='lower right')
plt.xlim([-0.01, np.max(strain_effective) * 1.05])
plt.ylim([0, np.max(Y_plot) * 1.05])
plt.xlabel('True Strain [%]')
plt.ylabel(label)
plt.show(block=False)
plt.pause(0.001)
plt.show()