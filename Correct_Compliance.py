import os
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from tkinter import filedialog as fd
import pathlib
from sklearn.linear_model import LinearRegression
from scipy.signal import savgol_filter as savogol_filter
# ===============================
# CONFIG / USER INPUTS
# ===============================
# Elastic-region detection
AUTO_ELASTIC = True                # set False to fall back to manual rectangle selection (not implemented in this new version)
AUTO_METHOD = "plastic"                # "r2" or "plastic"
R2_THRESHOLD = 0.999               # for AUTO_METHOD == "r2"
MIN_ELASTIC_POINTS = 325            # minimum number of points required in elastic window
MAX_R2_DROP_TOL = 5                # consecutive points below threshold allowed before we stop expanding the window

PLASTIC_THRESH = 0.002             # 0.2% plastic true strain for AUTO_METHOD == "plastic"
USE_TRUE_STRAIN = True             # compute true strain from displacement; if False use engineering strain

# Force_grips detection window (used to find the point after grips settle)
FORCE_GRIPS_WINDOW = (90, 130)     # [N] lower/upper search bounds for the two-line piecewise fit

# Material/geometry
E_MPa = 200000                    # Young's modulus (MPa)
A0_mm2 = 4 * 0.27                  # initial gauge area width x thickness (mm^2)
L0_mm = 8                          # initial gauge length (mm)

# Plotting
PLOT_STRESS = 1                    # 1 -> plot stress [MPa], 0 -> plot force [N]
SHOW_DEBUG_PLOTS = True            # show debug visualisation of auto-picked elastic region
# ===============================

print('Select CSV File (Disp in 2nd Col, Force in 3rd Col, Data starts from row 21)')
file_path = fd.askopenfilename()

datas = pd.read_csv(file_path, usecols=[1, 2], skiprows=20, header=None, names=['Disp_mm', 'Force_N'])

# Extract columns
Disp_mm = datas['Disp_mm'].to_numpy()
Force_N = datas['Force_N'].to_numpy()

# Helper: strains from displacement
def total_strain(disp_mm, true=USE_TRUE_STRAIN):
    if true:
        return np.log1p(disp_mm / L0_mm)
    return disp_mm / L0_mm

# ---------------
# Step 1: Auto detect elastic region (on RAW load–displacement) to estimate k_effective
# ---------------

# (a) Find Force_grips by two-segment linear regression within FORCE_GRIPS_WINDOW
low, high = FORCE_GRIPS_WINDOW
mask_win = (Force_N >= low) & (Force_N <= high)
if not np.any(mask_win):
    raise RuntimeError("No data points inside FORCE_GRIPS_WINDOW. Adjust FORCE_GRIPS_WINDOW.")

x = Disp_mm[mask_win]
y = Force_N[mask_win]

best_split = None
min_ssr = np.inf
for i in range(2, len(x) - 2):  # leave at least 2 pts in each segment
    x1, y1 = x[:i], y[:i]
    x2, y2 = x[i:], y[i:]
    m1 = LinearRegression().fit(x1.reshape(-1, 1), y1)
    m2 = LinearRegression().fit(x2.reshape(-1, 1), y2)
    r1 = y1 - m1.predict(x1.reshape(-1, 1))
    r2 = y2 - m2.predict(x2.reshape(-1, 1))
    ssr = (r1**2).sum() + (r2**2).sum()
    if ssr < min_ssr:
        min_ssr = ssr
        best_split = i

Force_grips = y[best_split]
start_idx = np.argmax(Force_N >= Force_grips)  # first index where force >= Force_grips

# (b) Build elastic window starting from start_idx
if AUTO_ELASTIC:
    if AUTO_METHOD == "r2":
        # Expand window [start_idx : end_idx] while R^2 stays high
        end_idx = start_idx + max(MIN_ELASTIC_POINTS, 5)
        end_idx = min(end_idx, len(Disp_mm) - 1)

        below_ctr = 0
        best_end = end_idx
        while end_idx < len(Disp_mm):
            X = Disp_mm[start_idx:end_idx+1].reshape(-1, 1)
            Y = Force_N[start_idx:end_idx+1]
            if len(Y) < max(MIN_ELASTIC_POINTS, 5):
                end_idx += 1
                continue
            model = LinearRegression().fit(X, Y)
            yhat = model.predict(X)
            ss_res = ((Y - yhat)**2).sum()
            ss_tot = ((Y - Y.mean())**2).sum()
            r2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else 0.0

            if r2 >= R2_THRESHOLD:
                best_end = end_idx
                below_ctr = 0
            else:
                below_ctr += 1
                if below_ctr >= MAX_R2_DROP_TOL:
                    break
            end_idx += 1

        elast_slice = slice(start_idx, max(best_end, start_idx + MIN_ELASTIC_POINTS))

    elif AUTO_METHOD == "plastic":
        # Use 0.2% plastic true strain criterion: plastic = total_true - sigma/E
        sigma = Force_N / A0_mm2  # MPa
        eps_e = sigma / E_MPa
        eps_t = total_strain(Disp_mm)
        eps_p = eps_t - eps_e
        # starting at start_idx, walk forward while eps_p < PLASTIC_THRESH
        end_idx = start_idx
        while end_idx < len(Disp_mm) and (eps_p[end_idx] < PLASTIC_THRESH or (end_idx - start_idx) < MIN_ELASTIC_POINTS):
            end_idx += 1
        elast_slice = slice(start_idx, min(end_idx, len(Disp_mm) - 1))
    else:
        raise ValueError("AUTO_METHOD must be 'r2' or 'plastic'")
else:
    raise NotImplementedError("Manual rectangle selection removed in this auto version. Set AUTO_ELASTIC=True or reintroduce the old selector.")

Disp_mm_elastic = Disp_mm[elast_slice]
Force_N_elastic = Force_N[elast_slice]

# Linear fit to get effective stiffness k_effective (slope dF/dΔ)
coef = np.polyfit(Disp_mm_elastic, Force_N_elastic, 1)
k_effective = coef[0]            # N/mm

# ---------------
# Step 2: Decompose stiffness and correct displacement
# ---------------

k_sample = (E_MPa * A0_mm2) / L0_mm  # N/mm
k_loadframe = 1.0 / ((1.0 / k_effective) - (1.0 / k_sample))

# frame compliance correction
Disp_mm_corr = Disp_mm - (Force_N / k_loadframe)
Disp_mm_corr = Disp_mm_corr - Disp_mm_corr[0]

# centre corrected displacement at theoretical starting point
start_load = Force_N[0]
start_disp = start_load / k_sample
Disp_mm_corr = Disp_mm_corr + start_disp

# ---------------
# Step 3: Determine Force_grips location in corrected data for grips correction
# ---------------
# reuse previously found Force_grips; find last fully elastic point wrt grips
elast_force_idx = np.where(Force_N < Force_grips)[0]
if len(elast_force_idx) == 0:
    elast_force_idx = np.array([0])

corr_disp_max = Disp_mm_corr[elast_force_idx[-1]]

# ---------------
# Step 4: Grips correction (gage-only displacement)
# ---------------

corr_elast_grips = []
offset_yield = None
for i in range(len(Disp_mm_corr)):
    if i < elast_force_idx[-1]:
        stress = Force_N[i] / A0_mm2
        corr_disp = Disp_mm_corr[i]
    else:
        stress = Force_grips / A0_mm2
        corr_disp = corr_disp_max

    eng_str = stress / E_MPa
    log_disp = math.log(1 + eng_str) * L0_mm
    offset = corr_disp - log_disp
    gage_disp = Disp_mm_corr[i] - offset
    corr_elast_grips.append(gage_disp)

    # 0.2% offset yield (same as original)
    if math.log(1 + gage_disp / L0_mm) >= 0.02 - Force_N[i] / (A0_mm2 * E_MPa):
        if offset_yield is None:
            offset_yield = Force_N[i] / A0_mm2

corr_elast_grips = np.asarray(corr_elast_grips)

# total true strain from grips-corrected displacement
eps_t = np.log1p(corr_elast_grips / L0_mm)  # true strain

# plastic (true) strain per your definition
eps_p = eps_t - eps_e

# find first index where eps_p >= 0.002 and linearly interpolate stress
offset_yield = None
idx = np.where(eps_p >= PLASTIC_THRESH)[0] #find where plastic strain exceeds threshold
if idx.size > 0:
    j = int(idx[0]) #first first instance where plastic strain exceeds threshold
    if j > 0: #linear interpolation between index where plastic strain exceeds threshold and previous index
        x0, x1 = eps_p[j-1], eps_p[j]
        y0, y1 = sigma[j-1], sigma[j]
        if (x1 - x0) != 0:
            frac = (PLASTIC_THRESH - x0) / (x1 - x0) #linear interpolation to where plastic strain = 0.002
            offset_yield = float(y0 + frac * (y1 - y0))  # MPa
        else:
            offset_yield = float(sigma[j])
    else:
        offset_yield = float(sigma[j])

# ---------------
# Step 5: Reporting
# ---------------

out_path = os.path.join(pathlib.Path(file_path).parent.resolve(), "Compliance_Correction_Data.txt")
with open(out_path, "w+") as f:
    def log(msg):
        print(msg)
        print(msg, file=f)

    log('====================== MEASURED =====================')
    log(f'max displacement: {np.round(Disp_mm[-1], 2)} mm')
    log(f'max strain: {np.round(np.log1p(Disp_mm[-1] / L0_mm) * 100, 2)} %')
    log('=============== STIFFNESS-CORRECTED =================')
    log(f'max displacement: {np.round(Disp_mm_corr[-1], 2)} mm')
    log(f'max strain: {np.round(np.log1p(Disp_mm_corr[-1] / L0_mm) * 100, 2)} %')
    log('================= GRIPS-CORRECTED ===================')
    log(f'max displacement: {np.round(corr_elast_grips[-1], 2)} mm')
    log(f'max strain: {np.round(np.log1p(corr_elast_grips[-1] / L0_mm) * 100, 2)} %')
    log("================= YOUNG's MODULUS ===================")
    log(f'Sample (theoretical): {np.round(E_MPa/1000, 2)} GPa')
    log(f'Total: {np.round(k_effective * L0_mm / A0_mm2 / 1000, 2)} GPa')
    log(f'Rig: {np.round(k_loadframe * L0_mm / A0_mm2 / 1000, 2)} GPa')
    log('==================== STIFFNESS ======================')
    log(f'Sample (theoretical): {np.round(k_sample, 2)} N/mm')
    log(f'Total: {np.round(k_effective, 2)} N/mm')
    log(f'Rig: {np.round(k_loadframe, 2)} N/mm')
    log('==================== MECHANICAL =====================')
    log(f'UTS: {np.round(np.max(Force_N)/A0_mm2, 2)} MPa')
    if offset_yield is not None:
        log(f'0.2% Offset yield: {np.round(offset_yield, 2)} MPa')
    else:
        log('0.2% Offset yield: not found')
    log('=====================================================')
    log('NOTE: Assumption: section of 8 mm is being deformed')
    log('=====================================================')


# ---------------
# Step 6: Plots and export
# ---------------

strain_effective = total_strain(Disp_mm) * 100
strain_corrected = total_strain(Disp_mm_corr) * 100
strain_grips_corrected = total_strain(corr_elast_grips) * 100

if PLOT_STRESS == 1:
    Y_plot = Force_N / A0_mm2
    Y_el = Force_N_elastic / A0_mm2
    ylab = 'Stress [MPa]'
else:
    Y_plot = Force_N
    Y_el = Force_N_elastic
    ylab = 'Force [N]'

result = pd.DataFrame({
    ylab: Y_plot,
    'Raw Elongation [mm]': Disp_mm,
    'Stiffness-Corrected Elongation [mm]': Disp_mm_corr,
    'Grips-Corrected Elongation [mm]': corr_elast_grips,
    'Raw True Strain [%]': strain_effective,
    'Stiffness-Corrected True Strain [%]': strain_corrected,
    'Grips-Corrected True Strain [%]': strain_grips_corrected
})
result.to_csv(os.path.join(os.path.dirname(file_path), 'Correction Results.csv'), index=False)

# Load–displacement
plt.figure(figsize=(8, 4))
plt.plot(Disp_mm, Y_plot, label='Raw')
plt.plot(Disp_mm_corr, Y_plot, label='Stiffness-Corrected')
plt.plot(corr_elast_grips, Y_plot, label='Grips-Corrected')
if SHOW_DEBUG_PLOTS:
    plt.plot(Disp_mm_elastic, Y_el, label='Auto Elastic Region', linewidth=2)
    #plt.plot(0.02+offset_yield/E_MPa, offset_yield, 'rx', label='0.2% Offset Yield')
plt.xlim([-0.01, np.max(Disp_mm) + 0.2])
plt.ylim([0, np.max(Y_plot) * 1.05])
plt.xlabel('Displacement [mm]')
plt.ylabel(ylab)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show(block=False)
plt.pause(0.001)

# Stress–strain
plt.figure(figsize=(8, 4))
plt.plot(strain_effective, Y_plot, label='Raw')
plt.plot(strain_corrected, Y_plot, label='Stiffness-Corrected')
plt.plot(strain_grips_corrected, Y_plot, label='Grips-Corrected')
plt.xlim([-0.01, np.max(strain_effective) * 1.05])
plt.ylim([0, np.max(Y_plot) * 1.05])
plt.xlabel('True Strain [%]')
plt.ylabel(ylab)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()