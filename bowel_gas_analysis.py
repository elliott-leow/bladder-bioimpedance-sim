#!/usr/bin/env python3
"""
How bad is bowel gas really? And how do we deal with it?

Simulates realistic bowel gas dynamics over time and shows that
temporal averaging dramatically reduces its impact because gas is
transient while bladder filling is monotonic.
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from bladder_sim.model import build_pelvis_model, get_tissue_labels
from bladder_sim.fem import compute_transfer_impedance, Image

# ============================================================
# Part 1: How much gas is realistic?
# ============================================================

N_PER_RING = 4
RING_Z = np.array([9.0, 10.0])

print('Building model...')
fmdl, img = build_pelvis_model(300, freq_kHz=50.0, n_per_ring=N_PER_RING,
                                ring_z=RING_Z, stim_pattern='none')
mesh = fmdl.mesh
labels = get_tissue_labels(mesh)
sigma_base = img.elem_data.copy()

# Compute SVD weights from bladder change
print('Computing SVD weights...')
_, img_lo = build_pelvis_model(100, mesh=mesh, freq_kHz=50.0,
                                n_per_ring=N_PER_RING, ring_z=RING_Z, stim_pattern='none')
img_lo.fwd_model = fmdl
_, img_hi = build_pelvis_model(500, mesh=mesh, freq_kHz=50.0,
                                n_per_ring=N_PER_RING, ring_z=RING_Z, stim_pattern='none')
img_hi.fwd_model = fmdl

Z_lo = compute_transfer_impedance(fmdl, img_lo)
Z_hi = compute_transfer_impedance(fmdl, img_hi)
Z_base = compute_transfer_impedance(fmdl, img)

n_elec = mesh.n_electrodes
dV = 400.0
M_bladder = (Z_hi - Z_lo) / dV * 1000

# Build full pair matrix
pairs = []
for i in range(n_elec):
    for j in range(i+1, n_elec):
        pairs.append((i, j))
n_pairs = len(pairs)

M_full = np.zeros((n_pairs, n_pairs))
for di, (d1, d2) in enumerate(pairs):
    for si, (s1, s2) in enumerate(pairs):
        M_full[di, si] = M_bladder[d1, s1] - M_bladder[d1, s2] - M_bladder[d2, s1] + M_bladder[d2, s2]

U, S, Vt = np.linalg.svd(M_full)
u1 = U[:, 0]
v1 = Vt[0, :]

svd_bladder_sens = S[0]  # mOhm/mL
print(f'SVD rank-1 bladder sensitivity: {svd_bladder_sens:.3f} mOhm/mL')

# ============================================================
# Part 2: Sweep gas fractions — how much does it matter?
# ============================================================

bowel_mask = (labels == 5)
bowel_indices = np.where(bowel_mask)[0]
n_bowel = len(bowel_indices)
rng = np.random.default_rng(42)

gas_fractions = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
gas_artifacts = []

print('\nSweeping gas fractions...')
for frac in gas_fractions:
    # Random subset of bowel becomes gas
    n_gas = int(frac * n_bowel)
    gas_elems = rng.choice(bowel_indices, size=n_gas, replace=False)

    sigma_gas = sigma_base.copy()
    sigma_gas[gas_elems] = 0.01  # gas

    img_gas = Image(fwd_model=fmdl, elem_data=sigma_gas)
    Z_gas = compute_transfer_impedance(fmdl, img_gas)

    M_gas = (Z_gas - Z_base) * 1000  # mOhm

    # Compute through SVD weights
    M_gas_full = np.zeros((n_pairs, n_pairs))
    for di, (d1, d2) in enumerate(pairs):
        for si, (s1, s2) in enumerate(pairs):
            M_gas_full[di, si] = M_gas[d1, s1] - M_gas[d1, s2] - M_gas[d2, s1] + M_gas[d2, s2]

    artifact = abs(u1 @ M_gas_full @ v1)
    equiv_mL = artifact / svd_bladder_sens
    gas_artifacts.append((frac, artifact, equiv_mL))
    print(f'  {frac*100:5.1f}% gas: {artifact:.2f} mOhm = {equiv_mL:.1f} mL equivalent')

# ============================================================
# Part 3: Temporal simulation — gas moves, bladder doesn't
# ============================================================

print('\nSimulating gas dynamics over 30 minutes...')

# Time parameters
dt = 10.0  # seconds between measurements
T_total = 30 * 60  # 30 minutes
t = np.arange(0, T_total, dt)
n_samples = len(t)

# Bladder filling: 2 mL/min = 0.033 mL/s (normal diuresis)
filling_rate = 2.0 / 60  # mL/s
bladder_volume_change = filling_rate * t  # mL above baseline
bladder_signal = bladder_volume_change * svd_bladder_sens  # mOhm

# Bowel gas: model as Ornstein-Uhlenbeck process
# Gas pockets appear/disappear with correlation time tau
# This is a standard model for random fluctuations that mean-revert
tau_gas = 3 * 60  # 3-minute correlation time (gas pockets last ~3 min)
sigma_gas_noise = 5.8  # mOhm RMS (from our simulation, 30% gas worst case)
# Scale down: typical gas is more like 5-10% at any time, not 30%
# Average fraction ~10% gives about 2 mOhm of artifact
sigma_gas_typical = 2.0  # mOhm RMS for typical gas (not worst case)

# Generate OU process
gas_noise = np.zeros(n_samples)
gas_noise[0] = rng.normal(0, sigma_gas_typical)
alpha = np.exp(-dt / tau_gas)
innovation_std = sigma_gas_typical * np.sqrt(1 - alpha**2)
for i in range(1, n_samples):
    gas_noise[i] = alpha * gas_noise[i-1] + rng.normal(0, innovation_std)

# Also add respiratory artifact (already mostly filtered by bandstop)
resp_residual = 1.3  # mOhm after bandstop filter
resp_noise = resp_residual * np.sin(2 * np.pi * 0.25 * t)  # 15 breaths/min
# After bandstop, this is reduced to ~0.1 mOhm
resp_after_filter = 0.1 * np.sin(2 * np.pi * 0.25 * t + rng.uniform(0, 2*np.pi))

# Electronic noise (negligible)
electronic_noise = 0.013 * rng.standard_normal(n_samples)

# Total measured signal
total_signal = bladder_signal + gas_noise + resp_after_filter + electronic_noise

# ============================================================
# Part 4: Temporal filtering strategies
# ============================================================

from scipy.signal import savgol_filter
from scipy.ndimage import uniform_filter1d

# Strategy 1: Raw (no filtering beyond bandstop)
raw_estimate = total_signal / svd_bladder_sens  # mL

# Strategy 2: Moving average (2-minute window)
window_2min = int(2 * 60 / dt)  # samples in 2 minutes
if window_2min % 2 == 0:
    window_2min += 1
ma_2min = uniform_filter1d(total_signal, window_2min) / svd_bladder_sens

# Strategy 3: Moving average (5-minute window)
window_5min = int(5 * 60 / dt)
if window_5min % 2 == 0:
    window_5min += 1
ma_5min = uniform_filter1d(total_signal, window_5min) / svd_bladder_sens

# Strategy 4: Linear regression (fit line to last 10 minutes)
# This is the strongest — if bladder filling is approximately linear,
# a line fit rejects ALL zero-mean noise
from numpy.polynomial import polynomial as P

regression_estimate = np.full(n_samples, np.nan)
regression_window = int(10 * 60 / dt)  # 10-minute window
for i in range(regression_window, n_samples):
    t_win = t[i-regression_window:i+1]
    s_win = total_signal[i-regression_window:i+1]
    # Fit line: signal = a + b*t
    coeffs = np.polyfit(t_win - t_win[0], s_win, 1)
    slope = coeffs[0]  # mOhm/s
    # Convert slope to volume rate
    rate_mL_per_s = slope / svd_bladder_sens
    # Integrate from start
    regression_estimate[i] = rate_mL_per_s * t[i]

# True bladder volume
true_volume = bladder_volume_change

# Compute errors
raw_error = np.abs(raw_estimate - true_volume)
ma2_error = np.abs(ma_2min - true_volume)
ma5_error = np.abs(ma_5min - true_volume)

valid = ~np.isnan(regression_estimate)
reg_error = np.full(n_samples, np.nan)
reg_error[valid] = np.abs(regression_estimate[valid] - true_volume[valid])

print(f'\nVolume estimation errors (RMS over last 20 minutes):')
last_20 = t > 10 * 60
print(f'  Raw (10s readings):     {np.sqrt(np.mean(raw_error[last_20]**2)):.1f} mL')
print(f'  2-min moving average:   {np.sqrt(np.mean(ma2_error[last_20]**2)):.1f} mL')
print(f'  5-min moving average:   {np.sqrt(np.mean(ma5_error[last_20]**2)):.1f} mL')
valid_last = valid & last_20
print(f'  10-min linear fit:      {np.sqrt(np.nanmean(reg_error[valid_last]**2)):.1f} mL')

# ============================================================
# Figure
# ============================================================

fig = plt.figure(figsize=(16, 14))

# --- Panel 1: Gas fraction sweep ---
ax1 = fig.add_subplot(3, 2, 1)
fracs = [x[0]*100 for x in gas_artifacts]
artifacts_mOhm = [x[1] for x in gas_artifacts]
equiv_mLs = [x[2] for x in gas_artifacts]
ax1.bar(range(len(fracs)), equiv_mLs, color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=0.8)
ax1.set_xticks(range(len(fracs)))
ax1.set_xticklabels([f'{f:.0f}%' for f in fracs])
ax1.set_xlabel('Fraction of bowel that is gas')
ax1.set_ylabel('Equivalent volume error (mL)')
ax1.set_title('How bad is bowel gas?\n(single snapshot, SVD rank-1)')
# Add typical range annotation
ax1.axhspan(0, equiv_mLs[1], alpha=0.15, color='green')
ax1.text(0.5, equiv_mLs[1]*0.4, 'Typical\nrange', ha='center', fontsize=9, color='green',
         fontweight='bold')
ax1.axhspan(equiv_mLs[2], equiv_mLs[-1], alpha=0.1, color='red')
for i, (f, a, m) in enumerate(gas_artifacts):
    ax1.text(i, m + 0.3, f'{m:.0f} mL', ha='center', fontsize=9, fontweight='bold')

# --- Panel 2: Why gas is different from other noise ---
ax2 = fig.add_subplot(3, 2, 2)
noise_sources = ['Electronic\n(1s avg)', 'Respiratory\n(after filter)', 'Electrode\ndrift (1 min)', 'Bowel gas\n(10% typical)', 'Bowel gas\n(30% worst)']
noise_mOhm = [0.013, 0.1, 0.3, gas_artifacts[1][1], gas_artifacts[4][1]]
noise_equiv_mL = [n / svd_bladder_sens for n in noise_mOhm]
colors2 = ['#27ae60', '#27ae60', '#f39c12', '#e74c3c', '#c0392b']
bars2 = ax2.barh(noise_sources, noise_equiv_mL, color=colors2, alpha=0.8, edgecolor='black', linewidth=0.8)
ax2.set_xlabel('Equivalent volume error (mL)')
ax2.set_title('Noise budget\n(SVD rank-1, single reading)')
for bar, val in zip(bars2, noise_equiv_mL):
    ax2.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
             f'{val:.1f} mL', va='center', fontsize=9, fontweight='bold')
ax2.set_xlim(0, max(noise_equiv_mL) * 1.3)

# --- Panel 3: Time series showing gas is transient ---
ax3 = fig.add_subplot(3, 1, 2)
t_min = t / 60
ax3.plot(t_min, bladder_signal, 'b-', linewidth=2.5, label='Bladder signal (truth)', zorder=5)
ax3.plot(t_min, gas_noise, '-', color='#e74c3c', alpha=0.7, linewidth=1, label='Bowel gas artifact')
ax3.plot(t_min, total_signal, '-', color='gray', alpha=0.5, linewidth=0.8, label='Total measured')
ax3.fill_between(t_min, bladder_signal - 2*sigma_gas_typical,
                 bladder_signal + 2*sigma_gas_typical, alpha=0.1, color='red',
                 label='Gas uncertainty (±2σ)')
ax3.set_xlabel('Time (minutes)')
ax3.set_ylabel('Signal (mOhm)')
ax3.set_title('Bowel gas is random & transient; bladder filling is steady & monotonic')
ax3.legend(loc='upper left', fontsize=9)
ax3.grid(True, alpha=0.3)

# Add annotation
ax3.annotate('Gas fluctuates around zero\n(3-min correlation time)',
             xy=(15, gas_noise[int(15*60/dt)]),
             xytext=(8, -5),
             fontsize=9, color='#e74c3c',
             arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.5))
ax3.annotate('Bladder signal grows steadily\n(2 mL/min filling rate)',
             xy=(25, bladder_signal[int(25*60/dt)]),
             xytext=(18, bladder_signal[int(25*60/dt)] + 2),
             fontsize=9, color='blue',
             arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))

# --- Panel 4: Volume estimates with different filtering ---
ax4 = fig.add_subplot(3, 1, 3)
ax4.plot(t_min, true_volume, 'k--', linewidth=2, label='True bladder volume', zorder=5)
ax4.plot(t_min, raw_estimate, '-', color='gray', alpha=0.4, linewidth=0.8, label=f'Raw 10s readings')
ax4.plot(t_min, ma_2min, '-', color='#e67e22', linewidth=1.5, alpha=0.8, label='2-min moving average')
ax4.plot(t_min, ma_5min, '-', color='#2980b9', linewidth=2, label='5-min moving average')
ax4.plot(t_min[valid], regression_estimate[valid], '-', color='#27ae60', linewidth=2,
         label='10-min linear fit')
ax4.set_xlabel('Time (minutes)')
ax4.set_ylabel('Estimated volume change (mL)')
ax4.set_title('Temporal averaging kills gas noise because gas is zero-mean')
ax4.legend(loc='upper left', fontsize=9)
ax4.grid(True, alpha=0.3)

# Add error box
box_text = (f'RMS errors (last 20 min):\n'
            f'  Raw:          {np.sqrt(np.mean(raw_error[last_20]**2)):.1f} mL\n'
            f'  2-min avg:    {np.sqrt(np.mean(ma2_error[last_20]**2)):.1f} mL\n'
            f'  5-min avg:    {np.sqrt(np.mean(ma5_error[last_20]**2)):.1f} mL\n'
            f'  10-min fit:   {np.sqrt(np.nanmean(reg_error[valid_last]**2)):.1f} mL')
ax4.text(0.98, 0.05, box_text, transform=ax4.transAxes, fontsize=9,
         verticalalignment='bottom', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9),
         fontfamily='monospace')

fig.suptitle('Bowel Gas: The Dominant Noise Source and How to Beat It',
             fontsize=15, fontweight='bold', y=1.01)
fig.tight_layout()
fig.savefig('figures/bowel_gas_analysis.png', dpi=200, bbox_inches='tight')
print(f'\nSaved: figures/bowel_gas_analysis.png')
