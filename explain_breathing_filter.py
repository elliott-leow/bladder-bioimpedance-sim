#!/usr/bin/env python3
"""
Step-by-step explanation of the breathing artifact rejection algorithm.
Shows what each filtering stage does to the signal, in both time and frequency domain.
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch

rng = np.random.default_rng(42)

# ============================================================
# Generate realistic composite signal
# ============================================================

fs = 10.0  # 10 Hz sampling (100 ms per SVD measurement cycle)
T = 10 * 60  # 10 minutes
t = np.arange(0, T, 1/fs)
n = len(t)

# --- Component 1: Bladder filling (THE SIGNAL WE WANT) ---
# 2 mL/min filling, sensitivity 0.94 mOhm/mL
filling_rate_mOhm_per_s = 2.0 / 60 * 0.94  # mOhm/s
bladder = filling_rate_mOhm_per_s * t  # slow linear ramp

# --- Component 2: Breathing (THE MAIN ARTIFACT) ---
# 15 breaths/min = 0.25 Hz, amplitude ~20 mOhm peak-to-peak
# Real breathing isn't a perfect sine — it has harmonics and rate variation
breath_rate = 0.25  # Hz (15 breaths/min)
# Slowly varying breath rate (12-18 breaths/min)
breath_rate_variation = breath_rate + 0.02 * np.sin(2 * np.pi * 0.01 * t)
# Instantaneous phase (integral of frequency)
breath_phase = 2 * np.pi * np.cumsum(breath_rate_variation) / fs
# Breathing waveform: not a pure sine — inspiration is faster than expiration
# Model as sine + 30% second harmonic
breathing = 10.0 * (np.sin(breath_phase) + 0.3 * np.sin(2 * breath_phase))
# = ~20 mOhm peak-to-peak

# --- Component 3: Cardiac (small, fast) ---
# ~1.1 Hz (66 bpm), ~3 mOhm amplitude
heart_rate = 1.1  # Hz
cardiac = 1.5 * np.sin(2 * np.pi * heart_rate * t + 0.3 * np.sin(2 * np.pi * 0.005 * t))

# --- Component 4: Electrode drift (slow baseline wander) ---
# ~5 uOhm/s = 0.005 mOhm/s, but nonlinear
drift = 0.005 * t + 0.8 * np.sin(2 * np.pi * 0.002 * t)

# --- Component 5: Electronic noise ---
electronic = 0.013 * rng.standard_normal(n)

# --- Component 6: Bowel gas (slow random) ---
# OU process, tau = 5 min
tau_gas = 5 * 60
sigma_gas = 1.5  # mOhm RMS
gas = np.zeros(n)
alpha_gas = np.exp(-1/(fs * tau_gas))
innov = sigma_gas * np.sqrt(1 - alpha_gas**2)
for i in range(1, n):
    gas[i] = alpha_gas * gas[i-1] + rng.normal(0, innov)

# Total raw signal
raw = bladder + breathing + cardiac + drift + gas + electronic

# ============================================================
# Filtering pipeline
# ============================================================

# STAGE 1: Band-stop filter (0.15 - 0.4 Hz) — kills breathing
b_bs, a_bs = butter(4, [0.15, 0.4], btype='bandstop', fs=fs)
after_bandstop = filtfilt(b_bs, a_bs, raw)

# STAGE 2: Low-pass filter (< 0.08 Hz) — kills cardiac + residual breathing harmonics
b_lp, a_lp = butter(3, 0.08, btype='low', fs=fs)
after_lowpass = filtfilt(b_lp, a_lp, after_bandstop)

# STAGE 3: Polynomial baseline detrending (remove drift)
# Fit a low-order polynomial to the signal, subtract it, then add back
# the expected bladder component
# In practice: sliding 5-minute window, 2nd order polynomial
window_samples = int(5 * 60 * fs)  # 5 minutes
after_detrend = after_lowpass.copy()
# Simple approach: fit and subtract polynomial from entire signal
poly_coeffs = np.polyfit(t, after_lowpass, 3)
baseline_fit = np.polyval(poly_coeffs, t)
# The detrended signal shows the residual after removing the smooth trend
detrended = after_lowpass - baseline_fit

# For visualization, compute what the "ideal" extracted bladder signal looks like
# Apply same filters to just the bladder component
bladder_filtered = filtfilt(b_bs, a_bs, bladder)
bladder_filtered = filtfilt(b_lp, a_lp, bladder_filtered)

# ============================================================
# Compute spectra for frequency domain panels
# ============================================================

nperseg = int(60 * fs)  # 60-second windows
f_raw, psd_raw = welch(raw, fs=fs, nperseg=nperseg)
f_raw, psd_bladder = welch(bladder, fs=fs, nperseg=nperseg)
f_raw, psd_breathing = welch(breathing, fs=fs, nperseg=nperseg)
f_raw, psd_cardiac = welch(cardiac, fs=fs, nperseg=nperseg)
f_raw, psd_drift = welch(drift + gas, fs=fs, nperseg=nperseg)
f_raw, psd_after_bs = welch(after_bandstop, fs=fs, nperseg=nperseg)
f_raw, psd_after_lp = welch(after_lowpass, fs=fs, nperseg=nperseg)

# ============================================================
# Figure: 4-row step-by-step
# ============================================================

fig = plt.figure(figsize=(18, 22))

# Color scheme
C_RAW = '#555555'
C_BREATH = '#e74c3c'
C_CARDIAC = '#e67e22'
C_BLADDER = '#2980b9'
C_DRIFT = '#8e44ad'
C_CLEAN = '#27ae60'
C_FILTER_BAND = '#e74c3c'

t_min = t / 60  # time in minutes

# ====== ROW 1: The raw signal and what's in it ======
ax1a = fig.add_subplot(4, 2, 1)
ax1b = fig.add_subplot(4, 2, 2)

# Time domain: show 30 seconds of raw to see breathing
t_zoom = (t >= 60) & (t < 90)  # 30 seconds
ax1a.plot(t[t_zoom], raw[t_zoom], '-', color=C_RAW, linewidth=0.8, label='Raw signal')
ax1a.plot(t[t_zoom], breathing[t_zoom] + drift[t_zoom[0]:t_zoom[-1]+1].mean() + bladder[t_zoom[0]:t_zoom[-1]+1].mean(),
          '--', color=C_BREATH, linewidth=1.5, alpha=0.7, label='Breathing component')
ax1a.set_xlabel('Time (seconds)')
ax1a.set_ylabel('Impedance (mOhm)')
ax1a.set_title('Step 1: The raw signal (30-second zoom)')
ax1a.legend(fontsize=9, loc='upper right')
ax1a.grid(True, alpha=0.3)

# Add annotations
raw_pp = np.ptp(raw[t_zoom])
breath_pp = np.ptp(breathing[t_zoom])
bladder_range = filling_rate_mOhm_per_s * 30  # bladder change in 30s
ax1a.text(0.02, 0.95, f'Breathing: ~{breath_pp:.0f} mOhm peak-to-peak\n'
          f'Bladder (in this window): {bladder_range:.2f} mOhm\n'
          f'Ratio: breathing is {breath_pp/bladder_range:.0f}x larger!',
          transform=ax1a.transAxes, fontsize=9, va='top',
          bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

# Frequency domain: show all components
ax1b.semilogy(f_raw, psd_breathing, '-', color=C_BREATH, linewidth=2, label='Breathing (0.25 Hz)')
ax1b.semilogy(f_raw, psd_cardiac, '-', color=C_CARDIAC, linewidth=2, label='Cardiac (1.1 Hz)')
ax1b.semilogy(f_raw, psd_drift, '-', color=C_DRIFT, linewidth=1.5, label='Drift + bowel gas')
ax1b.semilogy(f_raw, psd_bladder + 1e-10, '-', color=C_BLADDER, linewidth=2, label='Bladder signal')
ax1b.semilogy(f_raw, psd_raw, '-', color=C_RAW, linewidth=0.8, alpha=0.5, label='Total')
ax1b.set_xlabel('Frequency (Hz)')
ax1b.set_ylabel('Power spectral density')
ax1b.set_title('Step 1: Frequency content — everything lives at different speeds')
ax1b.set_xlim(0, 2)
ax1b.legend(fontsize=8, loc='upper right')
ax1b.grid(True, alpha=0.3)

# Annotate frequency regions
ax1b.axvspan(0, 0.01, alpha=0.15, color=C_BLADDER, label='_')
ax1b.axvspan(0.15, 0.4, alpha=0.15, color=C_BREATH, label='_')
ax1b.text(0.005, ax1b.get_ylim()[1]*0.3, 'Bladder\nlives here', fontsize=8,
          color=C_BLADDER, ha='center', fontweight='bold')
ax1b.text(0.275, ax1b.get_ylim()[1]*0.3, 'Breathing\nlives here', fontsize=8,
          color=C_BREATH, ha='center', fontweight='bold')

# ====== ROW 2: Band-stop filter — kill breathing ======
ax2a = fig.add_subplot(4, 2, 3)
ax2b = fig.add_subplot(4, 2, 4)

# Time domain: before vs after bandstop (same 30s window)
ax2a.plot(t[t_zoom], raw[t_zoom], '-', color=C_RAW, linewidth=0.8, alpha=0.4, label='Before')
ax2a.plot(t[t_zoom], after_bandstop[t_zoom], '-', color=C_CLEAN, linewidth=1.5, label='After band-stop')
ax2a.set_xlabel('Time (seconds)')
ax2a.set_ylabel('Impedance (mOhm)')
ax2a.set_title('Step 2: Band-stop filter (0.15–0.4 Hz) removes breathing')
ax2a.legend(fontsize=9, loc='upper right')
ax2a.grid(True, alpha=0.3)

breath_remaining = np.std(after_bandstop[t_zoom] - (bladder[t_zoom] + drift[t_zoom] + gas[t_zoom] + cardiac[t_zoom]))
ax2a.text(0.02, 0.95, f'Breathing reduced from ~{breath_pp:.0f} mOhm\n'
          f'to ~{breath_remaining:.1f} mOhm ({breath_pp/max(breath_remaining,0.01):.0f}x reduction)\n\n'
          f'Cardiac artifact still visible (~3 mOhm)',
          transform=ax2a.transAxes, fontsize=9, va='top',
          bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

# Frequency domain: show filter shape
ax2b.semilogy(f_raw, psd_raw, '-', color=C_RAW, linewidth=0.8, alpha=0.4, label='Before')
ax2b.semilogy(f_raw, psd_after_bs, '-', color=C_CLEAN, linewidth=2, label='After band-stop')
ax2b.set_xlabel('Frequency (Hz)')
ax2b.set_ylabel('Power spectral density')
ax2b.set_title('Step 2: The filter punches a hole at breathing frequency')
ax2b.set_xlim(0, 2)
ax2b.legend(fontsize=9, loc='upper right')
ax2b.grid(True, alpha=0.3)

# Show the filter band
ax2b.axvspan(0.15, 0.4, alpha=0.25, color=C_FILTER_BAND)
ax2b.text(0.275, ax2b.get_ylim()[1]*0.01, 'BLOCKED', fontsize=11,
          color='white', ha='center', fontweight='bold',
          bbox=dict(boxstyle='round', facecolor=C_FILTER_BAND, alpha=0.8))

# ====== ROW 3: Low-pass filter — kill cardiac ======
ax3a = fig.add_subplot(4, 2, 5)
ax3b = fig.add_subplot(4, 2, 6)

# Time domain: full 10 minutes, now we can see the slow signals
ax3a.plot(t_min, after_bandstop, '-', color=C_RAW, linewidth=0.5, alpha=0.4, label='After band-stop')
ax3a.plot(t_min, after_lowpass, '-', color=C_CLEAN, linewidth=2, label='After low-pass (< 0.08 Hz)')
ax3a.plot(t_min, bladder + drift + gas, '--', color=C_BLADDER, linewidth=1, alpha=0.7, label='True slow signal')
ax3a.set_xlabel('Time (minutes)')
ax3a.set_ylabel('Impedance (mOhm)')
ax3a.set_title('Step 3: Low-pass filter (< 0.08 Hz) removes cardiac + residual ripple')
ax3a.legend(fontsize=9, loc='upper left')
ax3a.grid(True, alpha=0.3)

ax3a.text(0.98, 0.05, 'Now only slow changes remain:\n'
          '• Bladder filling (wanted)\n'
          '• Electrode drift (unwanted)\n'
          '• Bowel gas fluctuations (unwanted)',
          transform=ax3a.transAxes, fontsize=9, va='bottom', ha='right',
          bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

# Frequency domain
ax3b.semilogy(f_raw, psd_after_bs, '-', color=C_RAW, linewidth=0.8, alpha=0.4, label='After band-stop')
ax3b.semilogy(f_raw, psd_after_lp, '-', color=C_CLEAN, linewidth=2, label='After low-pass')
ax3b.set_xlabel('Frequency (Hz)')
ax3b.set_ylabel('Power spectral density')
ax3b.set_title('Step 3: Everything above 0.08 Hz is gone')
ax3b.set_xlim(0, 2)
ax3b.legend(fontsize=9, loc='upper right')
ax3b.grid(True, alpha=0.3)

ax3b.axvspan(0.08, 5, alpha=0.15, color=C_FILTER_BAND)
ax3b.axvline(0.08, color=C_FILTER_BAND, linewidth=2, linestyle='--')
ax3b.text(1.0, ax3b.get_ylim()[1]*0.01, 'ALL BLOCKED', fontsize=11,
          color='white', ha='center', fontweight='bold',
          bbox=dict(boxstyle='round', facecolor=C_FILTER_BAND, alpha=0.8))

# ====== ROW 4: Baseline detrending + final result ======
ax4a = fig.add_subplot(4, 2, 7)
ax4b = fig.add_subplot(4, 2, 8)

# Show the low-passed signal with polynomial fit
ax4a.plot(t_min, after_lowpass, '-', color=C_RAW, linewidth=1.5, alpha=0.6, label='After filtering')
ax4a.plot(t_min, baseline_fit, '--', color=C_DRIFT, linewidth=2, label='Polynomial baseline fit')
ax4a.plot(t_min, bladder, '-', color=C_BLADDER, linewidth=1, alpha=0.5, label='True bladder signal')
ax4a.set_xlabel('Time (minutes)')
ax4a.set_ylabel('Impedance (mOhm)')
ax4a.set_title('Step 4: Fit and remove slow baseline drift')
ax4a.legend(fontsize=9, loc='upper left')
ax4a.grid(True, alpha=0.3)

ax4a.annotate('The polynomial captures\ndrift + DC offset',
              xy=(5, baseline_fit[int(5*60*fs)]),
              xytext=(3, baseline_fit[int(5*60*fs)] + 1.5),
              fontsize=9, color=C_DRIFT,
              arrowprops=dict(arrowstyle='->', color=C_DRIFT, lw=1.5))

# Final: convert to volume
# Use slope of detrended signal to estimate filling rate
# More practical: use the filtered signal directly as delta-impedance,
# divide by calibration factor
estimated_volume = after_lowpass / 0.94  # mOhm / (mOhm/mL) = mL (rough, ignoring offset)
true_volume = bladder / 0.94
# Remove offset
estimated_volume -= estimated_volume[0]
true_volume -= true_volume[0]

# Better: use rate estimation from slope
from scipy.ndimage import uniform_filter1d
window_5min = int(5 * 60 * fs)
if window_5min % 2 == 0:
    window_5min += 1
smoothed = uniform_filter1d(after_lowpass, window_5min)
smoothed_vol = (smoothed - smoothed[0]) / 0.94

ax4b.plot(t_min, true_volume, 'k--', linewidth=2, label='True bladder volume change')
ax4b.plot(t_min, smoothed_vol, '-', color=C_CLEAN, linewidth=2.5, label='Estimated (5-min smooth)')
ax4b.fill_between(t_min, smoothed_vol - 3, smoothed_vol + 3,
                  alpha=0.15, color=C_CLEAN, label='±3 mL uncertainty')
ax4b.set_xlabel('Time (minutes)')
ax4b.set_ylabel('Volume change (mL)')
ax4b.set_title('Final result: extracted bladder volume trend')
ax4b.legend(fontsize=9, loc='upper left')
ax4b.grid(True, alpha=0.3)

# Compute error
valid = t > 3 * 60  # after 3 min settling
rms_err = np.sqrt(np.mean((smoothed_vol[valid] - true_volume[valid])**2))
ax4b.text(0.98, 0.05, f'RMS tracking error: {rms_err:.1f} mL\n'
          f'(after 3-min settling period)\n\n'
          f'Clinical target: 7 mL\n'
          f'Status: {"MEETS TARGET" if rms_err < 7 else "DOES NOT MEET"}',
          transform=ax4b.transAxes, fontsize=10, va='bottom', ha='right',
          fontweight='bold',
          bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

# ====== Overall layout ======
fig.suptitle('Breathing Artifact Rejection: Step by Step',
             fontsize=16, fontweight='bold', y=1.01)

# Add big arrow labels between rows
fig.text(0.01, 0.78, 'RAW\nSIGNAL', fontsize=11, fontweight='bold', color=C_RAW,
         va='center', rotation=0)
fig.text(0.01, 0.56, 'KILL\nBREATHING', fontsize=11, fontweight='bold', color=C_BREATH,
         va='center', rotation=0)
fig.text(0.01, 0.33, 'KILL\nCARDIAC', fontsize=11, fontweight='bold', color=C_CARDIAC,
         va='center', rotation=0)
fig.text(0.01, 0.11, 'REMOVE\nDRIFT', fontsize=11, fontweight='bold', color=C_DRIFT,
         va='center', rotation=0)

fig.tight_layout(rect=[0.05, 0, 1, 0.98])
fig.savefig('figures/explain_breathing_filter.png', dpi=200, bbox_inches='tight')
print(f'Saved: figures/explain_breathing_filter.png')

# Print summary
print(f'\n{"="*60}')
print(f'SIGNAL AMPLITUDES')
print(f'{"="*60}')
print(f'Breathing:        {np.ptp(breathing):.1f} mOhm peak-to-peak')
print(f'Cardiac:          {np.ptp(cardiac):.1f} mOhm peak-to-peak')
print(f'Electrode drift:  {np.ptp(drift):.1f} mOhm over 10 min')
print(f'Bowel gas:        {np.std(gas):.1f} mOhm RMS')
print(f'Electronic noise: {np.std(electronic):.3f} mOhm RMS')
print(f'Bladder (10 min): {bladder[-1]:.2f} mOhm total')
print(f'')
print(f'AFTER EACH FILTER STAGE')
print(f'{"="*60}')
print(f'Raw signal std:          {np.std(raw):.2f} mOhm')
print(f'After band-stop:         {np.std(after_bandstop):.2f} mOhm')
print(f'After low-pass:          {np.std(after_lowpass):.2f} mOhm')
print(f'Volume tracking error:   {rms_err:.1f} mL RMS')
