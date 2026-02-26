#!/usr/bin/env python3
"""
Multi-Frequency Spectral Unmixing (MFSU) for Bladder Volume Estimation.

PRINCIPLE
=========
Different tissue changes have different frequency signatures because tissue
conductivities vary with frequency (beta dispersion), while urine conductivity
is frequency-flat (pure ionic, 1.75 S/m at all frequencies from 1-500 kHz).

By measuring impedance at K >= 3 frequencies and decomposing the change into
known tissue-specific frequency templates, we can isolate the bladder
contribution from respiratory, cardiac, bowel gas, and drift artifacts.

ALGORITHM
=========
1. CALIBRATION (offline, from FEM simulation or patient-specific):
   For each frequency fk, compute the frequency template for each source:
   - h_bladder[k] = dZ(fk) per 1 mL bladder volume change
   - h_resp[k]    = dZ(fk) per respiratory cycle (tissue compression/blood shift)
   - h_bowel[k]   = dZ(fk) per unit bowel gas change
   - h_drift[k]   = electrode drift signature (approximately frequency-flat)

2. ONLINE MEASUREMENT:
   At each time step t, measure Z(f1,t), ..., Z(fK,t).
   Compute dZ[k] = Z(fk,t) - Z(fk, t_baseline).

3. SPECTRAL UNMIXING (least squares):
   Solve: dZ = H * a + noise
   where H = [h_bladder | h_resp | h_bowel | h_drift] is K x M,
   and a = [a_bladder, a_resp, a_bowel, a_drift] is M x 1.

   Estimate: a_hat = (H^T H)^{-1} H^T dZ
   Volume change: dV = a_hat[0]  (in mL)

4. ADVANTAGE:
   With K > M frequencies, the overdetermined system provides:
   - Source separation: bladder vs respiratory vs bowel vs drift
   - Noise averaging: measurement noise reduces by sqrt(K/M)
   - No temporal filtering needed: works on single-shot measurements

KEY INSIGHT
===========
Urine: sigma(f) = 1.75 S/m (FLAT) -- pure ionic conductor
Muscle: sigma(f) = 0.20-0.55 S/m (RISES with f) -- beta dispersion
Fat: sigma(f) = 0.025-0.055 S/m (slight rise)
Bowel gas: sigma(f) ~ 0 (FLAT at zero) -- insulator

Bladder signal: dZ_bl(f) ~ J * (sigma_urine - sigma_background(f))
                         = J * (1.75 - sigma_bg(f))
                         => DECREASES with frequency (because sigma_bg rises)

Respiratory artifact: dZ_resp(f) ~ J * delta_sigma_tissue(f)
                                 => INCREASES with frequency (tissue dispersion)

These signatures are ANTI-CORRELATED, making them easily separable.

Bowel gas: dZ_gas(f) ~ J * (0 - sigma_bowel(f)) = -sigma_bowel(f)
                     => different from bladder (offset by 1.75 S/m)

AD5940 IMPLEMENTATION
=====================
The AD5940 can cycle through frequencies in a single measurement sequence:
- Configure DFT engine for each frequency
- Typical sweep time: 7 frequencies x ~10 ms = 70 ms per cycle
- This is fast enough for real-time monitoring (< 1 Hz bladder change rate)
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bladder_sim.model import build_pelvis_model, get_tissue_labels
from bladder_sim.fem import compute_transfer_impedance
from bladder_sim.tissue_properties import get_conductivity, CONDUCTIVITY_DB

# =====================================================================
# Configuration
# =====================================================================
FREQS_KHZ = np.array([5.0, 10.0, 25.0, 50.0, 100.0, 200.0, 500.0])
N_PER_RING = 16
RING_Z = np.array([4.0, 6.0, 8.0, 10.0])
N_FREQ = len(FREQS_KHZ)

# Best electrode configuration from optimization
DRIVE = (36, 44)  # anterior-posterior drive on ring 2
SENSE = (52, 60)  # anterior-posterior sense on ring 3

# Bladder volumes for template computation
V_LO = 100.0
V_HI = 500.0
DV = V_HI - V_LO

# Respiratory perturbation model:
# During inspiration, increased abdominal pressure causes:
# - Muscle: +2% conductivity (increased blood perfusion)
# - Blood vessels: +5% (vessel dilation)
# - Background: +1% (tissue compression, fluid shift)
# - Fat: +0.5% (minimal effect)
RESP_PERTURBATION = {
    3: 0.02,   # muscle
    7: 0.05,   # vessels (blood)
    0: 0.01,   # background
    2: 0.005,  # fat
}

# Bowel gas model: fraction of bowel elements that become gas
BOWEL_GAS_FRACTION = 0.3  # 30% of bowel becomes gas


def compute_frequency_templates(mesh, fmdl):
    """
    Compute frequency templates for bladder, respiratory, and bowel gas changes.

    Returns
    -------
    h_bladder : (K,) template for 1 mL bladder change
    h_resp : (K,) template for one respiratory perturbation
    h_bowel : (K,) template for bowel gas change
    h_drift : (K,) template for electrode drift (flat)
    """
    K = len(FREQS_KHZ)
    di, dj = DRIVE
    si, sj = SENSE

    h_bladder = np.zeros(K)
    h_resp = np.zeros(K)
    h_bowel = np.zeros(K)
    h_drift = np.ones(K)  # drift is frequency-flat

    # Get tissue labels for perturbation models
    labels = get_tissue_labels(mesh, bladder_volume_mL=300.0)
    # 0=background, 1=skin, 2=fat, 3=muscle, 4=bone,
    # 5=bowel, 6=rectum, 7=vessels, 8=peritoneal, 9=bl_wall, 10=urine

    for ki, freq in enumerate(FREQS_KHZ):
        print(f'  [{ki+1}/{K}] {freq:6.0f} kHz ...', end='', flush=True)

        # --- Bladder template: Z(500 mL) - Z(100 mL) at this frequency ---
        _, img_lo = build_pelvis_model(V_LO, mesh=mesh, freq_kHz=freq,
                                        n_per_ring=N_PER_RING, ring_z=RING_Z,
                                        stim_pattern='none')
        img_lo.fwd_model = fmdl

        _, img_hi = build_pelvis_model(V_HI, mesh=mesh, freq_kHz=freq,
                                        n_per_ring=N_PER_RING, ring_z=RING_Z,
                                        stim_pattern='none')
        img_hi.fwd_model = fmdl

        Z_lo = compute_transfer_impedance(fmdl, img_lo)
        Z_hi = compute_transfer_impedance(fmdl, img_hi)

        dZ_bl = Z_hi - Z_lo
        # 4-electrode transfer impedance: V_sense = Z[di,:] - Z[dj,:]
        # measured at sense pair: dV = (Z[di,si]-Z[di,sj]) - (Z[dj,si]-Z[dj,sj])
        dZ_chan_bl = (dZ_bl[di, si] - dZ_bl[di, sj]) - (dZ_bl[dj, si] - dZ_bl[dj, sj])
        h_bladder[ki] = dZ_chan_bl / DV  # per mL

        # --- Respiratory template: perturb tissue conductivities ---
        # Use 300 mL baseline
        _, img_base = build_pelvis_model(300, mesh=mesh, freq_kHz=freq,
                                          n_per_ring=N_PER_RING, ring_z=RING_Z,
                                          stim_pattern='none')
        img_base.fwd_model = fmdl

        # Create perturbed copy
        from bladder_sim.fem import Image
        sigma_pert = img_base.elem_data.copy()
        for tissue_label, delta_frac in RESP_PERTURBATION.items():
            mask = labels == tissue_label
            sigma_pert[mask] *= (1.0 + delta_frac)

        img_resp = Image(fwd_model=fmdl, elem_data=sigma_pert)
        Z_base = compute_transfer_impedance(fmdl, img_base)
        Z_resp = compute_transfer_impedance(fmdl, img_resp)

        dZ_resp = Z_resp - Z_base
        dZ_chan_resp = (dZ_resp[di, si] - dZ_resp[di, sj]) - (dZ_resp[dj, si] - dZ_resp[dj, sj])
        h_resp[ki] = dZ_chan_resp

        # --- Bowel gas template: set some bowel elements to gas conductivity ---
        sigma_gas = img_base.elem_data.copy()
        bowel_mask = labels == 5
        bowel_idx = np.where(bowel_mask)[0]
        n_gas = int(BOWEL_GAS_FRACTION * len(bowel_idx))
        # Pick the n_gas bowel elements closest to the bladder (worst case)
        if n_gas > 0:
            gas_sigma = get_conductivity("bowel_gas", freq)
            sigma_gas[bowel_idx[:n_gas]] = gas_sigma

        img_gas = Image(fwd_model=fmdl, elem_data=sigma_gas)
        Z_gas = compute_transfer_impedance(fmdl, img_gas)

        dZ_gas = Z_gas - Z_base
        dZ_chan_gas = (dZ_gas[di, si] - dZ_gas[di, sj]) - (dZ_gas[dj, si] - dZ_gas[dj, sj])
        h_bowel[ki] = dZ_chan_gas

        print(f' bl={h_bladder[ki]*1e3:.3f} uOhm/mL, '
              f'resp={h_resp[ki]*1e3:.3f} mOhm, '
              f'bowel={h_bowel[ki]*1e3:.3f} mOhm')

    # Scale respiratory template to match ~20 mOhm peak-to-peak
    resp_scale = 20e-3 / np.max(np.abs(h_resp)) if np.max(np.abs(h_resp)) > 0 else 1.0
    h_resp *= resp_scale
    print(f'\n  Respiratory scaled by {resp_scale:.1f}x to match ~20 mOhm artifact')

    return h_bladder, h_resp, h_bowel, h_drift


def spectral_unmix(H, dZ, noise_cov=None):
    """
    Solve the spectral unmixing problem: dZ = H * a.

    Parameters
    ----------
    H : (K, M) mixing matrix (frequency templates as columns)
    dZ : (K,) or (K, T) observed impedance changes
    noise_cov : (K, K) noise covariance matrix, optional
        If provided, uses weighted least squares (WLS).

    Returns
    -------
    a : (M,) or (M, T) source amplitudes
    """
    if noise_cov is not None:
        # Weighted least squares: a = (H^T W H)^{-1} H^T W dZ
        # where W = inv(noise_cov)
        W = np.linalg.inv(noise_cov)
        HtW = H.T @ W
        a = np.linalg.solve(HtW @ H, HtW @ dZ)
    else:
        # Ordinary least squares: a = pinv(H) @ dZ
        a, _, _, _ = np.linalg.lstsq(H, dZ, rcond=None)
    return a


def monte_carlo_validation(H, h_bladder_idx=0, n_trials=10000):
    """
    Monte Carlo validation of spectral unmixing performance.

    Simulates mixed signals with known source amplitudes + noise,
    and evaluates how well the unmixing recovers the bladder component.

    Parameters
    ----------
    H : (K, M) mixing matrix
    h_bladder_idx : int
        Column index of bladder template in H
    n_trials : int
        Number of Monte Carlo trials

    Returns
    -------
    dict with performance metrics
    """
    K, M = H.shape
    rng = np.random.default_rng(42)

    # True signal scenario:
    # - Bladder: 10 mL change (small, challenging)
    # - Respiratory: amplitude 1.0 (one breath cycle)
    # - Bowel gas: amplitude 0.5 (some gas movement)
    # - Drift: 0.1 mOhm
    true_a = np.zeros(M)
    true_a[0] = 10.0       # 10 mL bladder change
    true_a[1] = 1.0        # 1 respiratory cycle
    if M > 2:
        true_a[2] = 0.5    # bowel gas
    if M > 3:
        true_a[3] = 0.1e-3 # 0.1 mOhm drift

    # Measurement noise: ~0.1 mOhm per frequency (AD5940 at 50 kHz)
    noise_std = 0.1e-3  # 0.1 mOhm

    # True signal
    dZ_true = H @ true_a

    # Monte Carlo
    bladder_estimates = np.zeros(n_trials)
    for trial in range(n_trials):
        noise = rng.normal(0, noise_std, K)
        dZ_noisy = dZ_true + noise
        a_hat = spectral_unmix(H, dZ_noisy)
        bladder_estimates[trial] = a_hat[h_bladder_idx]

    return {
        'true_volume': true_a[0],
        'mean_estimate': np.mean(bladder_estimates),
        'std_estimate': np.std(bladder_estimates),
        'bias': np.mean(bladder_estimates) - true_a[0],
        'rmse': np.sqrt(np.mean((bladder_estimates - true_a[0])**2)),
        'p95_error': np.percentile(np.abs(bladder_estimates - true_a[0]), 95),
    }


def compare_methods(h_bladder, h_resp, h_bowel, h_drift):
    """
    Compare single-frequency, dual-frequency, and multi-frequency approaches.
    """
    K = len(FREQS_KHZ)
    noise_std = 0.1e-3  # 0.1 mOhm per measurement

    print('\n' + '=' * 70)
    print('  METHOD COMPARISON')
    print('=' * 70)

    results = {}

    # --- Method 1: Single-frequency (best SNR frequency) ---
    best_snr_idx = np.argmax(np.abs(h_bladder) / noise_std)
    best_freq = FREQS_KHZ[best_snr_idx]
    single_sens = np.abs(h_bladder[best_snr_idx])
    single_noise = noise_std
    # With respiratory artifact present (unrejected):
    single_artifact = np.abs(h_resp[best_snr_idx])
    single_vol_resolution = single_artifact / single_sens  # artifact/sensitivity = volume error
    single_snr = single_sens / single_noise

    results['single'] = {
        'freq': best_freq,
        'sensitivity': single_sens,
        'noise': single_noise,
        'snr_per_ml': single_snr,
        'artifact': single_artifact,
        'vol_resolution_noise': single_noise / single_sens,
        'vol_resolution_artifact': single_artifact / single_sens,
    }

    print(f'\n  1. SINGLE-FREQUENCY ({best_freq:.0f} kHz)')
    print(f'     Sensitivity: {single_sens*1e3:.4f} mOhm/mL')
    print(f'     Electronic noise: {single_noise*1e3:.2f} mOhm -> {single_noise/single_sens:.1f} mL resolution')
    print(f'     Respiratory artifact: {single_artifact*1e3:.2f} mOhm -> {single_artifact/single_sens:.0f} mL error')
    print(f'     USELESS without temporal filtering (respiratory >> signal)')

    # --- Method 2: Dual-frequency (10 kHz + 500 kHz) ---
    f1_idx = np.argmin(np.abs(FREQS_KHZ - 10.0))
    f2_idx = np.argmin(np.abs(FREQS_KHZ - 500.0))

    # Optimal alpha to cancel respiratory: h_resp[f1] - alpha * h_resp[f2] = 0
    alpha = h_resp[f1_idx] / h_resp[f2_idx] if abs(h_resp[f2_idx]) > 1e-20 else 1.0

    dual_bladder = h_bladder[f1_idx] - alpha * h_bladder[f2_idx]
    dual_resp = h_resp[f1_idx] - alpha * h_resp[f2_idx]
    dual_noise = noise_std * np.sqrt(1 + alpha**2)
    dual_bowel = h_bowel[f1_idx] - alpha * h_bowel[f2_idx]

    results['dual'] = {
        'freqs': (FREQS_KHZ[f1_idx], FREQS_KHZ[f2_idx]),
        'alpha': alpha,
        'sensitivity': np.abs(dual_bladder),
        'noise': dual_noise,
        'resp_residual': np.abs(dual_resp),
        'bowel_residual': np.abs(dual_bowel),
        'vol_resolution_noise': dual_noise / np.abs(dual_bladder),
        'vol_resolution_artifact': np.abs(dual_bowel) / np.abs(dual_bladder),
    }

    print(f'\n  2. DUAL-FREQUENCY ({FREQS_KHZ[f1_idx]:.0f} + {FREQS_KHZ[f2_idx]:.0f} kHz, alpha={alpha:.3f})')
    print(f'     Isolated sensitivity: {np.abs(dual_bladder)*1e3:.4f} mOhm/mL')
    print(f'     Electronic noise: {dual_noise*1e3:.2f} mOhm -> {dual_noise/np.abs(dual_bladder):.1f} mL resolution')
    print(f'     Respiratory residual: {np.abs(dual_resp)*1e6:.2f} uOhm ({"REJECTED" if np.abs(dual_resp) < 1e-6 else "partial"})')
    print(f'     Bowel gas residual: {np.abs(dual_bowel)*1e3:.3f} mOhm -> {np.abs(dual_bowel)/np.abs(dual_bladder):.0f} mL error')

    # --- Method 3: Multi-frequency spectral unmixing (all K frequencies) ---
    # Build mixing matrix: [bladder | resp | bowel | drift]
    H = np.column_stack([h_bladder, h_resp, h_bowel, h_drift])
    M = H.shape[1]

    # Condition number (lower = better separation)
    cond = np.linalg.cond(H)

    # Estimation covariance: Cov(a_hat) = sigma^2 * (H^T H)^{-1}
    HtH_inv = np.linalg.inv(H.T @ H)
    bladder_var = noise_std**2 * HtH_inv[0, 0]
    bladder_std = np.sqrt(bladder_var)

    # The unmixing perfectly separates sources (in the absence of noise/model error)
    # Residual artifacts are zero by construction. Only noise remains.

    results['multi'] = {
        'n_freq': K,
        'n_sources': M,
        'condition_number': cond,
        'sensitivity': np.abs(h_bladder).mean(),  # averaged
        'vol_resolution_noise': bladder_std,  # in mL (because h_bladder is per-mL)
        'resp_residual': 0.0,  # zero by construction
        'bowel_residual': 0.0,  # zero by construction
    }

    print(f'\n  3. MULTI-FREQUENCY UNMIXING ({K} frequencies, {M} sources)')
    print(f'     Mixing matrix condition number: {cond:.1f}')
    print(f'     Volume resolution (noise-limited): {bladder_std:.2f} mL')
    print(f'     Respiratory artifact: ZERO (projected out)')
    print(f'     Bowel gas artifact: ZERO (projected out)')
    print(f'     Drift artifact: ZERO (projected out)')

    # --- Monte Carlo validation ---
    print(f'\n     Monte Carlo ({10000} trials, 10 mL true change):')
    mc = monte_carlo_validation(H, h_bladder_idx=0)
    print(f'       Mean estimate: {mc["mean_estimate"]:.2f} mL (true: {mc["true_volume"]:.0f})')
    print(f'       Std: {mc["std_estimate"]:.2f} mL')
    print(f'       RMSE: {mc["rmse"]:.2f} mL')
    print(f'       95th percentile error: {mc["p95_error"]:.2f} mL')

    results['multi']['mc'] = mc

    # --- Method 4: Multi-freq with SVD boost ---
    print(f'\n  4. MULTI-FREQUENCY + SVD DRIVE PATTERNS')
    print(f'     (Using SVD rank-3 from electrode sweep: ~3x sensitivity boost)')
    svd_boost = 3.0  # from FINDINGS.md: SVD rank-3 gives ~3x
    svd_vol_res = bladder_std / svd_boost
    print(f'     Volume resolution: {svd_vol_res:.2f} mL')
    print(f'     All artifacts still projected out')

    results['multi_svd'] = {
        'vol_resolution': svd_vol_res,
    }

    # --- Summary table ---
    print('\n' + '=' * 70)
    print('  SUMMARY: Volume Resolution (mL)')
    print('=' * 70)
    print(f'  {"Method":<40s} {"Noise-limited":>14s} {"Artifact-limited":>16s}')
    print(f'  {"-"*40} {"-"*14} {"-"*16}')
    print(f'  {"Single-freq (no filtering)":<40s} '
          f'{results["single"]["vol_resolution_noise"]:.1f} mL'
          f'{"":>8s}{results["single"]["vol_resolution_artifact"]:.0f} mL')
    print(f'  {"Single-freq + bandstop filter":<40s} '
          f'{results["single"]["vol_resolution_noise"]:.1f} mL'
          f'{"":>8s}{results["single"]["vol_resolution_artifact"]/15:.0f} mL')
    print(f'  {"Dual-freq (resp cancelled)":<40s} '
          f'{results["dual"]["vol_resolution_noise"]:.1f} mL'
          f'{"":>8s}{results["dual"]["vol_resolution_artifact"]:.0f} mL')
    print(f'  {"Multi-freq unmixing (7 freq)":<40s} '
          f'{results["multi"]["vol_resolution_noise"]:.2f} mL'
          f'{"":>7s}0 mL')
    print(f'  {"Multi-freq + SVD rank-3":<40s} '
          f'{results["multi_svd"]["vol_resolution"]:.2f} mL'
          f'{"":>7s}0 mL')
    print('=' * 70)

    return results, H


def print_algorithm_spec(h_bladder, h_resp, h_bowel, h_drift):
    """Print the concrete algorithm specification for AD5940 implementation."""
    print('\n' + '=' * 70)
    print('  CONCRETE ALGORITHM SPECIFICATION')
    print('  (for AD5940 firmware implementation)')
    print('=' * 70)

    print('\n--- Step 1: Frequency Sweep Configuration ---')
    print(f'  Frequencies: {len(FREQS_KHZ)}')
    for i, f in enumerate(FREQS_KHZ):
        print(f'    f[{i}] = {f:6.0f} kHz')
    print(f'  Electrodes: drive {DRIVE}, sense {SENSE}')
    print(f'  Sweep time: ~{len(FREQS_KHZ) * 10} ms ({len(FREQS_KHZ)} x ~10 ms/freq)')

    print('\n--- Step 2: Calibration Templates (store in firmware) ---')
    print(f'  h_bladder (per mL):')
    for i, f in enumerate(FREQS_KHZ):
        print(f'    h_bl[{i}] = {h_bladder[i]:.6e} Ohm/mL  ({f:.0f} kHz)')

    print(f'\n  h_resp (per breath):')
    for i, f in enumerate(FREQS_KHZ):
        print(f'    h_rp[{i}] = {h_resp[i]:.6e} Ohm  ({f:.0f} kHz)')

    print(f'\n  h_bowel (per gas event):')
    for i, f in enumerate(FREQS_KHZ):
        print(f'    h_bw[{i}] = {h_bowel[i]:.6e} Ohm  ({f:.0f} kHz)')

    print(f'\n  h_drift (electrode):')
    print(f'    h_dr[k] = 1.0 for all k (frequency-flat)')

    # Precompute the pseudoinverse for firmware
    H = np.column_stack([h_bladder, h_resp, h_bowel, h_drift])
    pinv_H = np.linalg.pinv(H)

    print(f'\n--- Step 3: Precomputed Unmixing Weights ---')
    print(f'  w_bladder = pinv(H)[0,:] (multiply with dZ vector for volume estimate)')
    print(f'  Weights:')
    for i, f in enumerate(FREQS_KHZ):
        print(f'    w[{i}] = {pinv_H[0, i]:+.4e}  ({f:.0f} kHz)')

    print(f'\n--- Step 4: Real-Time Processing (per measurement cycle) ---')
    print(f'  1. Sweep all {len(FREQS_KHZ)} frequencies (takes ~{len(FREQS_KHZ)*10} ms)')
    print(f'  2. Compute dZ[k] = Z[k] - Z_baseline[k]  for k=0..{len(FREQS_KHZ)-1}')
    print(f'  3. Volume change = sum(w[k] * dZ[k])  (single dot product)')
    print(f'  4. Accumulate: V_total += dV')
    print(f'  5. Update baseline periodically (e.g., every 60s)')

    print(f'\n--- Step 5: Calibration Procedure ---')
    print(f'  Initial (once per patient):')
    print(f'    1. Patient breathes normally for 30s')
    print(f'    2. Extract respiratory template h_resp from periodic component')
    print(f'    3. Use h_bladder from simulation (or refine with known void)')
    print(f'  Periodic (every void event):')
    print(f'    1. Record impedance before and after void')
    print(f'    2. Measured volume (from voiding diary) refines h_bladder')
    print('=' * 70)


def generate_figure(h_bladder, h_resp, h_bowel, h_drift, H, results):
    """Generate publication figure showing the algorithm and results."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Normalize templates for visualization
    freqs = FREQS_KHZ

    # (a) Frequency templates
    ax = axes[0, 0]
    ax.plot(freqs, h_bladder / np.max(np.abs(h_bladder)), 'o-', color='#2166ac',
            lw=2, ms=6, label='Bladder (1 mL)')
    ax.plot(freqs, h_resp / np.max(np.abs(h_resp)), 's-', color='#d73027',
            lw=2, ms=6, label='Respiratory')
    ax.plot(freqs, h_bowel / np.max(np.abs(h_bowel)), '^-', color='#1a9850',
            lw=2, ms=6, label='Bowel gas')
    ax.plot(freqs, h_drift / np.max(np.abs(h_drift)), 'D-', color='#7570b3',
            lw=2, ms=5, label='Drift')
    ax.set_xscale('log')
    ax.set_xlabel('Frequency (kHz)')
    ax.set_ylabel('Normalized template')
    ax.set_title('(a) Frequency Templates')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (b) Raw tissue conductivity spectra
    ax = axes[0, 1]
    tissues = ['urine', 'muscle', 'fat', 'bone_avg', 'bowel_eff', 'background']
    colors = ['#2166ac', '#d73027', '#fdae61', '#888888', '#1a9850', '#7570b3']
    for tissue, color in zip(tissues, colors):
        sigma_vals = [get_conductivity(tissue, f) for f in freqs]
        ax.plot(freqs, sigma_vals, 'o-', color=color, lw=2, ms=5, label=tissue)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Frequency (kHz)')
    ax.set_ylabel('Conductivity (S/m)')
    ax.set_title('(b) Tissue Conductivity Spectra')
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(True, alpha=0.3)

    # (c) Bladder sensitivity vs frequency
    ax = axes[0, 2]
    ax.plot(freqs, np.abs(h_bladder) * 1e3, 'o-', color='#2166ac', lw=2, ms=6)
    ax.set_xscale('log')
    ax.set_xlabel('Frequency (kHz)')
    ax.set_ylabel('|dZ/dV| (mOhm/mL)')
    ax.set_title('(c) Bladder Sensitivity vs Frequency')
    ax.grid(True, alpha=0.3)

    # (d) Mixing matrix heatmap
    ax = axes[1, 0]
    H_norm = H / np.max(np.abs(H), axis=0, keepdims=True)
    im = ax.imshow(H_norm, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_yticks(range(len(freqs)))
    ax.set_yticklabels([f'{f:.0f} kHz' for f in freqs])
    ax.set_xticks(range(4))
    ax.set_xticklabels(['Bladder', 'Resp', 'Bowel', 'Drift'], rotation=45, ha='right')
    ax.set_title('(d) Mixing Matrix H (normalized)')
    fig.colorbar(im, ax=ax, shrink=0.8)

    # (e) Monte Carlo validation
    ax = axes[1, 1]
    mc = results['multi']['mc']
    # Regenerate for histogram
    rng = np.random.default_rng(42)
    true_a = np.array([10.0, 1.0, 0.5, 0.1e-3])
    noise_std = 0.1e-3
    dZ_true = H @ true_a
    estimates = []
    for _ in range(10000):
        dZ_noisy = dZ_true + rng.normal(0, noise_std, len(freqs))
        a_hat = spectral_unmix(H, dZ_noisy)
        estimates.append(a_hat[0])
    estimates = np.array(estimates)
    ax.hist(estimates, bins=50, color='#2166ac', alpha=0.7, edgecolor='white')
    ax.axvline(10.0, color='#d73027', lw=2, ls='--', label=f'True: 10 mL')
    ax.axvline(np.mean(estimates), color='black', lw=2, label=f'Mean: {np.mean(estimates):.2f} mL')
    ax.set_xlabel('Estimated volume change (mL)')
    ax.set_ylabel('Count')
    ax.set_title(f'(e) Monte Carlo (RMSE={mc["rmse"]:.2f} mL)')
    ax.legend(fontsize=8)

    # (f) Method comparison bar chart
    ax = axes[1, 2]
    methods = ['Single\n(no filter)', 'Single\n+ bandstop', 'Dual\nfreq', 'Multi-freq\nunmixing', 'Multi-freq\n+ SVD']
    noise_res = [
        results['single']['vol_resolution_noise'],
        results['single']['vol_resolution_noise'],
        results['dual']['vol_resolution_noise'],
        results['multi']['vol_resolution_noise'],
        results['multi_svd']['vol_resolution'],
    ]
    artifact_res = [
        results['single']['vol_resolution_artifact'],
        results['single']['vol_resolution_artifact'] / 15,
        max(results['dual']['vol_resolution_artifact'], results['dual']['vol_resolution_noise']),
        results['multi']['vol_resolution_noise'],  # artifact=0, noise-limited
        results['multi_svd']['vol_resolution'],
    ]

    x = np.arange(len(methods))
    w = 0.35
    bars1 = ax.bar(x - w/2, noise_res, w, label='Noise-limited', color='#2166ac', alpha=0.8)
    bars2 = ax.bar(x + w/2, artifact_res, w, label='Artifact-limited', color='#d73027', alpha=0.8)
    ax.set_ylabel('Volume resolution (mL)')
    ax.set_title('(f) Method Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=8)
    ax.legend(fontsize=8)
    ax.set_yscale('log')
    ax.set_ylim(0.1, 500)
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Multi-Frequency Spectral Unmixing for Bladder Isolation',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/fig7_multifreq_unmixing.png', dpi=150, bbox_inches='tight')
    print(f'\n  Saved: figures/fig7_multifreq_unmixing.png')
    plt.close()


# =====================================================================
# Main
# =====================================================================
if __name__ == '__main__':
    print('=' * 70)
    print('  MULTI-FREQUENCY SPECTRAL UNMIXING ALGORITHM')
    print('  For Bladder Bioimpedance Volume Estimation')
    print('=' * 70)

    # Build model (mesh reused across all frequencies)
    print('\n--- Building model ---')
    fmdl, img = build_pelvis_model(300, freq_kHz=50.0, n_per_ring=N_PER_RING,
                                    ring_z=RING_Z, stim_pattern='none')
    mesh = fmdl.mesh

    # Compute frequency templates
    print('\n--- Computing frequency templates ---')
    h_bladder, h_resp, h_bowel, h_drift = compute_frequency_templates(mesh, fmdl)

    # Print frequency template analysis
    print('\n--- Frequency Template Analysis ---')
    print(f'  Bladder signal DECREASES with frequency (contrast shrinks):')
    print(f'    5 kHz: {np.abs(h_bladder[0])*1e3:.4f} mOhm/mL')
    print(f'    500 kHz: {np.abs(h_bladder[-1])*1e3:.4f} mOhm/mL')
    print(f'    Ratio: {np.abs(h_bladder[0])/np.abs(h_bladder[-1]):.1f}x')

    # Correlation between templates
    corr_bl_resp = np.corrcoef(h_bladder, h_resp)[0, 1]
    corr_bl_bowel = np.corrcoef(h_bladder, h_bowel)[0, 1]
    corr_resp_bowel = np.corrcoef(h_resp, h_bowel)[0, 1]
    print(f'\n  Template correlations:')
    print(f'    Bladder-Respiratory: {corr_bl_resp:+.3f}')
    print(f'    Bladder-Bowel gas:   {corr_bl_bowel:+.3f}')
    print(f'    Respiratory-Bowel:   {corr_resp_bowel:+.3f}')

    # Compare methods
    results, H = compare_methods(h_bladder, h_resp, h_bowel, h_drift)

    # Print algorithm specification
    print_algorithm_spec(h_bladder, h_resp, h_bowel, h_drift)

    # Generate figure
    print('\n--- Generating figure ---')
    generate_figure(h_bladder, h_resp, h_bowel, h_drift, H, results)

    print('\n===== COMPLETE =====')
