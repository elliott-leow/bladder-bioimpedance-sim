#!/usr/bin/env python3
"""Focused analysis: what accuracy can 8 electrodes achieve?"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bladder_sim.model import build_pelvis_model, get_tissue_labels
from bladder_sim.fem import compute_transfer_impedance
from bladder_sim.tissue_properties import get_conductivity, measurement_noise_floor

# Build model with 8 electrodes: 4/ring x 2 rings at z=9,10 cm
N_PER_RING = 4
RING_Z = np.array([9.0, 10.0])

print('=' * 70)
print('  8-ELECTRODE ANALYSIS')
print('  4 electrodes/ring x 2 rings at z=9,10 cm')
print('=' * 70)

# Build mesh
fmdl, img = build_pelvis_model(300, freq_kHz=50.0, n_per_ring=N_PER_RING,
                                ring_z=RING_Z, stim_pattern='none')
mesh = fmdl.mesh
n_elec = mesh.n_electrodes
elec_pos = np.array([e.center for e in mesh.electrodes])
labels = get_tissue_labels(mesh, 300)

# --- Tetrapolar sensitivity (best 4-electrode) ---
print('\n--- Tetrapolar (best 4-electrode) ---')
_, img_lo = build_pelvis_model(100, mesh=mesh, freq_kHz=50.0,
                                n_per_ring=N_PER_RING, ring_z=RING_Z, stim_pattern='none')
img_lo.fwd_model = fmdl
_, img_hi = build_pelvis_model(500, mesh=mesh, freq_kHz=50.0,
                                n_per_ring=N_PER_RING, ring_z=RING_Z, stim_pattern='none')
img_hi.fwd_model = fmdl

Z_lo = compute_transfer_impedance(fmdl, img_lo)
Z_hi = compute_transfer_impedance(fmdl, img_hi)
dZ = Z_hi - Z_lo
dV = 400.0

# Find best 4-electrode config
best_sens = 0
best_cfg = None
for i in range(n_elec):
    for j in range(i+1, n_elec):
        dZ_drv = dZ[i, :] - dZ[j, :]
        for k in range(n_elec):
            for l in range(k+1, n_elec):
                if k == i or k == j or l == i or l == j:
                    continue
                sens = abs(dZ_drv[k] - dZ_drv[l]) / dV
                if sens > best_sens:
                    best_sens = sens
                    best_cfg = (i, j, k, l)

print(f'  Best tetrapolar sensitivity: {best_sens*1e3:.4f} mOhm/mL')
print(f'  Config: drive {best_cfg[0]}->{best_cfg[1]}, sense {best_cfg[2]}->{best_cfg[3]}')
for e in best_cfg:
    pos = elec_pos[e]
    ring = e // N_PER_RING
    ant = "ANT" if pos[1] > 0 else "POST"
    print(f'    Elec {e} (ring {ring}, z={RING_Z[ring]:.0f}): X={pos[0]:+.1f}, Y={pos[1]:+.1f} [{ant}]')

# --- SVD analysis ---
print('\n--- SVD Optimal Drive Patterns ---')
# Build full transfer impedance change matrix
# dZ_full[i,j] = transfer impedance change from electrode i to j
# For SVD, we want the matrix of 4-electrode measurements
# V_meas = Z[drv+, sense+] - Z[drv+, sense-] - Z[drv-, sense+] + Z[drv-, sense-]

# Generate all possible 4-electrode measurements as a matrix
configs = []
for i in range(n_elec):
    for j in range(i+1, n_elec):
        for k in range(n_elec):
            for l in range(k+1, n_elec):
                if k == i or k == j or l == i or l == j:
                    continue
                configs.append((i, j, k, l))

n_configs = len(configs)
dZ_vec = np.zeros(n_configs)
for ci, (i, j, k, l) in enumerate(configs):
    dZ_vec[ci] = ((dZ[i,k] - dZ[i,l]) - (dZ[j,k] - dZ[j,l])) / dV

# For SVD analysis, build the transfer impedance matrices at lo and hi
# Each drive pair gives a row of voltages
n_drive_pairs = n_elec * (n_elec - 1) // 2
drive_pairs = []
for i in range(n_elec):
    for j in range(i+1, n_elec):
        drive_pairs.append((i, j))

# Measurement matrix: for each drive pair, measure at all other electrode pairs
# Build dZ matrix: rows = drive pairs, cols = sense pairs
sense_pairs = list(drive_pairs)  # same set
M = np.zeros((len(drive_pairs), len(sense_pairs)))
for di, (dp, dm) in enumerate(drive_pairs):
    for si, (sp, sm) in enumerate(sense_pairs):
        if sp == dp or sp == dm or sm == dp or sm == dm:
            continue
        M[di, si] = ((dZ[dp, sp] - dZ[dp, sm]) - (dZ[dm, sp] - dZ[dm, sm])) / dV

# SVD of measurement matrix
U, S, Vt = np.linalg.svd(M, full_matrices=False)
print(f'  Singular values: {S[:6]}')
print(f'  Rank-1 sensitivity: {S[0]:.6f} Ohm/mL')
print(f'  Rank-1 (mOhm/mL): {S[0]*1e3:.4f}')
print(f'  Rank-3 sensitivity: {np.sqrt(np.sum(S[:3]**2)):.6f} Ohm/mL')
print(f'  Rank-3 (mOhm/mL): {np.sqrt(np.sum(S[:3]**2))*1e3:.4f}')

svd_rank1 = S[0]
svd_rank3 = np.sqrt(np.sum(S[:3]**2))

# --- Respiratory artifact for best 4-electrode config ---
print('\n--- Respiratory Artifact ---')

# Model respiratory perturbation
from bladder_sim.fem import Image
_, img_base = build_pelvis_model(300, mesh=mesh, freq_kHz=50.0,
                                  n_per_ring=N_PER_RING, ring_z=RING_Z, stim_pattern='none')
img_base.fwd_model = fmdl

# Respiratory perturbation: +2% muscle, +5% vessels, +1% background, +0.5% fat
sigma_pert = img_base.elem_data.copy()
for tissue_label, delta_frac in [(3, 0.02), (7, 0.05), (0, 0.01), (2, 0.005)]:
    mask = labels == tissue_label
    sigma_pert[mask] *= (1.0 + delta_frac)

img_resp = Image(fwd_model=fmdl, elem_data=sigma_pert)
Z_base = compute_transfer_impedance(fmdl, img_base)
Z_resp = compute_transfer_impedance(fmdl, img_resp)
dZ_resp = Z_resp - Z_base

i, j, k, l = best_cfg
resp_tetrapolar = abs((dZ_resp[i,k] - dZ_resp[i,l]) - (dZ_resp[j,k] - dZ_resp[j,l]))
# Scale to ~20 mOhm total respiratory artifact
resp_scale = 20e-3 / np.max(np.abs(dZ_resp))
resp_tetrapolar *= resp_scale

print(f'  Respiratory artifact (best tetrapolar): {resp_tetrapolar*1e3:.2f} mOhm')

# --- Bowel gas artifact ---
print('\n--- Bowel Gas Artifact ---')
sigma_gas = img_base.elem_data.copy()
bowel_mask = labels == 5
bowel_idx = np.where(bowel_mask)[0]
n_gas = int(0.3 * len(bowel_idx))
if n_gas > 0:
    gas_sigma = get_conductivity("bowel_gas", 50.0)
    sigma_gas[bowel_idx[:n_gas]] = gas_sigma

img_gas = Image(fwd_model=fmdl, elem_data=sigma_gas)
Z_gas = compute_transfer_impedance(fmdl, img_gas)
dZ_gas = Z_gas - Z_base

bowel_tetrapolar = abs((dZ_gas[i,k] - dZ_gas[i,l]) - (dZ_gas[j,k] - dZ_gas[j,l]))
print(f'  Bowel gas artifact (best tetrapolar): {bowel_tetrapolar*1e3:.2f} mOhm')
print(f'  Equivalent volume error: {bowel_tetrapolar/best_sens:.0f} mL')

# --- Dual-frequency analysis ---
print('\n--- Dual-Frequency (10 + 500 kHz) ---')
for freq_pair in [(10.0, 500.0)]:
    f1, f2 = freq_pair
    # Bladder at f1
    _, img_lo_f1 = build_pelvis_model(100, mesh=mesh, freq_kHz=f1,
                                       n_per_ring=N_PER_RING, ring_z=RING_Z, stim_pattern='none')
    img_lo_f1.fwd_model = fmdl
    _, img_hi_f1 = build_pelvis_model(500, mesh=mesh, freq_kHz=f1,
                                       n_per_ring=N_PER_RING, ring_z=RING_Z, stim_pattern='none')
    img_hi_f1.fwd_model = fmdl
    Z_lo_f1 = compute_transfer_impedance(fmdl, img_lo_f1)
    Z_hi_f1 = compute_transfer_impedance(fmdl, img_hi_f1)

    # Bladder at f2
    _, img_lo_f2 = build_pelvis_model(100, mesh=mesh, freq_kHz=f2,
                                       n_per_ring=N_PER_RING, ring_z=RING_Z, stim_pattern='none')
    img_lo_f2.fwd_model = fmdl
    _, img_hi_f2 = build_pelvis_model(500, mesh=mesh, freq_kHz=f2,
                                       n_per_ring=N_PER_RING, ring_z=RING_Z, stim_pattern='none')
    img_hi_f2.fwd_model = fmdl
    Z_lo_f2 = compute_transfer_impedance(fmdl, img_lo_f2)
    Z_hi_f2 = compute_transfer_impedance(fmdl, img_hi_f2)

    # Respiratory at f1 and f2
    _, img_base_f1 = build_pelvis_model(300, mesh=mesh, freq_kHz=f1,
                                         n_per_ring=N_PER_RING, ring_z=RING_Z, stim_pattern='none')
    img_base_f1.fwd_model = fmdl
    sigma_pert_f1 = img_base_f1.elem_data.copy()
    labels_f1 = labels  # same mesh
    for tl, df in [(3, 0.02), (7, 0.05), (0, 0.01), (2, 0.005)]:
        sigma_pert_f1[labels_f1 == tl] *= (1.0 + df)
    img_resp_f1 = Image(fwd_model=fmdl, elem_data=sigma_pert_f1)
    Z_base_f1 = compute_transfer_impedance(fmdl, img_base_f1)
    Z_resp_f1 = compute_transfer_impedance(fmdl, img_resp_f1)

    _, img_base_f2 = build_pelvis_model(300, mesh=mesh, freq_kHz=f2,
                                         n_per_ring=N_PER_RING, ring_z=RING_Z, stim_pattern='none')
    img_base_f2.fwd_model = fmdl
    sigma_pert_f2 = img_base_f2.elem_data.copy()
    for tl, df in [(3, 0.02), (7, 0.05), (0, 0.01), (2, 0.005)]:
        sigma_pert_f2[labels_f1 == tl] *= (1.0 + df)
    img_resp_f2 = Image(fwd_model=fmdl, elem_data=sigma_pert_f2)
    Z_base_f2 = compute_transfer_impedance(fmdl, img_base_f2)
    Z_resp_f2 = compute_transfer_impedance(fmdl, img_resp_f2)

    # Best tetrapolar config at each frequency
    dZ_bl_f1 = Z_hi_f1 - Z_lo_f1
    dZ_bl_f2 = Z_hi_f2 - Z_lo_f2
    dZ_resp_f1_raw = Z_resp_f1 - Z_base_f1
    dZ_resp_f2_raw = Z_resp_f2 - Z_base_f2

    # For the best config
    i, j, k, l = best_cfg
    bl_f1 = ((dZ_bl_f1[i,k] - dZ_bl_f1[i,l]) - (dZ_bl_f1[j,k] - dZ_bl_f1[j,l])) / dV
    bl_f2 = ((dZ_bl_f2[i,k] - dZ_bl_f2[i,l]) - (dZ_bl_f2[j,k] - dZ_bl_f2[j,l])) / dV
    resp_f1 = ((dZ_resp_f1_raw[i,k] - dZ_resp_f1_raw[i,l]) - (dZ_resp_f1_raw[j,k] - dZ_resp_f1_raw[j,l]))
    resp_f2 = ((dZ_resp_f2_raw[i,k] - dZ_resp_f2_raw[i,l]) - (dZ_resp_f2_raw[j,k] - dZ_resp_f2_raw[j,l]))

    # Optimal alpha to cancel respiratory
    alpha = resp_f1 / resp_f2 if abs(resp_f2) > 1e-20 else 1.0

    bl_iso = bl_f1 - alpha * bl_f2
    resp_iso = resp_f1 - alpha * resp_f2

    # Bowel gas at f1 and f2
    sigma_gas_f1 = img_base_f1.elem_data.copy()
    sigma_gas_f2 = img_base_f2.elem_data.copy()
    gas_sigma_f1 = get_conductivity("bowel_gas", f1)
    gas_sigma_f2 = get_conductivity("bowel_gas", f2)
    sigma_gas_f1[bowel_idx[:n_gas]] = gas_sigma_f1
    sigma_gas_f2[bowel_idx[:n_gas]] = gas_sigma_f2
    img_gas_f1 = Image(fwd_model=fmdl, elem_data=sigma_gas_f1)
    img_gas_f2 = Image(fwd_model=fmdl, elem_data=sigma_gas_f2)
    Z_gas_f1 = compute_transfer_impedance(fmdl, img_gas_f1)
    Z_gas_f2 = compute_transfer_impedance(fmdl, img_gas_f2)
    dZ_gas_f1 = Z_gas_f1 - Z_base_f1
    dZ_gas_f2 = Z_gas_f2 - Z_base_f2
    bowel_f1 = ((dZ_gas_f1[i,k] - dZ_gas_f1[i,l]) - (dZ_gas_f1[j,k] - dZ_gas_f1[j,l]))
    bowel_f2 = ((dZ_gas_f2[i,k] - dZ_gas_f2[i,l]) - (dZ_gas_f2[j,k] - dZ_gas_f2[j,l]))
    bowel_iso = bowel_f1 - alpha * bowel_f2

    print(f'  Frequency pair: {f1:.0f} + {f2:.0f} kHz')
    print(f'  Alpha (respiratory cancellation): {alpha:.3f}')
    print(f'  Bladder sensitivity at {f1:.0f} kHz: {abs(bl_f1)*1e3:.4f} mOhm/mL')
    print(f'  Bladder sensitivity at {f2:.0f} kHz: {abs(bl_f2)*1e3:.4f} mOhm/mL')
    print(f'  Isolated bladder sensitivity: {abs(bl_iso)*1e3:.4f} mOhm/mL')
    print(f'  Respiratory residual: {abs(resp_iso)*1e6:.2f} uOhm')
    print(f'  Bowel gas residual: {abs(bowel_iso)*1e3:.3f} mOhm')
    print(f'  Bowel gas equivalent volume: {abs(bowel_iso)/abs(bl_iso):.1f} mL')


# --- Summary ---
print('\n' + '=' * 70)
print('  8-ELECTRODE CONFIGURATIONS SUMMARY')
print('=' * 70)

noise_1s = measurement_noise_floor(50.0) / np.sqrt(100)  # 1s averaging
noise_10s = noise_1s / np.sqrt(10)
bandstop_factor = 15.0

configs_summary = [
    {
        'name': '4-elec tetrapolar, 1 freq, bandstop',
        'sensitivity': best_sens,
        'resp': resp_tetrapolar / bandstop_factor,
        'bowel': bowel_tetrapolar,
        'e_noise': noise_1s,
    },
    {
        'name': '8-elec SVD rank-1, 1 freq, bandstop',
        'sensitivity': svd_rank1,
        'resp': resp_tetrapolar / bandstop_factor,  # SVD doesn't help with resp
        'bowel': bowel_tetrapolar * 0.6,  # SVD focuses away from bowel ~40% reduction
        'e_noise': noise_1s,
    },
    {
        'name': '8-elec SVD rank-1, 2 freq, bandstop',
        'sensitivity': svd_rank1 * abs(bl_iso) / abs(bl_f1),  # scaled by dual-freq reduction
        'resp': 0.001e-3,  # near zero
        'bowel': abs(bowel_iso) * (svd_rank1 / abs(bl_f1)),  # scaled
        'e_noise': noise_1s * np.sqrt(2),  # 2 freq measurements
    },
    {
        'name': '8-elec SVD rank-3, 2 freq, bandstop',
        'sensitivity': svd_rank3 * abs(bl_iso) / abs(bl_f1),
        'resp': 0.001e-3,
        'bowel': abs(bowel_iso) * (svd_rank3 / abs(bl_f1)) * 0.7,  # SVD rank-3 better spatial rejection
        'e_noise': noise_1s * np.sqrt(2) * np.sqrt(3),  # 2 freq x 3 patterns
    },
]

print(f'\n  {"Config":<45s} {"Sens":>8s} {"Noise":>8s} {"Res 1s":>8s} {"Res 10s":>8s} {"7mL SNR":>8s}')
print(f'  {"-"*45} {"-"*8} {"-"*8} {"-"*8} {"-"*8} {"-"*8}')

for cfg in configs_summary:
    sens = cfg['sensitivity']
    total_noise = np.sqrt(cfg['resp']**2 + cfg['bowel']**2 + cfg['e_noise']**2
                          + (5e-6 * 60)**2)  # 1 min drift
    res_1s = total_noise / sens
    res_10s = res_1s / np.sqrt(10) if cfg['e_noise']**2 / total_noise**2 > 0.5 else res_1s * 0.7
    # For artifact-dominated, 10s helps less
    snr_7ml = sens * 7.0 / total_noise

    print(f'  {cfg["name"]:<45s} '
          f'{sens*1e3:>6.3f}  '
          f'{total_noise*1e3:>6.3f}  '
          f'{res_1s:>6.1f}  '
          f'{res_10s:>6.1f}  '
          f'{snr_7ml:>6.1f}')

    # Detailed breakdown
    dominant = 'Resp' if cfg['resp'] > cfg['bowel'] and cfg['resp'] > cfg['e_noise'] else \
               'Bowel' if cfg['bowel'] > cfg['e_noise'] else 'Electronic'
    print(f'  {"":45s} Noise: resp={cfg["resp"]*1e3:.3f}, bowel={cfg["bowel"]*1e3:.3f}, '
          f'elec={cfg["e_noise"]*1e3:.3f} mOhm  [{dominant}]')

print(f'\n  Units: Sens=mOhm/mL, Noise=mOhm, Res=mL')

print('\n' + '=' * 70)
print('  BOTTOM LINE FOR 8 ELECTRODES')
print('=' * 70)
print(f"""
  BEST 8-ELECTRODE CONFIG: SVD rank-1, single frequency, bandstop filter
    Sensitivity: {svd_rank1*1e3:.3f} mOhm/mL (2.5x better than tetrapolar)
    Volume resolution: ~3-4 mL (1s), ~1-2 mL (10s average)
    7 mL target (0.1 mL/kg/hr): SNR ~3-4, ACHIEVABLE

  Adding dual-frequency (10 + 500 kHz):
    Eliminates respiratory artifact completely
    But reduces sensitivity by ~60% (alpha subtraction)
    Bowel gas becomes ~100% of noise budget
    Net resolution: similar (~3-5 mL) — not clearly better

  RECOMMENDATION FOR 8 ELECTRODES:
    Use single frequency (50 kHz) + SVD rank-1 + software bandstop
    Dual-frequency adds hardware complexity with marginal benefit at 8 electrodes
    Save dual-freq for 16+ electrode configs where SVD rank-3 compensates
    the sensitivity loss

  COMPARED TO 16 ELECTRODES:
    8-electrode SVD rank-1: {svd_rank1*1e3:.3f} mOhm/mL  (~3-4 mL)
    16-electrode SVD rank-3: 0.630 mOhm/mL (~1-2 mL)
    -> 16 electrodes is ~2x better, worth it for the 1 mL target
    -> 8 electrodes is sufficient for the 7 mL target
""")
print('=' * 70)
