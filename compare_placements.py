#!/usr/bin/env python3
"""Compare anterior-only vs AP (mixed) electrode placements."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bladder_sim.model import build_pelvis_model
from bladder_sim.fem import compute_transfer_impedance

n_per_ring = 16
ring_z = np.array([4.0, 6.0, 8.0, 10.0])
freq_kHz = 50.0

# Build model
fmdl, img = build_pelvis_model(300, freq_kHz=freq_kHz, n_per_ring=n_per_ring,
                                ring_z=ring_z, stim_pattern='none')
mesh = fmdl.mesh
n_elec = mesh.n_electrodes
elec_pos = np.array([e.center for e in mesh.electrodes])

# Transfer impedance at 100 and 500 mL
print('Computing transfer impedance at 100 mL...')
_, img_lo = build_pelvis_model(100, mesh=mesh, freq_kHz=freq_kHz,
                                n_per_ring=n_per_ring, ring_z=ring_z, stim_pattern='none')
img_lo.fwd_model = fmdl

print('Computing transfer impedance at 500 mL...')
_, img_hi = build_pelvis_model(500, mesh=mesh, freq_kHz=freq_kHz,
                                n_per_ring=n_per_ring, ring_z=ring_z, stim_pattern='none')
img_hi.fwd_model = fmdl

Z_lo = compute_transfer_impedance(fmdl, img_lo)
Z_hi = compute_transfer_impedance(fmdl, img_hi)
dZ = Z_hi - Z_lo
dV = 400.0

# Classify electrodes
anterior_mask = elec_pos[:, 1] > 0
anterior_elecs = np.where(anterior_mask)[0]
posterior_elecs = np.where(~anterior_mask)[0]
print(f'Anterior electrodes: {len(anterior_elecs)}, Posterior: {len(posterior_elecs)}')

# Exhaustive 4-electrode search
best_all = 0
best_all_cfg = None
best_ant = 0
best_ant_cfg = None
best_ap = 0
best_ap_cfg = None

for i in range(n_elec):
    for j in range(i+1, n_elec):
        dZ_drv = dZ[i, :] - dZ[j, :]
        for k in range(n_elec):
            for l in range(k+1, n_elec):
                if k == i or k == j or l == i or l == j:
                    continue
                sens = abs(dZ_drv[k] - dZ_drv[l]) / dV

                all_4 = [i, j, k, l]
                all_ant = all(anterior_mask[e] for e in all_4)
                has_ant = any(anterior_mask[e] for e in all_4)
                has_post = any(not anterior_mask[e] for e in all_4)

                if sens > best_all:
                    best_all = sens
                    best_all_cfg = (i, j, k, l)
                if all_ant and sens > best_ant:
                    best_ant = sens
                    best_ant_cfg = (i, j, k, l)
                if has_ant and has_post and sens > best_ap:
                    best_ap = sens
                    best_ap_cfg = (i, j, k, l)

def describe_cfg(cfg, sens):
    i, j, k, l = cfg
    print(f'    Drive: elec {i} -> {j},  Sense: elec {k} -> {l}')
    print(f'    Sensitivity: {sens*1e3:.4f} mOhm/mL')
    for e in cfg:
        pos = elec_pos[e]
        ring = e // n_per_ring
        side = "ANTERIOR" if anterior_mask[e] else "POSTERIOR"
        print(f'      Elec {e:2d} (ring {ring}, z={ring_z[ring]:.0f} cm): '
              f'X={pos[0]:+.1f}, Y={pos[1]:+.1f} [{side}]')

print()
print('=' * 60)
print('  ELECTRODE PLACEMENT COMPARISON')
print('=' * 60)
print()

print('--- Global best (any placement) ---')
describe_cfg(best_all_cfg, best_all)
print()

print('--- Best anterior-only (all 4 electrodes on abdomen) ---')
describe_cfg(best_ant_cfg, best_ant)
print()

print('--- Best anterior-posterior (mixed) ---')
describe_cfg(best_ap_cfg, best_ap)
print()

print('=' * 60)
print('  SUMMARY')
print('=' * 60)
ratio_ant = best_ant / best_all
ratio_ap = best_ap / best_all
print(f'  Global best:       {best_all*1e3:.4f} mOhm/mL')
print(f'  Anterior-only:     {best_ant*1e3:.4f} mOhm/mL  ({ratio_ant*100:.0f}% of global)')
print(f'  Anterior-posterior: {best_ap*1e3:.4f} mOhm/mL  ({ratio_ap*100:.0f}% of global)')
print(f'')
print(f'  Sensitivity loss from anterior-only: {(1-ratio_ant)*100:.0f}%')
print(f'  -> Anterior-only is practical for a supra-pubic belt.')
print(f'     No electrodes on the back needed.')
print('=' * 60)
