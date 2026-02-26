#!/usr/bin/env python3
"""
Sweep BMI (fat thickness) and analyze how SVD weights and sensitivity change.

Fat thickness mapping (approximate):
  BMI ~18 (lean):    0.5 cm
  BMI ~22 (normal):  1.0 cm
  BMI ~25 (avg):     1.5 cm (default)
  BMI ~28 (overweight): 2.5 cm
  BMI ~32 (obese):   3.5 cm
  BMI ~36 (obese+):  4.5 cm
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from bladder_sim.model import build_pelvis_model, TORSO_RX, TORSO_RY, torso_dimensions
from bladder_sim.fem import compute_transfer_impedance

N_PER_RING = 4
RING_Z = np.array([9.0, 10.0])

# BMI-to-fat mapping
bmi_fat_map = [
    (18, 0.5, 'Lean'),
    (22, 1.0, 'Normal'),
    (25, 1.5, 'Average'),
    (28, 2.5, 'Overweight'),
    (32, 3.5, 'Obese I'),
    (36, 4.5, 'Obese II'),
]

# Build electrode pairs list
def get_pairs(n_elec):
    pairs = []
    for i in range(n_elec):
        for j in range(i+1, n_elec):
            pairs.append((i, j))
    return pairs

# Store results
results = []
all_u1 = []  # SVD drive weight vectors
all_v1 = []  # SVD sense weight vectors

for bmi, fat_cm, label in bmi_fat_map:
    print(f'\n{"="*60}')
    print(f'BMI ~{bmi} ({label}): fat = {fat_cm} cm')
    print(f'{"="*60}')

    # Build mesh for this fat thickness
    fmdl, img = build_pelvis_model(300, freq_kHz=50.0, n_per_ring=N_PER_RING,
                                    ring_z=RING_Z, stim_pattern='none',
                                    fat_thick=fat_cm)
    mesh = fmdl.mesh
    n_elec = mesh.n_electrodes

    # Compute transfer impedances at two volumes
    _, img_lo = build_pelvis_model(100, mesh=mesh, freq_kHz=50.0,
                                    n_per_ring=N_PER_RING, ring_z=RING_Z,
                                    stim_pattern='none', fat_thick=fat_cm)
    img_lo.fwd_model = fmdl
    _, img_hi = build_pelvis_model(500, mesh=mesh, freq_kHz=50.0,
                                    n_per_ring=N_PER_RING, ring_z=RING_Z,
                                    stim_pattern='none', fat_thick=fat_cm)
    img_hi.fwd_model = fmdl

    Z_lo = compute_transfer_impedance(fmdl, img_lo)
    Z_hi = compute_transfer_impedance(fmdl, img_hi)

    dV = 400.0
    M_bladder = (Z_hi - Z_lo) / dV * 1000  # mOhm/mL

    # Build full pair-wise matrix
    pairs = get_pairs(n_elec)
    n_pairs = len(pairs)
    M_full = np.zeros((n_pairs, n_pairs))
    for di, (d1, d2) in enumerate(pairs):
        for si, (s1, s2) in enumerate(pairs):
            M_full[di, si] = (M_bladder[d1, s1] - M_bladder[d1, s2]
                             - M_bladder[d2, s1] + M_bladder[d2, s2])

    # SVD
    U, S, Vt = np.linalg.svd(M_full)
    u1 = U[:, 0]
    v1 = Vt[0, :]

    # Fix sign ambiguity: make largest component positive
    if u1[np.abs(u1).argmax()] < 0:
        u1 = -u1
        v1 = -v1

    # Best tetrapolar
    best_idx = np.unravel_index(np.abs(M_full).argmax(), M_full.shape)
    tetra_sens = abs(M_full[best_idx])

    # SVD sensitivities
    svd1_sens = S[0]
    svd3_sens = np.sqrt(S[0]**2 + S[1]**2 + S[2]**2)  # rank-3 cumulative

    print(f'  Tetrapolar best: {tetra_sens:.4f} mOhm/mL')
    print(f'  SVD rank-1:      {svd1_sens:.4f} mOhm/mL')
    print(f'  SVD rank-3:      {svd3_sens:.4f} mOhm/mL')
    print(f'  SVD/Tetra ratio:  {svd1_sens/tetra_sens:.1f}x')

    results.append({
        'bmi': bmi, 'fat_cm': fat_cm, 'label': label,
        'tetra': tetra_sens, 'svd1': svd1_sens, 'svd3': svd3_sens,
        'singular_values': S[:10].copy(),
        'best_tetra_idx': best_idx,
    })
    all_u1.append(u1.copy())
    all_v1.append(v1.copy())

# ============================================================
# Compute weight stability: correlation between SVD weights at different BMIs
# ============================================================

n_configs = len(results)
u_corr = np.zeros((n_configs, n_configs))
v_corr = np.zeros((n_configs, n_configs))
for i in range(n_configs):
    for j in range(n_configs):
        u_corr[i, j] = abs(np.dot(all_u1[i], all_u1[j]))
        v_corr[i, j] = abs(np.dot(all_v1[i], all_v1[j]))

# What if we use the "average BMI" weights on all patients?
# Cross-sensitivity: apply BMI=25 weights to each BMI's measurement matrix
avg_idx = 2  # BMI 25 (index 2)
u_avg = all_u1[avg_idx]
v_avg = all_v1[avg_idx]

print(f'\n{"="*60}')
print(f'CROSS-BMI WEIGHT STABILITY')
print(f'{"="*60}')
print(f'\nUsing BMI=25 (average) SVD weights on all body types:')
print(f'{"BMI":<6} {"Fat (cm)":<10} {"Optimal SVD":>12} {"Using avg weights":>18} {"Efficiency":>12}')
print('-' * 60)

# We need to recompute M_full for each BMI to apply cross-weights
# Actually we already have them implicitly through the SVD — let me recompute
cross_results = []
for idx, (bmi, fat_cm, label) in enumerate(bmi_fat_map):
    # Rebuild M_full for this BMI
    fmdl, img = build_pelvis_model(300, freq_kHz=50.0, n_per_ring=N_PER_RING,
                                    ring_z=RING_Z, stim_pattern='none',
                                    fat_thick=fat_cm)
    mesh = fmdl.mesh

    _, img_lo = build_pelvis_model(100, mesh=mesh, freq_kHz=50.0,
                                    n_per_ring=N_PER_RING, ring_z=RING_Z,
                                    stim_pattern='none', fat_thick=fat_cm)
    img_lo.fwd_model = fmdl
    _, img_hi = build_pelvis_model(500, mesh=mesh, freq_kHz=50.0,
                                    n_per_ring=N_PER_RING, ring_z=RING_Z,
                                    stim_pattern='none', fat_thick=fat_cm)
    img_hi.fwd_model = fmdl

    Z_lo = compute_transfer_impedance(fmdl, img_lo)
    Z_hi = compute_transfer_impedance(fmdl, img_hi)

    M_bladder = (Z_hi - Z_lo) / 400.0 * 1000
    pairs = get_pairs(mesh.n_electrodes)
    n_pairs = len(pairs)
    M_full = np.zeros((n_pairs, n_pairs))
    for di, (d1, d2) in enumerate(pairs):
        for si, (s1, s2) in enumerate(pairs):
            M_full[di, si] = (M_bladder[d1, s1] - M_bladder[d1, s2]
                             - M_bladder[d2, s1] + M_bladder[d2, s2])

    optimal = results[idx]['svd1']
    cross = abs(u_avg @ M_full @ v_avg)
    efficiency = cross / optimal * 100
    cross_results.append(cross)
    print(f'{bmi:<6} {fat_cm:<10.1f} {optimal:>10.4f}   {cross:>16.4f}   {efficiency:>10.1f}%')

# ============================================================
# Figure
# ============================================================

fig = plt.figure(figsize=(18, 18))
gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

# Color map for BMIs
cmap = plt.cm.RdYlBu_r
bmi_colors = [cmap(i / (n_configs - 1)) for i in range(n_configs)]

# --- Panel 1: Sensitivity vs fat thickness ---
ax1 = fig.add_subplot(gs[0, 0])
bmis = [r['bmi'] for r in results]
fats = [r['fat_cm'] for r in results]
tetras = [r['tetra'] for r in results]
svd1s = [r['svd1'] for r in results]
svd3s = [r['svd3'] for r in results]

ax1.plot(fats, tetras, 'o-', color='#888888', linewidth=2, markersize=8, label='Tetrapolar (best pair)')
ax1.plot(fats, svd1s, 's-', color='#2980b9', linewidth=2.5, markersize=8, label='SVD rank-1')
ax1.plot(fats, svd3s, '^-', color='#27ae60', linewidth=2, markersize=8, label='SVD rank-3')
ax1.set_xlabel('Fat thickness (cm)')
ax1.set_ylabel('Sensitivity (mOhm/mL)')
ax1.set_title('Sensitivity drops with fat thickness')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Add BMI labels on top axis
ax1_top = ax1.twiny()
ax1_top.set_xlim(ax1.get_xlim())
ax1_top.set_xticks(fats)
ax1_top.set_xticklabels([f'BMI {b}' for b in bmis], fontsize=8)

# Annotate sensitivity loss
loss = (1 - svd1s[-1] / svd1s[0]) * 100
ax1.annotate(f'{loss:.0f}% sensitivity loss\nfrom lean → obese II',
             xy=(fats[-1], svd1s[-1]),
             xytext=(fats[-2], svd1s[0]*0.5),
             fontsize=10, color='red',
             arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

# --- Panel 2: SVD improvement ratio ---
ax2 = fig.add_subplot(gs[0, 1])
ratios = [s/t for s, t in zip(svd1s, tetras)]
ax2.bar(range(n_configs), ratios, color=bmi_colors, edgecolor='black', linewidth=0.8)
ax2.set_xticks(range(n_configs))
ax2.set_xticklabels([f'BMI {b}\n{f} cm' for b, f in zip(bmis, fats)], fontsize=9)
ax2.set_ylabel('SVD rank-1 / Tetrapolar ratio')
ax2.set_title('SVD improvement is consistent across BMI')
ax2.grid(True, alpha=0.3, axis='y')
for i, r in enumerate(ratios):
    ax2.text(i, r + 0.05, f'{r:.1f}x', ha='center', fontsize=10, fontweight='bold')

# --- Panel 3: SVD drive weight vectors ---
ax3 = fig.add_subplot(gs[1, 0])
pairs = get_pairs(8)
pair_labels = [f'{p[0]}-{p[1]}' for p in pairs]
x = np.arange(len(pairs))
width = 0.13

for i, (bmi, fat_cm, label) in enumerate(bmi_fat_map):
    offset = (i - n_configs/2 + 0.5) * width
    ax3.bar(x + offset, all_u1[i], width, color=bmi_colors[i], alpha=0.8,
            edgecolor='black', linewidth=0.3, label=f'BMI {bmi}')
ax3.set_xticks(x)
ax3.set_xticklabels(pair_labels, fontsize=8, rotation=45)
ax3.set_xlabel('Drive pair (electrode i - electrode j)')
ax3.set_ylabel('SVD drive weight (U[:,0])')
ax3.set_title('SVD drive weights across BMI')
ax3.legend(fontsize=8, ncol=3, loc='upper right')
ax3.grid(True, alpha=0.3, axis='y')
ax3.axhline(0, color='black', linewidth=0.5)

# --- Panel 4: Weight correlation matrix ---
ax4 = fig.add_subplot(gs[1, 1])
im = ax4.imshow(u_corr, cmap='YlOrRd', vmin=0.5, vmax=1.0, aspect='equal')
ax4.set_xticks(range(n_configs))
ax4.set_yticks(range(n_configs))
labels = [f'BMI {b}\n({l})' for b, _, l in bmi_fat_map]
ax4.set_xticklabels(labels, fontsize=8)
ax4.set_yticklabels(labels, fontsize=8)
ax4.set_title('Drive weight similarity across BMI\n(|dot product| of U[:,0] vectors)')
for i in range(n_configs):
    for j in range(n_configs):
        color = 'white' if u_corr[i,j] < 0.75 else 'black'
        ax4.text(j, i, f'{u_corr[i,j]:.2f}', ha='center', va='center',
                fontsize=9, fontweight='bold', color=color)
fig.colorbar(im, ax=ax4, shrink=0.8)

# --- Panel 5: Cross-BMI efficiency ---
ax5 = fig.add_subplot(gs[2, 0])
optimal_sens = [r['svd1'] for r in results]
bars1 = ax5.bar(np.arange(n_configs) - 0.2, optimal_sens, 0.35,
                color=bmi_colors, edgecolor='black', linewidth=0.8, alpha=0.8,
                label='Patient-specific SVD weights')
bars2 = ax5.bar(np.arange(n_configs) + 0.2, cross_results, 0.35,
                color='#95a5a6', edgecolor='black', linewidth=0.8, alpha=0.8,
                label='Universal weights (from BMI 25)')
ax5.set_xticks(range(n_configs))
ax5.set_xticklabels([f'BMI {b}' for b in bmis], fontsize=9)
ax5.set_ylabel('Sensitivity (mOhm/mL)')
ax5.set_title('Universal weights vs patient-specific')
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3, axis='y')

for i in range(n_configs):
    eff = cross_results[i] / optimal_sens[i] * 100
    ax5.text(i + 0.2, cross_results[i] + 0.02, f'{eff:.0f}%', ha='center',
             fontsize=9, fontweight='bold', color='#e74c3c' if eff < 80 else '#27ae60')

# --- Panel 6: Summary ---
ax6 = fig.add_subplot(gs[2, 1])
ax6.axis('off')

# Compute summary stats
min_eff = min(cross_results[i] / optimal_sens[i] * 100 for i in range(n_configs))
max_eff = max(cross_results[i] / optimal_sens[i] * 100 for i in range(n_configs))
min_corr = u_corr[np.triu_indices(n_configs, k=1)].min()

summary = f"""KEY FINDINGS

Sensitivity range:
  BMI 18 (lean):     {svd1s[0]:.2f} mOhm/mL
  BMI 25 (average):  {svd1s[2]:.2f} mOhm/mL
  BMI 36 (obese II): {svd1s[-1]:.2f} mOhm/mL
  Drop: {(1-svd1s[-1]/svd1s[0])*100:.0f}% from lean to obese

SVD improvement ratio:
  Min: {min(ratios):.1f}x  Max: {max(ratios):.1f}x
  → SVD benefit is {("CONSISTENT" if max(ratios)-min(ratios) < 1 else "VARIABLE")} across BMI

Weight stability:
  Min weight correlation: {min_corr:.2f}
  → Weights are {("STABLE" if min_corr > 0.8 else "VARIABLE")} across BMI

Universal weights (BMI 25 → all patients):
  Efficiency range: {min_eff:.0f}% – {max_eff:.0f}%
  → {"ONE SET of weights works for everyone!" if min_eff > 80 else "Patient-specific calibration needed"}

Clinical implication:
  {"The same firmware drive patterns work across" if min_eff > 80 else "Need patient-specific"}
  {"all body types. No per-patient SVD calibration needed." if min_eff > 80 else "drive pattern calibration."}
  {"Only the sensitivity SCALING factor (mOhm/mL)" if min_eff > 80 else ""}
  {"needs per-patient calibration (via void event)." if min_eff > 80 else ""}
"""

ax6.text(0.05, 0.95, summary, transform=ax6.transAxes, fontsize=11,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, pad=1))

fig.suptitle('BMI Sweep: How Body Fat Affects SVD-Optimal Bladder Bioimpedance',
             fontsize=15, fontweight='bold', y=1.01)
fig.savefig('figures/bmi_sweep.png', dpi=200, bbox_inches='tight')
print(f'\nSaved: figures/bmi_sweep.png')
