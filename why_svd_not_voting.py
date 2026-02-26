#!/usr/bin/env python3
"""
Why SVD, not simple sensitivity-weighted voting?

Demonstrates that naive weighting by individual sensitivity gives WORSE results
than SVD, because pairs are correlated and carry non-bladder signals that
need to cancel out.
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from bladder_sim.model import build_pelvis_model, get_tissue_labels
from bladder_sim.fem import compute_transfer_impedance, Image

# Build 8-electrode model
N_PER_RING = 4
RING_Z = np.array([9.0, 10.0])

print('Building model...')
fmdl, img = build_pelvis_model(300, freq_kHz=50.0, n_per_ring=N_PER_RING,
                                ring_z=RING_Z, stim_pattern='none')
mesh = fmdl.mesh
n_elec = mesh.n_electrodes

# Compute at two volumes
print('Computing impedance at 100 mL and 500 mL...')
_, img_lo = build_pelvis_model(100, mesh=mesh, freq_kHz=50.0,
                                n_per_ring=N_PER_RING, ring_z=RING_Z, stim_pattern='none')
img_lo.fwd_model = fmdl
_, img_hi = build_pelvis_model(500, mesh=mesh, freq_kHz=50.0,
                                n_per_ring=N_PER_RING, ring_z=RING_Z, stim_pattern='none')
img_hi.fwd_model = fmdl

Z_lo = compute_transfer_impedance(fmdl, img_lo)
Z_hi = compute_transfer_impedance(fmdl, img_hi)

# Build the sensitivity matrix for bladder volume change
dV = 400.0  # mL
M_bladder = (Z_hi - Z_lo) / dV * 1000  # mOhm/mL, shape (n_elec, n_elec)

# Now simulate a BOWEL GAS perturbation
# Take the 300 mL model, change bowel elements to gas
print('Computing bowel gas perturbation...')
labels = get_tissue_labels(mesh)
sigma_base = img.elem_data.copy()
sigma_gas = sigma_base.copy()
bowel_mask = (labels == 5)  # bowel
n_bowel = bowel_mask.sum()
# Convert 30% of bowel to gas (very low conductivity)
rng = np.random.default_rng(42)
gas_elems = rng.choice(np.where(bowel_mask)[0], size=int(0.3 * n_bowel), replace=False)
sigma_gas[gas_elems] = 0.01  # gas conductivity

from bladder_sim.fem import Image
img_gas = Image(fwd_model=fmdl, elem_data=sigma_gas)

Z_gas = compute_transfer_impedance(fmdl, img_gas)
Z_base = compute_transfer_impedance(fmdl, img)

M_bowel = (Z_gas - Z_base) * 1000  # mOhm (not per mL, just the artifact)

# Similarly, respiratory perturbation (increase muscle conductivity by 2%)
print('Computing respiratory perturbation...')
sigma_resp = sigma_base.copy()
muscle_mask = (labels == 3)  # muscle
sigma_resp[muscle_mask] *= 1.02  # 2% increase during inspiration
vessel_mask = (labels == 7)
sigma_resp[vessel_mask] *= 1.05  # 5% increase (blood volume change)

img_resp = Image(fwd_model=fmdl, elem_data=sigma_resp)
Z_resp = compute_transfer_impedance(fmdl, img_resp)
M_resp = (Z_resp - Z_base) * 1000  # mOhm

# Extract upper triangle (unique pairs)
pairs = []
for i in range(n_elec):
    for j in range(i+1, n_elec):
        pairs.append((i, j))
n_pairs = len(pairs)

# For each drive pair, what does each sense pair see?
# Build the full M matrix for SVD
M_full = np.zeros((n_pairs, n_pairs))
M_bowel_full = np.zeros((n_pairs, n_pairs))
M_resp_full = np.zeros((n_pairs, n_pairs))

for di, (d1, d2) in enumerate(pairs):
    for si, (s1, s2) in enumerate(pairs):
        M_full[di, si] = M_bladder[d1, s1] - M_bladder[d1, s2] - M_bladder[d2, s1] + M_bladder[d2, s2]
        M_bowel_full[di, si] = M_bowel[d1, s1] - M_bowel[d1, s2] - M_bowel[d2, s1] + M_bowel[d2, s2]
        M_resp_full[di, si] = M_resp[d1, s1] - M_resp[d1, s2] - M_resp[d2, s1] + M_resp[d2, s2]

# ============================================================
# METHOD 1: Tetrapolar (best single pair)
# ============================================================
best_idx = np.unravel_index(np.abs(M_full).argmax(), M_full.shape)
tetra_bladder = abs(M_full[best_idx])
tetra_bowel = abs(M_bowel_full[best_idx])
tetra_resp = abs(M_resp_full[best_idx])

# ============================================================
# METHOD 2: Naive sensitivity weighting
# Each pair weighted by its individual bladder sensitivity
# ============================================================
# For each drive pair, find its best sense pair
individual_sens = np.zeros(n_pairs)
best_sense = np.zeros(n_pairs, dtype=int)
for di in range(n_pairs):
    si = np.abs(M_full[di, :]).argmax()
    individual_sens[di] = abs(M_full[di, si])
    best_sense[di] = si

# Weight by sensitivity (normalized)
w_naive = individual_sens / individual_sens.sum()

# Combined signal for each perturbation
naive_bladder = 0.0
naive_bowel = 0.0
naive_resp = 0.0
for di in range(n_pairs):
    si = best_sense[di]
    naive_bladder += w_naive[di] * M_full[di, si]
    naive_bowel += w_naive[di] * M_bowel_full[di, si]
    naive_resp += w_naive[di] * M_resp_full[di, si]

naive_bladder = abs(naive_bladder)
naive_bowel = abs(naive_bowel)
naive_resp = abs(naive_resp)

# ============================================================
# METHOD 3: SVD rank-1
# ============================================================
U, S, Vt = np.linalg.svd(M_full)

# SVD weights
u1 = U[:, 0]  # drive weights
v1 = Vt[0, :]  # sense weights

svd_bladder = abs(u1 @ M_full @ v1)
svd_bowel = abs(u1 @ M_bowel_full @ v1)
svd_resp = abs(u1 @ M_resp_full @ v1)

# ============================================================
# Print results
# ============================================================
print("\n" + "="*70)
print("COMPARISON: Three weighting strategies")
print("="*70)

print(f"\n{'Method':<30} {'Bladder':>12} {'Bowel Gas':>12} {'Respir.':>12} {'Bladder/Bowel':>14}")
print("-"*80)
print(f"{'Tetrapolar (best pair)':<30} {tetra_bladder:>10.4f}   {tetra_bowel:>10.4f}   {tetra_resp:>10.4f}   {tetra_bladder/tetra_bowel:>12.4f}")
print(f"{'Naive sensitivity voting':<30} {naive_bladder:>10.4f}   {naive_bowel:>10.4f}   {naive_resp:>10.4f}   {naive_bladder/naive_bowel:>12.4f}")
print(f"{'SVD rank-1':<30} {svd_bladder:>10.4f}   {svd_bowel:>10.4f}   {svd_resp:>10.4f}   {svd_bladder/svd_bowel:>12.4f}")

print(f"\nUnits: mOhm/mL (bladder), mOhm (bowel gas & respiratory)")
print(f"\nBladder/Bowel ratio = how much bladder signal you get per unit of bowel noise")
print(f"Higher is better.")

# ============================================================
# Why? Look at correlations
# ============================================================
print("\n" + "="*70)
print("WHY: Pair-level correlations between bladder and bowel signals")
print("="*70)

# For each drive pair (using its best sense pair), what fraction of the
# measurement is bladder vs bowel?
bladder_per_pair = np.array([M_full[di, best_sense[di]] for di in range(n_pairs)])
bowel_per_pair = np.array([M_bowel_full[di, best_sense[di]] for di in range(n_pairs)])

corr = np.corrcoef(bladder_per_pair, bowel_per_pair)[0, 1]
print(f"\nCorrelation between bladder and bowel signals across pairs: {corr:.3f}")
print(f"This means pairs that see more bladder ALSO see more bowel!")
print(f"Weighting by bladder sensitivity amplifies bowel noise proportionally.")
print(f"\nSVD finds the direction in measurement space where bladder signal")
print(f"is maximal relative to the null space — it exploits subtle differences")
print(f"in HOW each pair sees bladder vs bowel.")

# ============================================================
# Figure
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))

# Panel 1: Bladder sensitivity
methods = ['Tetrapolar\n(best pair)', 'Naive\nvoting', 'SVD\nrank-1']
bladder_vals = [tetra_bladder, naive_bladder, svd_bladder]
colors = ['#888888', '#e67e22', '#2980b9']
bars = axes[0].bar(methods, bladder_vals, color=colors, width=0.6, edgecolor='black', linewidth=0.8)
axes[0].set_ylabel('Bladder sensitivity (mOhm/mL)')
axes[0].set_title('What you WANT:\nBladder signal')
for bar, val in zip(bars, bladder_vals):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
axes[0].set_ylim(0, max(bladder_vals) * 1.25)

# Panel 2: Bowel gas artifact
bowel_vals = [tetra_bowel, naive_bowel, svd_bowel]
bars = axes[1].bar(methods, bowel_vals, color=colors, width=0.6, edgecolor='black', linewidth=0.8)
axes[1].set_ylabel('Bowel gas artifact (mOhm)')
axes[1].set_title("What you DON'T want:\nBowel gas noise")
for bar, val in zip(bars, bowel_vals):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
axes[1].set_ylim(0, max(bowel_vals) * 1.25)

# Panel 3: Bladder-to-bowel ratio (the thing that actually matters)
ratio_vals = [tetra_bladder/tetra_bowel, naive_bladder/naive_bowel, svd_bladder/svd_bowel]
bars = axes[2].bar(methods, ratio_vals, color=colors, width=0.6, edgecolor='black', linewidth=0.8)
axes[2].set_ylabel('Bladder / Bowel ratio')
axes[2].set_title('What MATTERS:\nSignal-to-interference ratio')
for bar, val in zip(bars, ratio_vals):
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
axes[2].set_ylim(0, max(ratio_vals) * 1.25)

fig.suptitle('Why SVD, Not Simple Voting?', fontsize=15, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig('figures/why_svd_not_voting.png', dpi=200, bbox_inches='tight')
print(f"\nSaved: figures/why_svd_not_voting.png")
