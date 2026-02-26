#!/usr/bin/env python3
"""
Step-by-step SVD explanation using the actual 8-electrode simulation.
Generates a pedagogical figure that walks through the process.
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch
from matplotlib.collections import LineCollection

from bladder_sim.model import (
    build_pelvis_model, TORSO_RX, TORSO_RY,
    bladder_semi_axes, bladder_center_y,
)
from bladder_sim.fem import compute_transfer_impedance

WONG_BLUE = np.array([0.000, 0.447, 0.741])
WONG_VERM = np.array([0.850, 0.325, 0.098])
WONG_GREEN = np.array([0.000, 0.620, 0.451])
WONG_PURPLE = np.array([0.580, 0.404, 0.741])

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.labelsize": 11, "savefig.dpi": 300,
})

# Build 8-electrode model
N_PER_RING = 4
RING_Z = np.array([9.0, 10.0])

print('Building model...')
fmdl, img = build_pelvis_model(300, freq_kHz=50.0, n_per_ring=N_PER_RING,
                                ring_z=RING_Z, stim_pattern='none')
mesh = fmdl.mesh
n_elec = mesh.n_electrodes

print('Computing impedance at 100 mL and 500 mL...')
_, img_lo = build_pelvis_model(100, mesh=mesh, freq_kHz=50.0,
                                n_per_ring=N_PER_RING, ring_z=RING_Z, stim_pattern='none')
img_lo.fwd_model = fmdl
_, img_hi = build_pelvis_model(500, mesh=mesh, freq_kHz=50.0,
                                n_per_ring=N_PER_RING, ring_z=RING_Z, stim_pattern='none')
img_hi.fwd_model = fmdl

Z_lo = compute_transfer_impedance(fmdl, img_lo)
Z_hi = compute_transfer_impedance(fmdl, img_hi)
dZ = (Z_hi - Z_lo) / 400.0  # per mL

# Electrode positions
ex = np.array([e.center[0] for e in mesh.electrodes])
ey = np.array([e.center[1] for e in mesh.electrodes])
ez = np.array([e.center[2] for e in mesh.electrodes])

# Label electrodes
elec_names = []
for i in range(n_elec):
    ring = i // N_PER_RING
    pos_in_ring = i % N_PER_RING
    names = ['Right', 'Front', 'Left', 'Back']
    elec_names.append(f'{names[pos_in_ring]}\nz={RING_Z[ring]:.0f}')

# All drive pairs
drive_pairs = []
for i in range(n_elec):
    for j in range(i+1, n_elec):
        drive_pairs.append((i, j))

n_dp = len(drive_pairs)
pair_names = [f'{i}-{j}' for i, j in drive_pairs]

# Build sensitivity matrix M
M = np.zeros((n_dp, n_dp))
for di, (dp, dm) in enumerate(drive_pairs):
    for si, (sp, sm) in enumerate(drive_pairs):
        if sp == dp or sp == dm or sm == dp or sm == dm:
            continue
        M[di, si] = ((dZ[dp, sp] - dZ[dp, sm]) - (dZ[dm, sp] - dZ[dm, sm]))

# Individual tetrapolar sensitivities (best sense pair for each drive pair)
drive_sensitivities = np.zeros(n_dp)
for di in range(n_dp):
    drive_sensitivities[di] = np.max(np.abs(M[di, :]))

# SVD
U, S, Vt = np.linalg.svd(M, full_matrices=False)

# Best tetrapolar
best_tetra = np.max(np.abs(M))
best_di, best_si = np.unravel_index(np.argmax(np.abs(M)), M.shape)

print(f'\nBest tetrapolar: {best_tetra*1e3:.4f} mOhm/mL')
print(f'  Drive pair: {drive_pairs[best_di]}')
print(f'  Sense pair: {drive_pairs[best_si]}')
print(f'SVD rank-1: {S[0]*1e3:.4f} mOhm/mL')
print(f'Improvement: {S[0]/best_tetra:.1f}x')


# =====================================================================
# Generate the teaching figure
# =====================================================================
fig = plt.figure(figsize=(18, 22))

# Layout: 4 rows
gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.35,
                       top=0.95, bottom=0.03, left=0.06, right=0.97)


def draw_torso(ax, title=''):
    """Draw the torso cross-section with bladder."""
    ax.set_aspect('equal')
    if title:
        ax.set_title(title, fontsize=12)
    torso = Ellipse((0, 0), 2*TORSO_RX, 2*TORSO_RY,
                     facecolor='#e8dcc8', edgecolor='#555', lw=1.5)
    ax.add_patch(torso)
    bl_a, bl_b, _ = bladder_semi_axes(300)
    bl_cy = bladder_center_y(300)
    bladder = Ellipse((0, bl_cy), 2*bl_a, 2*bl_b,
                       facecolor='#4488cc', edgecolor=WONG_BLUE, lw=2, alpha=0.7)
    ax.add_patch(bladder)
    ax.text(0, bl_cy, 'Bladder', ha='center', va='center',
            fontsize=9, fontweight='bold', color='white')
    # Draw all 4 unique positions (one ring)
    for i in range(4):
        ax.plot(ex[i], ey[i], 'o', color='#ddd', ms=12, mec='#999', mew=1, zorder=5)
    ax.set_xlim(-19, 19)
    ax.set_ylim(-14, 14)
    ax.set_xlabel('X (cm)')
    ax.set_ylabel('Y (cm)')


# =====================================================================
# ROW 1: Step 1 — What is a single measurement?
# =====================================================================
ax = fig.add_subplot(gs[0, 0])
draw_torso(ax, 'Step 1: A Single Measurement')

# Best tetrapolar drive pair
dp, dm = drive_pairs[best_di]
dp_r, dm_r = dp % 4, dm % 4
sp, sm = drive_pairs[best_si]
sp_r, sm_r = sp % 4, sm % 4

# Drive electrodes
ax.plot(ex[dp_r], ey[dp_r], 'o', color=WONG_VERM, ms=16, mec='black', mew=2, zorder=10)
ax.plot(ex[dm_r], ey[dm_r], 'o', color=WONG_VERM, ms=16, mec='black', mew=2, zorder=10)
ax.text(ex[dp_r]+1.5, ey[dp_r]+1.5, 'I+', fontsize=11, fontweight='bold', color=WONG_VERM)
ax.text(ex[dm_r]+1.5, ey[dm_r]-1.5, 'I−', fontsize=11, fontweight='bold', color=WONG_VERM)

# Sense electrodes
ax.plot(ex[sp_r], ey[sp_r], 'o', color=WONG_BLUE, ms=16, mec='black', mew=2, zorder=10)
ax.plot(ex[sm_r], ey[sm_r], 'o', color=WONG_BLUE, ms=16, mec='black', mew=2, zorder=10)
ax.text(ex[sp_r]-3, ey[sp_r]+1.5, 'V+', fontsize=11, fontweight='bold', color=WONG_BLUE)
ax.text(ex[sm_r]-3, ey[sm_r]-1.5, 'V−', fontsize=11, fontweight='bold', color=WONG_BLUE)

# Current flow (crude)
ax.annotate('', xy=(ex[dm_r]*0.85, ey[dm_r]*0.85),
            xytext=(ex[dp_r]*0.85, ey[dp_r]*0.85),
            arrowprops=dict(arrowstyle='->', color=WONG_VERM, lw=3,
                           connectionstyle='arc3,rad=0.2'))

ax.text(0, -12.5, f'Sensitivity: {best_tetra*1e3:.2f} mOhm/mL',
        ha='center', fontsize=10, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='#ffe0e0'))


# Step 1 explanation text
ax = fig.add_subplot(gs[0, 1:])
ax.axis('off')
explanation1 = """
STEP 1: What is a single measurement?

You pick 4 electrodes out of 8:
  • 2 for DRIVE — inject 1 mA of AC current (red)
  • 2 for SENSE — measure the voltage difference (blue)

The current flows through the body from I+ to I−. As it passes
through the bladder, the voltage measured at V+/V− changes slightly
depending on how much urine is inside.

This is called a "tetrapolar" measurement.

THE PROBLEM: you can only pick ONE path for the current.
The body is full of conductive tissue (muscle, blood, fat), and
most of the current takes the easy path through muscle — it
doesn't go through the bladder at all.

Best single measurement: 0.22 mOhm per mL of urine.
That means 100 mL of urine changes the voltage by only 0.022 mV.
The respiratory artifact is 100× bigger.
"""
ax.text(0.02, 0.95, explanation1, transform=ax.transAxes,
        fontsize=11, verticalalignment='top', fontfamily='sans-serif',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#fff8f0'))


# =====================================================================
# ROW 2: Step 2 — There are 28 possible drive pairs
# =====================================================================
ax = fig.add_subplot(gs[1, 0])
ax.set_title('Step 2: Each Pair Sees Something Different')

# Bar chart of individual drive pair sensitivities
sorted_idx = np.argsort(drive_sensitivities)[::-1]
colors = []
for idx in sorted_idx:
    dp, dm = drive_pairs[idx]
    # Color by whether it's a "good" pair (cross-ring or AP)
    colors.append(WONG_VERM if drive_sensitivities[idx] > 0.15e-3 else
                  WONG_BLUE if drive_sensitivities[idx] > 0.05e-3 else '#ccc')

ax.barh(range(n_dp), drive_sensitivities[sorted_idx]*1e3,
        color=colors, edgecolor='white', lw=0.3)
ax.set_xlabel('Best sensitivity (mOhm/mL)')
ax.set_ylabel('Drive pair (ranked)')
ax.set_yticks([0, n_dp//4, n_dp//2, 3*n_dp//4, n_dp-1])
ax.set_yticklabels(['#1 (best)', f'#{n_dp//4+1}', f'#{n_dp//2+1}', f'#{3*n_dp//4+1}', f'#{n_dp} (worst)'])
ax.invert_yaxis()
ax.axvline(best_tetra*1e3, color='black', ls='--', lw=1.5)
ax.text(best_tetra*1e3 + 0.005, n_dp-2, f'Best single\n= {best_tetra*1e3:.2f}', fontsize=8)

# Step 2 explanation
ax = fig.add_subplot(gs[1, 1:])
ax.axis('off')
explanation2 = f"""
STEP 2: There are {n_dp} possible drive pairs

With 8 electrodes, you can pick any 2 for drive current.
That's C(8,2) = {n_dp} different pairs.

Each pair sends current through the body in a different direction.
Some pairs send current that happens to pass near the bladder
(high sensitivity — red bars). Others mostly go through
muscle and bone (low sensitivity — grey bars).

The key insight: each drive pair gives you a DIFFERENT "view"
of what's inside the body. It's like looking at an object from
different angles — each angle reveals different information.

With tetrapolar, you pick the SINGLE BEST pair and throw away
the other 27. That wastes a lot of information!

What if we could use ALL 28 pairs and combine them smartly?
That's exactly what SVD does.
"""
ax.text(0.02, 0.95, explanation2, transform=ax.transAxes,
        fontsize=11, verticalalignment='top', fontfamily='sans-serif',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f8ff'))


# =====================================================================
# ROW 3: Step 3 — The measurement matrix and SVD
# =====================================================================
ax = fig.add_subplot(gs[2, 0])
ax.set_title('Step 3: The Measurement Matrix')

# Heatmap of M
vmax = np.max(np.abs(M)) * 1e3
im = ax.imshow(M * 1e3, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
ax.set_xlabel('Sense pair index')
ax.set_ylabel('Drive pair index')
fig.colorbar(im, ax=ax, label='mOhm/mL', shrink=0.8)

ax.text(n_dp/2, -3, f'{n_dp} × {n_dp} = {n_dp*n_dp} measurements',
        ha='center', fontsize=9, fontweight='bold')

# SVD decomposition visualization
ax = fig.add_subplot(gs[2, 1])
ax.set_title('Step 3b: SVD Decomposes It')

# Show M ≈ U * S * V^T schematically
# Draw colored rectangles
from matplotlib.patches import Rectangle

ax.axis('off')
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)

# M
rect_M = Rectangle((0.2, 2.5), 1.8, 1.8, facecolor='#d0e0f0', edgecolor='black', lw=1.5)
ax.add_patch(rect_M)
ax.text(1.1, 3.4, 'M', ha='center', va='center', fontsize=16, fontweight='bold')
ax.text(1.1, 2.2, f'{n_dp}×{n_dp}', ha='center', fontsize=8)

# = sign
ax.text(2.3, 3.4, '=', fontsize=20, ha='center', va='center')

# U
rect_U = Rectangle((2.8, 2.5), 1.2, 1.8, facecolor='#ffe0e0', edgecolor='black', lw=1.5)
ax.add_patch(rect_U)
ax.text(3.4, 3.4, 'U', ha='center', va='center', fontsize=16, fontweight='bold',
        color=WONG_VERM)
ax.text(3.4, 2.2, 'drive\nweights', ha='center', fontsize=7)

# ×
ax.text(4.3, 3.4, '×', fontsize=16, ha='center', va='center')

# S (diagonal)
rect_S = Rectangle((4.7, 2.8), 1.2, 1.2, facecolor='#e0ffe0', edgecolor='black', lw=1.5)
ax.add_patch(rect_S)
ax.text(5.3, 3.4, 'S', ha='center', va='center', fontsize=16, fontweight='bold',
        color=WONG_GREEN)
ax.text(5.3, 2.5, 'importance', ha='center', fontsize=7)

# ×
ax.text(6.2, 3.4, '×', fontsize=16, ha='center', va='center')

# V^T
rect_V = Rectangle((6.6, 2.5), 1.8, 1.2, facecolor='#e0e0ff', edgecolor='black', lw=1.5)
ax.add_patch(rect_V)
ax.text(7.5, 3.1, 'V', ha='center', va='center', fontsize=16, fontweight='bold',
        color=WONG_BLUE)
ax.text(7.5, 2.2, 'sense\nweights', ha='center', fontsize=7)

# Arrow to rank-1 explanation
ax.text(5.0, 5.5, 'SVD says: "Here are the patterns,\nranked by importance"',
        ha='center', fontsize=10, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='#fffde0'))

ax.annotate('', xy=(5.3, 4.2), xytext=(5.0, 5.2),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

# Rank-1 explanation
ax.text(5.0, 1.2, 'Rank-1 = use only the FIRST column of U\n'
        '(best drive weights) and first row of V\n'
        '(best sense weights). S[0] = the sensitivity.',
        ha='center', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='#f0fff0'))

# The actual weights
ax = fig.add_subplot(gs[2, 2])
ax.set_title('Step 3c: The SVD Weights')

# Drive weights (U[:,0])
drive_weights = U[:, 0]
# Sense weights (Vt[0,:])
sense_weights = Vt[0, :]

# Sort by magnitude
sort_d = np.argsort(np.abs(drive_weights))[::-1]

x = np.arange(n_dp)
colors_d = [WONG_VERM if drive_weights[i] > 0 else WONG_BLUE for i in sort_d]
ax.barh(x, drive_weights[sort_d], color=colors_d, alpha=0.7, edgecolor='white', lw=0.3)

# Label top pairs
for rank, idx in enumerate(sort_d[:6]):
    dp, dm = drive_pairs[idx]
    dp_r, dm_r = dp % 4, dm % 4
    names = ['R', 'F', 'L', 'B']
    rings = ['₉', '₁₀']
    label = f'{names[dp_r]}{rings[dp//4]}→{names[dm_r]}{rings[dm//4]}'
    ax.text(0.005 * np.sign(drive_weights[idx]), rank,
            f' {label}: {drive_weights[idx]:+.3f}',
            fontsize=7, va='center',
            color='black')

ax.set_xlabel('Weight')
ax.set_ylabel('Drive pair (ranked by |weight|)')
ax.set_yticks([0, 6, 13, 20, 27])
ax.set_yticklabels(['#1', '#7', '#14', '#21', '#28'])
ax.invert_yaxis()
ax.axvline(0, color='black', lw=0.5)

ax.text(0, n_dp + 2, 'Red = positive weight, Blue = negative\n'
        '(opposite current directions add up!)',
        fontsize=8, ha='center', style='italic')


# =====================================================================
# ROW 4: Step 4 — The result
# =====================================================================
ax = fig.add_subplot(gs[3, 0])
draw_torso(ax, 'Step 4a: Tetrapolar (1 pair)')

# Show best tetrapolar
dp, dm = drive_pairs[best_di]
dp_r, dm_r = dp % 4, dm % 4
ax.plot(ex[dp_r], ey[dp_r], 'o', color=WONG_VERM, ms=16, mec='black', mew=2, zorder=10)
ax.plot(ex[dm_r], ey[dm_r], 'o', color=WONG_VERM, ms=16, mec='black', mew=2, zorder=10)

ax.annotate('', xy=(ex[dm_r]*0.88, ey[dm_r]*0.88),
            xytext=(ex[dp_r]*0.88, ey[dp_r]*0.88),
            arrowprops=dict(arrowstyle='->', color=WONG_VERM, lw=3,
                           connectionstyle='arc3,rad=0.15'))

ax.text(0, -12.5, f'1 measurement\nSensitivity: {best_tetra*1e3:.2f} mOhm/mL',
        ha='center', fontsize=10, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='#ffe0e0'))


ax = fig.add_subplot(gs[3, 1])
draw_torso(ax, 'Step 4b: SVD Rank-1 (all pairs, weighted)')

# Show ALL drive pairs with weights
max_w = np.max(np.abs(drive_weights))
for di in range(n_dp):
    w = drive_weights[di]
    if abs(w) < 0.02 * max_w:
        continue
    dp, dm = drive_pairs[di]
    dp_r, dm_r = dp % 4, dm % 4
    lw = 0.5 + 4 * abs(w) / max_w
    alpha = 0.15 + 0.85 * abs(w) / max_w
    color = WONG_VERM if w > 0 else WONG_BLUE

    rad = 0.15 if dp_r != (dm_r + 2) % 4 else 0
    ax.annotate('', xy=(ex[dm_r]*0.88, ey[dm_r]*0.88),
                xytext=(ex[dp_r]*0.88, ey[dp_r]*0.88),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=lw, alpha=alpha,
                                connectionstyle=f'arc3,rad={rad}'),
                zorder=7)

# Highlight all electrodes as active
for i in range(4):
    ax.plot(ex[i], ey[i], 'o', color=WONG_VERM, ms=14, mec='black', mew=2, zorder=10)

ax.text(0, -12.5, f'{n_dp} weighted measurements, combined\n'
        f'Sensitivity: {S[0]*1e3:.2f} mOhm/mL  ({S[0]/best_tetra:.1f}× better)',
        ha='center', fontsize=10, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='#e0ffe0'))


# Step 4 explanation
ax = fig.add_subplot(gs[3, 2])
ax.axis('off')
explanation4 = f"""
THE RESULT

Tetrapolar (pick best 4 of 8):
  → 1 measurement
  → 0.22 mOhm/mL
  → 44 mL resolution

SVD Rank-1 (use all 8, weighted):
  → 2 sequential measurements
  → 0.93 mOhm/mL
  → 6 mL resolution
  → 4.3× improvement!

WHY IT WORKS:

Think of it like voting.
Each drive pair "votes" on whether the
bladder has more or less urine.

Some pairs have a strong opinion (high
weight) — they send current near the
bladder. Others have a weak opinion.
Some vote the opposite way (negative
weight) — their current goes AWAY
from the bladder, so a DECREASE in
their signal means MORE bladder volume.

SVD figures out the optimal ballot:
how much to trust each voter, and
whether to flip their vote.

The result: all the "bladder information"
from {n_dp} pairs adds up constructively,
while noise cancels out.

In hardware: the AD5940 cycles through
2 programmed current patterns in ~20 ms.
Same electrodes, smarter drive.
"""
ax.text(0.02, 0.98, explanation4, transform=ax.transAxes,
        fontsize=10.5, verticalalignment='top', fontfamily='sans-serif',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0fff0'))


fig.suptitle('How SVD Works: A Step-by-Step Guide',
             fontsize=16, fontweight='bold', y=0.98)

plt.savefig('figures/writeup_fig6_svd_tutorial.png', dpi=300, bbox_inches='tight')
print('\nSaved: figures/writeup_fig6_svd_tutorial.png')
plt.close()
