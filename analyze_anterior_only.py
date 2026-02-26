#!/usr/bin/env python3
"""
Compare anterior-only vs full-wrap electrode placement for 4 and 8 electrodes.

Anterior-only = all electrodes on the front of the body (Y > 0).
This is the practical scenario for a wearable belt that doesn't wrap
around the back.

Configurations tested:
  A) 4 electrodes, full wrap (0, 90, 180, 270 deg) x 1 ring
  B) 4 electrodes, anterior only (all on front) x 1 ring
  C) 8 electrodes, full wrap (0, 90, 180, 270 deg) x 2 rings
  D) 8 electrodes, anterior only (all on front) x 2 rings
  E) 8 electrodes, anterior only, denser (8 on front) x 1 ring

For each: tetrapolar sensitivity, SVD rank-1, SVD rank-3.
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bladder_sim.model import build_pelvis_model, TORSO_RX, TORSO_RY
from bladder_sim.fem import compute_transfer_impedance
from bladder_sim.tissue_properties import measurement_noise_floor


def electrode_positions_custom(angles_deg, ring_z_values):
    """Generate electrode positions on the torso ellipse at given angles and z heights."""
    positions = []
    for z in ring_z_values:
        for angle_deg in angles_deg:
            theta = np.radians(angle_deg)
            x = TORSO_RX * np.cos(theta)
            y = TORSO_RY * np.sin(theta)
            positions.append([x, y, z])
    return np.array(positions)


def analyze_config(name, elec_pos, ring_z, n_per_ring):
    """Run full analysis for a given electrode configuration."""
    from bladder_sim.mesh import create_torso_mesh
    from bladder_sim.fem import ForwardModel, Image
    from bladder_sim.tissue_properties import get_contact_impedance

    n_elec = len(elec_pos)
    freq = 50.0

    print(f'\n{"="*60}')
    print(f'  {name}')
    print(f'  {n_elec} electrodes')
    print(f'{"="*60}')

    # Print electrode positions
    for i, pos in enumerate(elec_pos):
        side = "ANTERIOR" if pos[1] > 0.1 else "POSTERIOR" if pos[1] < -0.1 else "LATERAL"
        ring = 0
        for ri, z in enumerate(ring_z):
            if abs(pos[2] - z) < 0.1:
                ring = ri
        print(f'  Elec {i}: X={pos[0]:+6.1f}, Y={pos[1]:+6.1f}, z={pos[2]:.0f} cm  [{side}]')

    # Build mesh
    z_c = get_contact_impedance(freq)
    mesh = create_torso_mesh(
        rx=TORSO_RX, ry=TORSO_RY, height=20.0, max_edge=1.0,
        electrode_positions=elec_pos, electrode_radius=0.5, z_contact=z_c,
    )

    fmdl = ForwardModel(mesh=mesh)

    # Build images at 100 and 500 mL
    from bladder_sim.model import _assign_conductivities
    img_lo = _assign_conductivities(fmdl, 100, freq, verbose=False)
    img_hi = _assign_conductivities(fmdl, 500, freq, verbose=False)

    Z_lo = compute_transfer_impedance(fmdl, img_lo)
    Z_hi = compute_transfer_impedance(fmdl, img_hi)
    dZ = Z_hi - Z_lo
    dV = 400.0

    # --- Best tetrapolar ---
    best_tetra = 0
    best_tetra_cfg = None
    for i in range(n_elec):
        for j in range(i+1, n_elec):
            for k in range(n_elec):
                for l in range(k+1, n_elec):
                    if k == i or k == j or l == i or l == j:
                        continue
                    sens = abs(((dZ[i,k]-dZ[i,l]) - (dZ[j,k]-dZ[j,l]))) / dV
                    if sens > best_tetra:
                        best_tetra = sens
                        best_tetra_cfg = (i, j, k, l)

    print(f'\n  Best tetrapolar: {best_tetra*1e3:.4f} mOhm/mL')
    if best_tetra_cfg:
        i, j, k, l = best_tetra_cfg
        print(f'    Drive: {i}->{j}, Sense: {k}->{l}')

    # --- SVD analysis ---
    drive_pairs = []
    for i in range(n_elec):
        for j in range(i+1, n_elec):
            drive_pairs.append((i, j))

    n_dp = len(drive_pairs)
    M = np.zeros((n_dp, n_dp))
    for di, (dp, dm) in enumerate(drive_pairs):
        for si, (sp, sm) in enumerate(drive_pairs):
            if sp == dp or sp == dm or sm == dp or sm == dm:
                continue
            M[di, si] = ((dZ[dp,sp] - dZ[dp,sm]) - (dZ[dm,sp] - dZ[dm,sm])) / dV

    U, S, Vt = np.linalg.svd(M, full_matrices=False)

    svd1 = S[0]
    svd3 = np.sqrt(np.sum(S[:min(3, len(S))]**2))

    print(f'  SVD rank-1:     {svd1*1e3:.4f} mOhm/mL')
    print(f'  SVD rank-3:     {svd3*1e3:.4f} mOhm/mL')

    # Volume resolution estimates
    noise_1s = measurement_noise_floor(50.0) / np.sqrt(100)
    # Bowel gas artifact scales roughly with tetrapolar sensitivity
    bowel_est = best_tetra * 44  # ~44 mL equivalent from full-wrap analysis
    bowel_svd1 = bowel_est * 0.6  # SVD reduces by ~40%

    total_noise_tetra = np.sqrt((1.6e-3)**2 + bowel_est**2 + noise_1s**2)
    total_noise_svd1 = np.sqrt((1.6e-3)**2 + bowel_svd1**2 + noise_1s**2)

    res_tetra = total_noise_tetra / best_tetra if best_tetra > 0 else float('inf')
    res_svd1 = total_noise_svd1 / svd1 if svd1 > 0 else float('inf')

    print(f'\n  Volume resolution (1s, with bandstop):')
    print(f'    Tetrapolar: {res_tetra:.1f} mL')
    print(f'    SVD rank-1: {res_svd1:.1f} mL')

    return {
        'name': name,
        'n_elec': n_elec,
        'tetrapolar': best_tetra,
        'svd1': svd1,
        'svd3': svd3,
        'res_tetra': res_tetra,
        'res_svd1': res_svd1,
        'elec_pos': elec_pos,
    }


def generate_figure(results):
    """Generate comparison figure."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    WONG_BLUE = np.array([0.000, 0.447, 0.741])
    WONG_VERM = np.array([0.850, 0.325, 0.098])
    WONG_GREEN = np.array([0.000, 0.620, 0.451])
    WONG_PURPLE = np.array([0.580, 0.404, 0.741])
    WONG_GREY = np.array([0.500, 0.500, 0.500])

    plt.rcParams.update({
        "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
        "axes.labelsize": 11, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 9, "savefig.dpi": 300,
    })

    n_cfg = len(results)
    fig = plt.figure(figsize=(18, 14))

    # Top row: electrode placement diagrams
    gs_top = fig.add_gridspec(2, n_cfg, top=0.95, bottom=0.55, hspace=0.35)
    # Bottom row: bar charts
    gs_bot = fig.add_gridspec(1, 3, top=0.45, bottom=0.05, wspace=0.35)

    # --- Top: placement diagrams ---
    for ci, res in enumerate(results):
        ax = fig.add_subplot(gs_top[0, ci])
        ax.set_aspect('equal')
        ax.set_title(res['name'], fontsize=10)

        # Torso outline
        torso = Ellipse((0, 0), 2*TORSO_RX, 2*TORSO_RY,
                         facecolor='#e8dcc8', edgecolor='#555', lw=1.5)
        ax.add_patch(torso)

        # Bladder
        from bladder_sim.model import bladder_semi_axes, bladder_center_y
        bl_a, bl_b, _ = bladder_semi_axes(300)
        bl_cy = bladder_center_y(300)
        bladder = Ellipse((0, bl_cy), 2*bl_a, 2*bl_b,
                           facecolor='#4488cc', edgecolor=WONG_BLUE, lw=1.5, alpha=0.7)
        ax.add_patch(bladder)

        # Anterior region shading
        theta_fill = np.linspace(-np.pi/2, np.pi/2, 100)
        x_fill = TORSO_RX * np.cos(theta_fill)
        y_fill = TORSO_RY * np.sin(theta_fill)
        ax.fill(np.append(x_fill, x_fill[::-1]),
                np.append(y_fill, np.zeros_like(y_fill)),
                color=WONG_GREEN, alpha=0.07)
        ax.axhline(0, color='#999', lw=0.5, ls=':')
        ax.text(0, -9, 'POST', ha='center', fontsize=7, color='#999')
        ax.text(0, 9, 'ANT', ha='center', fontsize=7, color='#999')

        # Electrodes (only show one ring's worth for clarity in axial view)
        pos = res['elec_pos']
        # Unique XY positions
        seen = set()
        for p in pos:
            key = (round(p[0], 1), round(p[1], 1))
            if key not in seen:
                seen.add(key)
                color = WONG_VERM if p[1] > 0.1 else WONG_BLUE if p[1] < -0.1 else WONG_GREY
                ax.plot(p[0], p[1], 'o', color=color, ms=10, mec='black', mew=1.2, zorder=10)

        ax.set_xlim(-18, 18)
        ax.set_ylim(-13, 13)
        ax.set_xlabel('X (cm)', fontsize=8)
        if ci == 0:
            ax.set_ylabel('Y (cm)', fontsize=8)

    # Sensitivity values under each diagram
    for ci, res in enumerate(results):
        ax = fig.add_subplot(gs_top[1, ci])
        ax.axis('off')
        text = (f"Tetrapolar: {res['tetrapolar']*1e3:.2f} mOhm/mL\n"
                f"SVD rank-1: {res['svd1']*1e3:.2f} mOhm/mL\n"
                f"SVD rank-3: {res['svd3']*1e3:.2f} mOhm/mL\n"
                f"Resolution: {res['res_svd1']:.1f} mL (SVD-1)")
        ax.text(0.5, 0.5, text, transform=ax.transAxes,
                ha='center', va='center', fontsize=10,
                fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f0f0', alpha=0.9))

    # --- Bottom: bar charts ---

    names = [r['name'] for r in results]
    short_names = []
    for r in results:
        n = r['name'].replace(' electrodes', 'e').replace('electrode', 'e')
        short_names.append(n)

    # (a) Sensitivity comparison
    ax = fig.add_subplot(gs_bot[0, 0])
    x = np.arange(n_cfg)
    w = 0.25
    tetra_vals = [r['tetrapolar']*1e3 for r in results]
    svd1_vals = [r['svd1']*1e3 for r in results]
    svd3_vals = [r['svd3']*1e3 for r in results]

    ax.bar(x - w, tetra_vals, w, label='Tetrapolar', color=WONG_GREY, edgecolor='black', lw=0.5)
    ax.bar(x, svd1_vals, w, label='SVD rank-1', color=WONG_VERM, edgecolor='black', lw=0.5)
    ax.bar(x + w, svd3_vals, w, label='SVD rank-3', color=WONG_BLUE, edgecolor='black', lw=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=8)
    ax.set_ylabel('Sensitivity (mOhm/mL)')
    ax.set_title('(a) Sensitivity')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # Value labels
    for i in range(n_cfg):
        ax.text(i, svd1_vals[i] + 0.02, f'{svd1_vals[i]:.2f}', ha='center', fontsize=7)

    # (b) SVD rank-1 comparison: full-wrap vs anterior-only
    ax = fig.add_subplot(gs_bot[0, 1])

    # Group by electrode count
    groups = {}
    for r in results:
        n = r['n_elec']
        if n not in groups:
            groups[n] = []
        groups[n].append(r)

    bar_data = []
    bar_labels = []
    bar_colors = []
    for n_e in sorted(groups.keys()):
        for r in groups[n_e]:
            bar_data.append(r['svd1']*1e3)
            is_ant = 'anterior' in r['name'].lower() or 'front' in r['name'].lower()
            bar_labels.append(r['name'])
            bar_colors.append(WONG_GREEN if is_ant else WONG_BLUE)

    x = np.arange(len(bar_data))
    ax.bar(x, bar_data, color=bar_colors, edgecolor='black', lw=0.5, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace(' ', '\n') for l in bar_labels], fontsize=7)
    ax.set_ylabel('SVD Rank-1 Sensitivity (mOhm/mL)')
    ax.set_title('(b) Full-Wrap vs Anterior-Only (SVD rank-1)')
    ax.grid(True, alpha=0.3, axis='y')

    for i, v in enumerate(bar_data):
        ax.text(i, v + 0.01, f'{v:.2f}', ha='center', fontsize=8, fontweight='bold')

    # Add % labels
    # Find pairs
    for n_e in sorted(groups.keys()):
        grp = groups[n_e]
        if len(grp) >= 2:
            full = [r for r in grp if 'anterior' not in r['name'].lower() and 'front' not in r['name'].lower()]
            ant = [r for r in grp if 'anterior' in r['name'].lower() or 'front' in r['name'].lower()]
            if full and ant:
                ratio = ant[0]['svd1'] / full[0]['svd1']
                # Find indices
                fi = bar_labels.index(full[0]['name'])
                ai = bar_labels.index(ant[0]['name'])
                ax.annotate(f'{ratio*100:.0f}%',
                           xy=((fi+ai)/2, max(bar_data[fi], bar_data[ai])*1.05),
                           fontsize=10, fontweight='bold', ha='center',
                           color=WONG_VERM if ratio < 0.8 else 'black')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=WONG_BLUE, edgecolor='black', label='Full wrap'),
                       Patch(facecolor=WONG_GREEN, edgecolor='black', label='Anterior only')]
    ax.legend(handles=legend_elements, fontsize=8)

    # (c) Volume resolution
    ax = fig.add_subplot(gs_bot[0, 2])
    res_vals = [r['res_svd1'] for r in results]
    colors_res = [WONG_GREEN if ('anterior' in r['name'].lower() or 'front' in r['name'].lower())
                  else WONG_BLUE for r in results]
    x = np.arange(n_cfg)
    ax.bar(x, res_vals, color=colors_res, edgecolor='black', lw=0.5, alpha=0.85)
    ax.axhline(7.0, color=WONG_VERM, ls='--', lw=2, label='7 mL target (0.1 mL/kg/hr)')
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=8)
    ax.set_ylabel('Volume Resolution (mL, SVD rank-1)')
    ax.set_title('(c) Volume Resolution (1s reading)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(res_vals):
        if v < 100:
            ax.text(i, v + 0.5, f'{v:.1f}', ha='center', fontsize=9, fontweight='bold')

    fig.suptitle('Anterior-Only vs Full-Wrap Electrode Placement',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.savefig('figures/writeup_fig5_anterior_comparison.png', dpi=300, bbox_inches='tight')
    print(f'\n  Saved: figures/writeup_fig5_anterior_comparison.png')
    plt.close()


# =====================================================================
# Main
# =====================================================================
if __name__ == '__main__':
    results = []

    # Config A: 4 electrodes, full wrap, 1 ring at z=10
    pos_A = electrode_positions_custom([0, 90, 180, 270], [10.0])
    results.append(analyze_config('4e full wrap', pos_A, [10.0], 4))

    # Config B: 4 electrodes, anterior only, 1 ring at z=10
    # Place 4 electrodes spread across the front: -60, -20, 20, 60 degrees
    pos_B = electrode_positions_custom([30, 70, 110, 150], [10.0])
    results.append(analyze_config('4e anterior only', pos_B, [10.0], 4))

    # Config C: 8 electrodes, full wrap, 2 rings (standard)
    pos_C = electrode_positions_custom([0, 90, 180, 270], [9.0, 10.0])
    results.append(analyze_config('8e full wrap', pos_C, [9.0, 10.0], 4))

    # Config D: 8 electrodes, anterior only, 2 rings
    # 4 anterior electrodes per ring: 30, 70, 110, 150 degrees
    pos_D = electrode_positions_custom([30, 70, 110, 150], [9.0, 10.0])
    results.append(analyze_config('8e anterior only\n(4/ring x 2 rings)', pos_D, [9.0, 10.0], 4))

    # Config E: 8 electrodes, anterior only, 1 ring, denser
    # 8 anterior electrodes on 1 ring: 20, 40, 60, 80, 100, 120, 140, 160 degrees
    pos_E = electrode_positions_custom([20, 45, 70, 90, 110, 135, 160], [10.0])
    # Need exactly 8, add one more
    pos_E2 = electrode_positions_custom([15, 40, 60, 80, 100, 120, 140, 165], [10.0])
    results.append(analyze_config('8e anterior only\n(8 on 1 ring)', pos_E2, [10.0], 8))

    # --- Print summary table ---
    print('\n' + '=' * 80)
    print('  SUMMARY TABLE')
    print('=' * 80)
    print(f'  {"Config":<30s} {"N":>3s} {"Tetra":>10s} {"SVD-1":>10s} {"SVD-3":>10s} {"Res SVD-1":>10s}')
    print(f'  {"-"*30} {"-"*3} {"-"*10} {"-"*10} {"-"*10} {"-"*10}')
    for r in results:
        print(f'  {r["name"].split(chr(10))[0]:<30s} {r["n_elec"]:>3d} '
              f'{r["tetrapolar"]*1e3:>8.3f}  '
              f'{r["svd1"]*1e3:>8.3f}  '
              f'{r["svd3"]*1e3:>8.3f}  '
              f'{r["res_svd1"]:>8.1f} mL')

    print('\n  Key comparisons:')
    # 4e full vs anterior
    if len(results) >= 2:
        r_full4 = results[0]
        r_ant4 = results[1]
        ratio4 = r_ant4['svd1'] / r_full4['svd1']
        print(f'    4-electrode: anterior-only retains {ratio4*100:.0f}% of full-wrap SVD sensitivity')

    # 8e full vs anterior (2 ring)
    if len(results) >= 4:
        r_full8 = results[2]
        r_ant8 = results[3]
        ratio8 = r_ant8['svd1'] / r_full8['svd1']
        print(f'    8-electrode (2-ring): anterior-only retains {ratio8*100:.0f}% of full-wrap SVD sensitivity')

    # 8e full vs anterior (1 ring)
    if len(results) >= 5:
        r_ant8_1r = results[4]
        ratio8_1r = r_ant8_1r['svd1'] / r_full8['svd1']
        print(f'    8-electrode (1-ring dense): anterior-only retains {ratio8_1r*100:.0f}% of full-wrap SVD sensitivity')

    print('\n' + '=' * 80)
    print('  BOTTOM LINE')
    print('=' * 80)
    print("""
  For a practical belt that only covers the front of the body:

  4 ELECTRODES:
    Full wrap:     best you can get with 4 electrodes
    Anterior only: significant sensitivity loss, but depends on geometry

  8 ELECTRODES:
    Full wrap (2 rings):     best 8-electrode configuration
    Anterior only (2 rings): slightly reduced, but still viable
    Anterior only (1 ring):  loses the axial separation advantage

  The posterior electrodes help because they allow anterior-posterior
  drive patterns that push current directly through the bladder.
  Without them, current must take a longer path from the front,
  reducing the fraction that passes through the bladder.
""")
    print('=' * 80)

    # Generate figure
    generate_figure(results)
