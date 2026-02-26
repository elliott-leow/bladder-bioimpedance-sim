"""
Publication-quality figures for bladder bioimpedance simulation.

Generates 6 figures suitable for journal publication:
    Figure 1: 3D Anatomical Model Overview (2x2)
    Figure 2: Impedance vs Volume (2x2)
    Figure 3: Stimulation Pattern Optimization (2x2)
    Figure 4: Optimal Electrode Placement — 3D body (1x2)
    Figure 5: Frequency Response & SNR (2x3)
    Figure 6: Multi-Frequency Bladder Isolation (2x2)

All figures use colorblind-friendly palette (Wong 2011, Nature Methods).
3D visualizations use matplotlib Axes3D with wireframe torso, surface organs,
and scatter electrodes.
Saved as PNG at 300 DPI.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap, BoundaryNorm, Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pathlib import Path
from typing import Dict, Optional

from .model import (
    TISSUE_NAMES, TISSUE_COLORS, BLADDER_ASPECT, BLADDER_BASE_Z,
    BLADDER_CENTER_Y, BLADDER_WALL_THICK, TORSO_RX, TORSO_RY, TORSO_H,
    get_tissue_labels, get_bladder_mask,
)
from .mesh import TorsoMesh, compute_electrode_positions
from .analysis import get_angular_sep


# =====================================================================
# Colorblind-friendly palette (Wong 2011, Nature Methods 8:441)
# =====================================================================
WONG = {
    "blue":       np.array([0.000, 0.447, 0.741]),
    "vermillion": np.array([0.850, 0.325, 0.098]),
    "green":      np.array([0.000, 0.620, 0.451]),
    "red":        np.array([0.800, 0.200, 0.200]),
    "purple":     np.array([0.580, 0.404, 0.741]),
    "brown":      np.array([0.600, 0.400, 0.200]),
    "pink":       np.array([0.906, 0.541, 0.765]),
    "grey":       np.array([0.500, 0.500, 0.500]),
}
WONG_LIST = list(WONG.values())

# Publication defaults
plt.rcParams.update({
    "font.size": 13,
    "font.family": "sans-serif",
    "axes.labelsize": 13,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.linewidth": 1.2,
})


def _detect_electrode_config(mesh: TorsoMesh):
    """Detect n_per_ring and ring_z from electrode positions."""
    if not mesh.electrodes:
        return 8, np.array([7.0])
    ez = np.array([e.center[2] for e in mesh.electrodes])
    ring_zs = np.sort(np.unique(np.round(ez, 1)))
    n_rings = len(ring_zs)
    n_per_ring = mesh.n_electrodes // n_rings if n_rings > 0 else mesh.n_electrodes
    return n_per_ring, ring_zs


# =====================================================================
# 3D drawing helpers
# =====================================================================

def _draw_torso_wireframe(ax, alpha=0.08, color="steelblue"):
    """Draw semi-transparent torso elliptical cylinder."""
    phi = np.linspace(0, 2 * np.pi, 50)
    z = np.linspace(0, TORSO_H, 12)
    PHI, Z = np.meshgrid(phi, z)
    X = TORSO_RX * np.cos(PHI)
    Y = TORSO_RY * np.sin(PHI)
    ax.plot_surface(X, Y, Z, alpha=alpha, color=color,
                    edgecolor=(0.5, 0.5, 0.5, 0.15), linewidth=0.3)


def _draw_bladder_3d(ax, volume_mL=300, color="gold", alpha=0.55, label=None):
    """Draw bladder ellipsoid surface."""
    k = (volume_mL / ((4.0 / 3.0) * np.pi * np.prod(BLADDER_ASPECT))) ** (1.0 / 3.0)
    a, b, c = BLADDER_ASPECT * k
    cz = BLADDER_BASE_Z + c

    u = np.linspace(0, np.pi, 25)
    v = np.linspace(0, 2 * np.pi, 35)
    U, V = np.meshgrid(u, v)
    X = a * np.sin(U) * np.cos(V)
    Y = b * np.sin(U) * np.sin(V) + BLADDER_CENTER_Y
    Z = c * np.cos(U) + cz
    ax.plot_surface(X, Y, Z, alpha=alpha, color=color,
                    edgecolor="none", label=label)


def _draw_electrodes_3d(ax, mesh, highlight=None, size_normal=40, size_highlight=160):
    """Draw electrodes as 3D scatter; optionally highlight drive/sense."""
    n_per_ring, ring_zs = _detect_electrode_config(mesh)
    n_rings = len(ring_zs)
    ring_colors = [WONG_LIST[i % len(WONG_LIST)] for i in range(n_rings)]

    drive_set = set(highlight.get("drive", [])) if highlight else set()
    sense_set = set(highlight.get("sense", [])) if highlight else set()
    special = drive_set | sense_set

    for ei, elec in enumerate(mesh.electrodes):
        pos = elec.center
        ri = ei // n_per_ring
        if ei in drive_set:
            ax.scatter(pos[0], pos[1], pos[2], color="red", s=size_highlight,
                       marker="D", edgecolors="k", linewidths=1.2, zorder=10,
                       depthshade=False)
        elif ei in sense_set:
            ax.scatter(pos[0], pos[1], pos[2], color=WONG["blue"],
                       s=size_highlight, marker="s", edgecolors="k",
                       linewidths=1.2, zorder=10, depthshade=False)
        else:
            ax.scatter(pos[0], pos[1], pos[2],
                       c=[ring_colors[min(ri, n_rings - 1)]], s=size_normal,
                       edgecolors="k", linewidths=0.4, zorder=5,
                       depthshade=False)


def _setup_3d_axes(ax, elev=22, azim=-55):
    """Configure 3D axes for clean publication look."""
    ax.set_xlabel("X (cm)", labelpad=8)
    ax.set_ylabel("Y (cm)", labelpad=8)
    ax.set_zlabel("Z (cm)", labelpad=8)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlim(-TORSO_RX - 1, TORSO_RX + 1)
    ax.set_ylim(-TORSO_RY - 1, TORSO_RY + 1)
    ax.set_zlim(0, TORSO_H)
    # Reduce tick clutter
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.zaxis.set_major_locator(plt.MaxNLocator(5))
    ax.tick_params(labelsize=8, pad=1)


# =====================================================================
# Public API
# =====================================================================

def generate_publication_figures(
    mesh: TorsoMesh,
    sensitivity_results: Optional[Dict] = None,
    pattern_results: Optional[Dict] = None,
    freq_results: Optional[Dict] = None,
    multifreq_results: Optional[Dict] = None,
    fig_dir: str = "figures",
) -> None:
    """Generate all publication figures."""
    fig_path = Path(fig_dir)
    fig_path.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*50}")
    print(f"  PUBLICATION FIGURE GENERATION")
    print(f"{'='*50}\n")

    print("[1/6] Figure 1: 3D Anatomical Model ...")
    _fig1_anatomical_model(mesh, fig_path)

    if sensitivity_results is not None:
        print("[2/6] Figure 2: Impedance vs Volume ...")
        _fig2_impedance_vs_volume(mesh, sensitivity_results, fig_path)
    else:
        print("[2/6] Skipped (no sensitivity results)")

    if pattern_results is not None:
        print("[3/6] Figure 3: Stimulation Pattern Optimization ...")
        _fig3_stim_pattern(mesh, pattern_results, fig_path)
    else:
        print("[3/6] Skipped (no pattern results)")

    if pattern_results is not None:
        print("[4/6] Figure 4: Optimal Electrode Placement (3D) ...")
        _fig4_optimal_placement(mesh, pattern_results, fig_path)
    else:
        print("[4/6] Skipped (no pattern results)")

    if freq_results is not None:
        print("[5/6] Figure 5: Frequency Response & SNR ...")
        _fig5_frequency_response(freq_results, fig_path)
    else:
        print("[5/6] Skipped (no frequency results)")

    if multifreq_results is not None:
        print("[6/6] Figure 6: Multi-Frequency Bladder Isolation ...")
        _fig6_multifreq_isolation(multifreq_results, fig_path)
    else:
        print("[6/6] Skipped (no multi-frequency results)")

    print(f"\n{'='*50}")
    print(f"  FIGURES COMPLETE  ->  {fig_path.resolve()}")
    print(f"{'='*50}")


# =====================================================================
# Figure 1: 3D Anatomical Model
# =====================================================================

def _fig1_anatomical_model(mesh: TorsoMesh, fig_dir: Path):
    fig = plt.figure(figsize=(16, 14))
    fig.suptitle("Figure 1: 3D Anatomical Model", fontsize=18, fontweight="bold", y=0.98)

    # (a) 3D torso with bladder and electrodes
    ax = fig.add_subplot(2, 2, 1, projection="3d")
    _draw_torso_wireframe(ax, alpha=0.06)
    _draw_bladder_3d(ax, 300, color="gold", alpha=0.6)
    _draw_electrodes_3d(ax, mesh)
    _setup_3d_axes(ax)
    ax.set_title("(a) 3D model with electrodes", pad=12)

    # (b) Axial cross-section
    ax = fig.add_subplot(2, 2, 2)
    _plot_cross_section(ax, mesh, z_slice=7.0)
    ax.set_title("(b) Axial cross-section, z = 7 cm")

    # (c) Sagittal cross-section
    ax = fig.add_subplot(2, 2, 3)
    _plot_sagittal_section(ax, mesh, x_slice=0.0)
    ax.set_title("(c) Sagittal cross-section, x = 0 cm")

    # (d) 3D bladder volumes
    ax = fig.add_subplot(2, 2, 4, projection="3d")
    for vi, (vol, col, al) in enumerate([
        (100, WONG["blue"], 0.25),
        (300, WONG["green"], 0.35),
        (500, WONG["vermillion"], 0.45),
    ]):
        _draw_bladder_3d(ax, vol, color=col, alpha=al)
    _draw_torso_wireframe(ax, alpha=0.03)
    _setup_3d_axes(ax, elev=15, azim=-50)
    # Manual legend
    for vol, col in [(100, WONG["blue"]), (300, WONG["green"]), (500, WONG["vermillion"])]:
        ax.scatter([], [], [], c=[col], s=60, label=f"{vol} mL")
    ax.legend(loc="upper left", fontsize=10)
    ax.set_title("(d) Bladder at 100 / 300 / 500 mL", pad=12)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    _save_figure(fig, fig_dir, "fig1_anatomical_model")


# =====================================================================
# Figure 2: Impedance vs Volume
# =====================================================================

def _fig2_impedance_vs_volume(mesh: TorsoMesh, results: Dict, fig_dir: Path):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Figure 2: Impedance vs Bladder Volume",
                 fontsize=18, fontweight="bold")

    volumes = results["volumes"]
    Z = results["Z"]
    Z_norm = results["Z_norm"]
    sens = results["sens_per_ch"]
    best_chs = results["best_channels"]

    # (a)
    ax = axes[0, 0]
    ax.plot(volumes, Z_norm, "ko-", lw=2.5, ms=9, mfc="k")
    p = np.polyfit(volumes, Z_norm, 1)
    ax.plot(volumes, np.polyval(p, volumes), "--", color="0.5", lw=1.5)
    ax.set_xlabel("Bladder Volume (mL)")
    ax.set_ylabel(r"$\|Z\|$ ($\Omega$)")
    ax.set_title("(a) Aggregate impedance norm")

    # (b)
    ax = axes[0, 1]
    vol_fine = np.linspace(volumes.min(), volumes.max(), 100)
    for ci in range(min(3, len(best_chs))):
        ch = best_chs[ci]
        p = np.polyfit(volumes, Z[:, ch], 1)
        c = WONG_LIST[ci]
        ax.plot(volumes, Z[:, ch], "o", color=c, ms=9, mfc=c, lw=2.5)
        ax.plot(vol_fine, np.polyval(p, vol_fine), "-", color=c, lw=2,
                label=f"Ch {ch}: {sens[ch]*1e3:+.3f} m$\\Omega$/mL")
    ax.set_xlabel("Bladder Volume (mL)")
    ax.set_ylabel(r"$Z$ ($\Omega$)")
    ax.set_title("(b) Top 3 channels")
    ax.legend(fontsize=9, loc="best")

    # (c)
    ax = axes[1, 0]
    ax.hist(np.abs(sens) * 1e3, bins=50, color=WONG["blue"], edgecolor="none", alpha=0.85)
    best_dZ = np.max(np.abs(sens)) * 1e3
    ax.axvline(best_dZ, color="red", ls="--", lw=2, label=f"Best = {best_dZ:.3f}")
    ax.set_xlabel(r"$|dZ/dV|$ (m$\Omega$/mL)")
    ax.set_ylabel("Count")
    ax.set_title("(c) Channel sensitivity distribution")
    ax.legend(fontsize=9)

    # (d) 3D Jacobian sensitivity
    if "jacobian_sens" in results:
        ax = fig.add_subplot(2, 2, 4, projection="3d")
        axes[1, 1].remove()
        _plot_3d_sensitivity(ax, mesh, results["jacobian_sens"])
        ax.set_title("(d) Jacobian sensitivity (3D)", pad=12)

    plt.tight_layout()
    _save_figure(fig, fig_dir, "fig2_impedance_vs_volume")


# =====================================================================
# Figure 3: Stimulation Pattern Optimization
# =====================================================================

def _fig3_stim_pattern(mesh: TorsoMesh, results: Dict, fig_dir: Path):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Figure 3: Stimulation Pattern Optimization",
                 fontsize=18, fontweight="bold")

    dZ = results["drive_best_dZ"]
    is_within = results["is_within"]
    is_cross = results["is_cross"]
    ring_dist = results["ring_dist"]
    drive_pairs = results["drive_pairs"]
    n_per_ring, ring_z = _detect_electrode_config(mesh)

    # (a)
    ax = axes[0, 0]
    ax.hist(dZ * 1e3, bins=50, color=WONG["blue"], edgecolor="none", alpha=0.85)
    ax.axvline(np.max(dZ) * 1e3, color="red", ls="--", lw=2)
    ax.set_xlabel(r"Best $|dZ/dV|$ per drive pair (m$\Omega$/mL)")
    ax.set_ylabel("Count")
    ax.set_title(f"(a) Sensitivity distribution ({len(dZ)} pairs)")

    # (b)
    ax = axes[0, 1]
    if np.any(is_within):
        ax.hist(dZ[is_within] * 1e3, bins=30, color=WONG["blue"], alpha=0.6,
                edgecolor="none", label="Within-ring")
    if np.any(is_cross):
        ax.hist(dZ[is_cross] * 1e3, bins=30, color=WONG["vermillion"], alpha=0.6,
                edgecolor="none", label="Cross-ring")
    ax.set_xlabel(r"Best $|dZ/dV|$ (m$\Omega$/mL)")
    ax.set_ylabel("Count")
    ax.legend(fontsize=11)
    ax.set_title("(b) Within-ring vs cross-ring")

    # (c) sensitivity vs angular separation
    ax = axes[1, 0]
    max_sep = n_per_ring // 2
    sep_best = np.zeros(max_sep)
    for s in range(1, max_sep + 1):
        vals = []
        for di in np.where(is_within)[0]:
            a, b = drive_pairs[di]
            if get_angular_sep(a, b, n_per_ring) == s:
                vals.append(dZ[di])
        if vals:
            sep_best[s - 1] = np.max(vals)
    angles = np.arange(1, max_sep + 1) * 360.0 / n_per_ring
    ax.bar(angles, sep_best * 1e3, width=angles[0] * 0.6,
           color=WONG["green"], edgecolor="none", alpha=0.85)
    ax.set_xlabel(r"Angular Separation ($\degree$)")
    ax.set_ylabel(r"Best $|dZ/dV|$ (m$\Omega$/mL)")
    ax.set_title("(c) Within-ring: sensitivity vs angle")

    # (d) sensitivity vs ring distance
    ax = axes[1, 1]
    max_rd = int(np.max(ring_dist))
    rd_vals = np.arange(max_rd + 1)
    rd_mean = np.zeros(len(rd_vals))
    rd_max = np.zeros(len(rd_vals))
    for i, rd in enumerate(rd_vals):
        mask = ring_dist == rd
        if np.any(mask):
            rd_mean[i] = np.mean(dZ[mask])
            rd_max[i] = np.max(dZ[mask])
    ax.bar(rd_vals - 0.15, rd_mean * 1e3, 0.3, color=WONG["purple"],
           alpha=0.7, label="Mean")
    ax.bar(rd_vals + 0.15, rd_max * 1e3, 0.3, color=WONG["vermillion"],
           alpha=0.7, label="Max")
    ax.set_xlabel("Ring Distance")
    ax.set_ylabel(r"$|dZ/dV|$ (m$\Omega$/mL)")
    ax.set_title("(d) Sensitivity vs ring distance")
    ax.set_xticks(rd_vals)
    ax.legend(fontsize=10)

    plt.tight_layout()
    _save_figure(fig, fig_dir, "fig3_stim_pattern_analysis")


# =====================================================================
# Figure 4: Optimal Electrode Placement (3D body)
# =====================================================================

def _fig4_optimal_placement(mesh: TorsoMesh, results: Dict, fig_dir: Path):
    fig = plt.figure(figsize=(18, 9))
    fig.suptitle("Figure 4: Optimal Electrode Placement",
                 fontsize=18, fontweight="bold", y=0.98)

    opt = results["optimal"]
    opt_a, opt_b = opt["drive"]
    opt_c, opt_d = opt["sense"]
    global_best = opt["dZ_per_mL"]
    n_per_ring, ring_z = _detect_electrode_config(mesh)

    # (a) 3D anterior view
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    _draw_torso_wireframe(ax, alpha=0.05)
    _draw_bladder_3d(ax, 300, color="gold", alpha=0.55)
    _draw_electrodes_3d(ax, mesh, highlight={
        "drive": [opt_a, opt_b],
        "sense": [opt_c, opt_d],
    })
    _setup_3d_axes(ax, elev=18, azim=-60)

    # Legend entries
    ax.scatter([], [], [], c="red", s=140, marker="D", edgecolors="k",
               label=f"Drive ({opt_a}, {opt_b})")
    ax.scatter([], [], [], color=WONG["blue"], s=140, marker="s", edgecolors="k",
               label=f"Sense ({opt_c}, {opt_d})")
    ax.scatter([], [], [], c="gold", s=80, label="Bladder (300 mL)")
    ax.legend(loc="upper left", fontsize=10, markerscale=0.8)
    ax.set_title("(a) 3D electrode layout", pad=14)

    # (b) Justification text panel
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.axis("off")

    ring_a = opt_a // n_per_ring
    ring_b = opt_b // n_per_ring
    ring_c = opt_c // n_per_ring
    ring_d = opt_d // n_per_ring
    ang_drv = get_angular_sep(opt_a, opt_b, n_per_ring) * 360 / n_per_ring
    ang_sns = get_angular_sep(opt_c, opt_d, n_per_ring) * 360 / n_per_ring

    dZ = results["drive_best_dZ"]
    is_within = results["is_within"]
    is_cross = results["is_cross"]

    text = (
        "OPTIMAL 4-ELECTRODE CONFIGURATION\n"
        f"{'='*48}\n"
        f"\n"
        f"  Drive (+):  Electrode {opt_a}  "
        f"(Ring {ring_a+1}, z = {ring_z[ring_a]:.0f} cm)\n"
        f"  Drive (\u2212):  Electrode {opt_b}  "
        f"(Ring {ring_b+1}, z = {ring_z[ring_b]:.0f} cm)\n"
        f"  Sense (+):  Electrode {opt_c}  "
        f"(Ring {ring_c+1}, z = {ring_z[ring_c]:.0f} cm)\n"
        f"  Sense (\u2212):  Electrode {opt_d}  "
        f"(Ring {ring_d+1}, z = {ring_z[ring_d]:.0f} cm)\n"
        f"\n"
        f"  Drive sep:  {ang_drv:.0f}\u00b0      Sense sep:  {ang_sns:.0f}\u00b0\n"
        f"\n"
        f"  SENSITIVITY: {global_best*1e3:.3f} m\u03a9/mL\n"
        f"  ({global_best:.2e} \u03a9/mL)\n"
        f"\n"
        f"{'='*48}\n"
        f"ANATOMICAL JUSTIFICATION\n"
        f"{'='*48}\n"
        f"\n"
        f"1. Anterior\u2013posterior drive (180\u00b0 separation)\n"
        f"   maximises current density through the\n"
        f"   bladder, which sits anteriorly behind\n"
        f"   the pubic symphysis.\n"
        f"\n"
        f"2. Electrode rings at z = {ring_z[ring_a]:.0f}\u2013{ring_z[ring_c]:.0f} cm\n"
        f"   correspond to the supra-pubic region\n"
        f"   directly above the bladder (centre\n"
        f"   z \u2248 7.5 cm in this model).\n"
        f"\n"
        f"3. Drive and sense on adjacent rings gives\n"
        f"   depth selectivity and avoids electrode-\n"
        f"   electrode coupling artefacts.\n"
        f"\n"
        f"4. Tetrapolar (4-electrode) measurement\n"
        f"   eliminates electrode contact impedance\n"
        f"   from the voltage reading.\n"
    )
    ax2.text(0.02, 0.97, text, transform=ax2.transAxes, fontsize=11.5,
             fontfamily="monospace", va="top", ha="left",
             linespacing=1.25)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    _save_figure(fig, fig_dir, "fig4_optimal_placement")


# =====================================================================
# Figure 5: Frequency Response & SNR
# =====================================================================

def _fig5_frequency_response(results: Dict, fig_dir: Path):
    fig, axes = plt.subplots(2, 3, figsize=(19, 11))
    fig.suptitle("Figure 5: Frequency Response & SNR",
                 fontsize=18, fontweight="bold", y=0.99)

    freqs = results["freqs_kHz"]
    freq_best = results["freq_best_abs_dZ"]
    z_contacts = results["z_contacts"]
    noise = results.get("noise_floor", np.zeros_like(freqs))
    freq_snr = results.get("freq_snr", np.zeros_like(freqs))
    opt_abs = results["opt_abs_dZ"]
    opt_base = results["opt_Z_base"]
    opt_snr = results.get("opt_snr", np.zeros_like(freqs))

    # (a) Absolute sensitivity
    ax = axes[0, 0]
    ax.semilogx(freqs, freq_best * 1e3, "-o", color=WONG["blue"], lw=2.5, ms=5)
    pk = np.argmax(freq_best)
    ax.plot(freqs[pk], freq_best[pk] * 1e3, "rp", ms=14)
    ax.annotate(f"{freqs[pk]:.0f} kHz", (freqs[pk], freq_best[pk] * 1e3),
                textcoords="offset points", xytext=(8, 8), fontsize=10,
                fontweight="bold", color="red")
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel(r"Best $|dZ/dV|$ (m$\Omega$/mL)")
    ax.set_title("(a) Absolute sensitivity")

    # (b) SNR — key plot
    ax = axes[0, 1]
    ax.semilogx(freqs, freq_snr, "-o", color=WONG["vermillion"], lw=2.5, ms=5)
    spk = np.argmax(freq_snr)
    ax.plot(freqs[spk], freq_snr[spk], "rp", ms=14)
    ax.annotate(f"{freqs[spk]:.0f} kHz\nSNR = {freq_snr[spk]:.2f}",
                (freqs[spk], freq_snr[spk]),
                textcoords="offset points", xytext=(12, -18), fontsize=10,
                fontweight="bold", color="red",
                arrowprops=dict(arrowstyle="->", color="red", lw=1.5))
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("SNR per mL")
    ax.set_title("(b) SNR vs frequency")

    # (c) Signal vs noise
    ax = axes[0, 2]
    ax.semilogx(freqs, freq_best * 1e3, "-o", color=WONG["blue"], lw=2, ms=4,
                label="Signal |dZ/dV|")
    ax.semilogx(freqs, noise * 1e3, "-s", color=WONG["red"], lw=2, ms=4,
                label="Noise floor")
    ax.axvline(freqs[spk], color="0.5", ls="--", lw=1.2, alpha=0.6,
               label=f"SNR peak ({freqs[spk]:.0f} kHz)")
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel(r"m$\Omega$/mL  or  m$\Omega$")
    ax.set_title("(c) Signal vs noise")
    ax.legend(fontsize=9, loc="upper right")

    # (d) Contact impedance
    ax = axes[1, 0]
    ax.semilogx(freqs, z_contacts, "-s", color="k", lw=2.5, ms=5, mfc="k")
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel(r"$z_c$ ($\Omega\!\cdot\!$cm$^2$)")
    ax.set_title("(d) Contact impedance")

    # (e) Optimal config absolute
    ax = axes[1, 1]
    ax.semilogx(freqs, opt_abs * 1e3, "-^", color=WONG["blue"], lw=2.5, ms=5)
    pk3 = np.argmax(opt_abs)
    ax.plot(freqs[pk3], opt_abs[pk3] * 1e3, "rp", ms=12)
    ax.annotate(f"{freqs[pk3]:.0f} kHz", (freqs[pk3], opt_abs[pk3] * 1e3),
                textcoords="offset points", xytext=(8, 8), fontsize=10,
                fontweight="bold", color="red")
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel(r"$|dZ/dV|$ (m$\Omega$/mL)")
    ax.set_title("(e) Optimal config: |dZ/dV|")

    # (f) Optimal config SNR
    ax = axes[1, 2]
    ax.semilogx(freqs, opt_snr, "-^", color=WONG["vermillion"], lw=2.5, ms=5)
    ospk = np.argmax(opt_snr)
    ax.plot(freqs[ospk], opt_snr[ospk], "rp", ms=12)
    ax.annotate(f"{freqs[ospk]:.0f} kHz\nSNR = {opt_snr[ospk]:.2f}",
                (freqs[ospk], opt_snr[ospk]),
                textcoords="offset points", xytext=(12, -18), fontsize=10,
                fontweight="bold", color="red",
                arrowprops=dict(arrowstyle="->", color="red", lw=1.5))
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("SNR per mL")
    ax.set_title("(f) Optimal config: SNR")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    _save_figure(fig, fig_dir, "fig5_frequency_response")


# =====================================================================
# Figure 6: Multi-Frequency Bladder Isolation
# =====================================================================

def _fig6_multifreq_isolation(results: Dict, fig_dir: Path):
    fig, axes = plt.subplots(2, 2, figsize=(16, 13))
    fig.suptitle("Figure 6: Multi-Frequency Bladder Isolation",
                 fontsize=18, fontweight="bold")

    freqs = results["freqs_kHz"]
    ts = results["tissue_sigma"]
    bl_shape = results["bl_spectral_shape"]
    art_shape = results["art_spectral_shape"]
    iso = results["isolation"]
    volumes = results["volumes"]
    Z_vf = results["Z_vol_freq"]
    dZ_bl = results["dZ_bladder_per_mL"]
    dZ_art = results["Z_artifact_freq"]

    # (a) Tissue conductivity spectra
    ax = axes[0, 0]
    tissue_plot = [
        ("urine", "Urine (1.75 S/m, flat)", WONG["vermillion"], "-"),
        ("muscle", "Muscle (\u03b2 dispersion)", WONG["blue"], "-"),
        ("fat", "Fat (low, weak disp.)", WONG["green"], "--"),
        ("background", "Connective tissue", WONG["purple"], "--"),
        ("bone_avg", "Bone (very low)", WONG["grey"], ":"),
        ("bladder_wall", "Bladder wall", WONG["brown"], "-."),
    ]
    for tissue, label, color, ls in tissue_plot:
        if tissue in ts:
            ax.semilogx(freqs, ts[tissue], ls, color=color, lw=2.5, label=label)
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Conductivity (S/m)")
    ax.set_title("(a) Tissue conductivity spectra")
    ax.legend(fontsize=8.5, loc="center right")

    # (b) Spectral fingerprints: bladder vs respiratory artifact
    ax = axes[0, 1]
    ax.semilogx(freqs, bl_shape, "-o", color=WONG["blue"], lw=2.5, ms=7,
                label="Bladder (volume change)")
    ax.semilogx(freqs, art_shape, "-s", color=WONG["vermillion"], lw=2.5, ms=7,
                label="Respiratory artifact")
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Normalised |dZ| (a.u.)")
    ax.set_title("(b) Spectral fingerprints")
    ax.legend(fontsize=10)
    ax.set_ylim(-0.05, 1.15)

    # (c) Dual-frequency isolation: Z_isolated vs volume
    ax = axes[1, 0]
    Z_iso = iso["Z_isolated"]
    f1, f2 = iso["f1_kHz"], iso["f2_kHz"]
    alpha = iso["alpha"]
    # Also plot single-frequency for comparison
    mid_idx = len(freqs) // 2
    ax.plot(volumes, Z_vf[:, mid_idx] * 1e3, "s--", color=WONG["grey"], lw=1.5,
            ms=7, label=f"Single freq ({freqs[mid_idx]:.0f} kHz)")
    ax.plot(volumes, Z_iso * 1e3, "o-", color=WONG["blue"], lw=2.5, ms=8,
            label=f"Isolated: Z({f1:.0f}) \u2212 {alpha:.2f}\u00b7Z({f2:.0f})")
    p_iso = np.polyfit(volumes, Z_iso * 1e3, 1)
    v_fine = np.linspace(100, 500, 100)
    ax.plot(v_fine, np.polyval(p_iso, v_fine), "--", color=WONG["blue"], lw=1.5,
            alpha=0.6)
    ax.set_xlabel("Bladder Volume (mL)")
    ax.set_ylabel(r"Impedance (m$\Omega$)")
    ax.set_title("(c) Dual-frequency isolated signal")
    ax.legend(fontsize=9)

    # (d) Explanation / method summary with SNR comparison
    ax = axes[1, 1]
    ax.axis("off")

    single_snr_val = iso.get("single_snr", 0)
    iso_snr_val = iso.get("iso_snr", 0)
    single_f = iso.get("single_freq_kHz", freqs[len(freqs)//2])
    art_rej = iso.get("art_rejection", 0)
    iso_noise_val = iso.get("iso_noise", 0)

    explanation = (
        "MULTI-FREQUENCY BLADDER ISOLATION\n"
        f"{'='*46}\n"
        "\n"
        "Principle:\n"
        "  Urine is a pure ionic conductor \u2192\n"
        "  conductivity is CONSTANT (1.75 S/m)\n"
        "  across 1\u2013500 kHz.\n"
        "\n"
        "  Surrounding tissues show \u03b2 dispersion:\n"
        "  conductivity INCREASES with frequency\n"
        "  as cell membranes become transparent.\n"
        "\n"
        "Method: dual-frequency subtraction\n"
        f"  Z_isolated = Z(f\u2081) \u2212 \u03b1\u00b7Z(f\u2082)\n"
        f"  f\u2081 = {f1:.0f} kHz,  f\u2082 = {f2:.0f} kHz,  \u03b1 = {alpha:.3f}\n"
        "\n"
        "  \u03b1 chosen to maximise ISOLATED SNR\n"
        "  (cancel artefacts, preserve bladder,\n"
        "   minimise noise amplification).\n"
        "\n"
        f"{'='*46}\n"
        f"  Single-freq SNR/mL:  {single_snr_val:.2f}\n"
        f"    (at {single_f:.0f} kHz)\n"
        f"  Isolated SNR/mL:     {iso_snr_val:.2f}\n"
        f"  Isolated sensitivity:{iso['sensitivity']*1e3:.3f} m\u03a9/mL\n"
        f"  Isolated noise:      {iso_noise_val*1e3:.3f} m\u03a9\n"
        f"  Artefact rejection:  {art_rej:.0f}\u00d7\n"
        f"{'='*46}\n"
        "\n"
        "Trade-off: small SNR loss vs complete\n"
        "immunity to respiratory/motion artefacts."
    )
    ax.text(0.02, 0.97, explanation, transform=ax.transAxes, fontsize=11.5,
            fontfamily="monospace", va="top", ha="left", linespacing=1.25)

    plt.tight_layout()
    _save_figure(fig, fig_dir, "fig6_multifreq_isolation")


# =====================================================================
# Helper plot functions
# =====================================================================

def _plot_cross_section(ax, mesh: TorsoMesh, z_slice: float):
    """Axial cross-section colored by tissue type."""
    ec = mesh.element_centroids()
    labels = get_tissue_labels(mesh)

    dz = max(mesh.nodes[mesh.elements].std(axis=1)[:, 2].mean(), 0.5)
    mask = np.abs(ec[:, 2] - z_slice) < dz
    if not np.any(mask):
        dz *= 3
        mask = np.abs(ec[:, 2] - z_slice) < dz

    cx, cy = ec[mask, 0], ec[mask, 1]
    lbl = labels[mask]

    cmap = ListedColormap(TISSUE_COLORS)
    norm = BoundaryNorm(np.arange(-0.5, len(TISSUE_NAMES) + 0.5), len(TISSUE_NAMES))
    sc = ax.scatter(cx, cy, c=lbl, cmap=cmap, norm=norm, s=6, edgecolors="none")
    cbar = plt.colorbar(sc, ax=ax, ticks=range(len(TISSUE_NAMES)), shrink=0.85)
    cbar.ax.set_yticklabels(TISSUE_NAMES, fontsize=8)

    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(TORSO_RX * np.cos(theta), TORSO_RY * np.sin(theta), "k-", lw=1.5)
    ax.set_xlabel("X (cm)")
    ax.set_ylabel("Y (cm)")
    ax.set_aspect("equal")


def _plot_sagittal_section(ax, mesh: TorsoMesh, x_slice: float):
    """Sagittal (y-z) cross-section."""
    ec = mesh.element_centroids()
    labels = get_tissue_labels(mesh)

    dx = max(mesh.nodes[mesh.elements].std(axis=1)[:, 0].mean(), 0.5)
    mask = np.abs(ec[:, 0] - x_slice) < dx
    if not np.any(mask):
        dx *= 3
        mask = np.abs(ec[:, 0] - x_slice) < dx

    cy, cz = ec[mask, 1], ec[mask, 2]
    lbl = labels[mask]

    cmap = ListedColormap(TISSUE_COLORS)
    norm = BoundaryNorm(np.arange(-0.5, len(TISSUE_NAMES) + 0.5), len(TISSUE_NAMES))
    sc = ax.scatter(cy, cz, c=lbl, cmap=cmap, norm=norm, s=6, edgecolors="none")
    cbar = plt.colorbar(sc, ax=ax, ticks=range(len(TISSUE_NAMES)), shrink=0.85)
    cbar.ax.set_yticklabels(TISSUE_NAMES, fontsize=8)
    ax.set_xlabel("Y (cm)")
    ax.set_ylabel("Z (cm)")
    ax.set_aspect("equal")


def _plot_3d_sensitivity(ax, mesh: TorsoMesh, sensitivity: np.ndarray):
    """3D scatter cloud colored by Jacobian sensitivity."""
    ec = mesh.element_centroids()
    # Subsample for performance
    n = len(sensitivity)
    step = max(1, n // 5000)
    idx = np.arange(0, n, step)

    sv = sensitivity[idx]
    sv_log = np.log10(sv + 1e-15)

    sc = ax.scatter(ec[idx, 0], ec[idx, 1], ec[idx, 2],
                    c=sv_log, cmap="hot", s=2, alpha=0.6,
                    edgecolors="none", depthshade=False)
    plt.colorbar(sc, ax=ax, label=r"log$_{10}$ |J|", shrink=0.6, pad=0.12)
    _setup_3d_axes(ax, elev=25, azim=-55)


def _save_figure(fig, fig_dir: Path, name: str):
    """Save figure as PNG at 300 DPI."""
    png_path = fig_dir / f"{name}.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"      Saved: {png_path}")
