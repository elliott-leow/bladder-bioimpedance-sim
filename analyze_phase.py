#!/usr/bin/env python3
"""
Phase sensitivity analysis for bladder bioimpedance.

Compares impedance magnitude vs phase sensitivity to bladder volume
using complex admittivity (sigma + j*omega*eps_0*eps_r).

Key question: Does impedance phase provide additional information
beyond magnitude for detecting bladder volume changes?
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bladder_sim.model import build_pelvis_model
from bladder_sim.fem import forward_solve, compute_transfer_impedance
from bladder_sim.tissue_properties import get_complex_admittivity, get_conductivity


def run_phase_analysis():
    print("=" * 60)
    print("  PHASE SENSITIVITY ANALYSIS")
    print("  Complex admittivity: gamma = sigma + j*omega*eps0*eps_r")
    print("=" * 60)

    # Configuration
    n_per_ring = 16
    ring_z = np.array([4.0, 6.0, 8.0, 10.0])
    freq_kHz = 50.0
    volumes = np.array([100, 200, 300, 400, 500])

    # ─── Complex admittivity comparison ────────────────────────────
    print("\n--- Tissue Complex Admittivity at 50 kHz ---")
    print(f"  {'Tissue':15s}  {'sigma':>8s}  {'|gamma|':>8s}  {'phase':>8s}  {'eps_r_eff':>10s}")
    for tissue in ["urine", "muscle", "fat", "skin", "bladder_wall",
                    "bone_avg", "bowel_eff", "background"]:
        sigma = get_conductivity(tissue, freq_kHz)
        gamma = get_complex_admittivity(tissue, freq_kHz)
        phase = np.degrees(np.angle(gamma))
        print(f"  {tissue:15s}  {sigma:8.4f}  {abs(gamma):8.4f}  {phase:+7.2f}°  {gamma.imag / (2*np.pi*freq_kHz*1e3*8.854e-12):10.0f}")

    # ─── Build mesh once ───────────────────────────────────────────
    print("\n--- Building model (real conductivity) ---")
    fmdl_real, img_real = build_pelvis_model(
        300, freq_kHz=freq_kHz, n_per_ring=n_per_ring, ring_z=ring_z,
        stim_pattern="adjacent",
    )
    mesh = fmdl_real.mesh

    print("\n--- Building model (complex admittivity) ---")
    fmdl_cplx, img_cplx = build_pelvis_model(
        300, mesh=mesh, freq_kHz=freq_kHz, n_per_ring=n_per_ring, ring_z=ring_z,
        stim_pattern="adjacent", use_complex=True,
    )

    # ─── Compare real vs complex forward solutions ─────────────────
    print("\n--- Forward solve: real vs complex at 300 mL ---")
    sol_real = forward_solve(img_real)
    sol_cplx = forward_solve(img_cplx)

    Z_real = sol_real.meas
    Z_cplx = sol_cplx.meas

    print(f"  Real:    {len(Z_real)} measurements, median |Z| = {np.median(np.abs(Z_real)):.6e} V")
    print(f"  Complex: {len(Z_cplx)} measurements, median |Z| = {np.median(np.abs(Z_cplx)):.6e} V")
    print(f"  Complex phase: median = {np.median(np.degrees(np.angle(Z_cplx))):.3f}°, "
          f"range [{np.min(np.degrees(np.angle(Z_cplx))):.3f}°, "
          f"{np.max(np.degrees(np.angle(Z_cplx))):.3f}°]")

    # ─── Volume sweep with both real and complex ───────────────────
    print(f"\n--- Volume sweep (complex admittivity, {freq_kHz} kHz) ---")

    Z_mag_all = []
    Z_phase_all = []
    Z_real_part_all = []
    Z_imag_part_all = []

    for vol in volumes:
        fmdl_v, img_v = build_pelvis_model(
            vol, mesh=mesh, freq_kHz=freq_kHz, n_per_ring=n_per_ring,
            ring_z=ring_z, stim_pattern="adjacent", use_complex=True,
        )
        sol = forward_solve(img_v)
        Z = sol.meas

        Z_mag_all.append(np.abs(Z))
        Z_phase_all.append(np.angle(Z))  # radians
        Z_real_part_all.append(np.real(Z))
        Z_imag_part_all.append(np.imag(Z))

        print(f"  {vol} mL: median |Z|={np.median(np.abs(Z)):.6e}, "
              f"median phase={np.median(np.degrees(np.angle(Z))):.3f}°")

    Z_mag = np.array(Z_mag_all)      # (n_vol, n_meas)
    Z_phase = np.array(Z_phase_all)
    Z_re = np.array(Z_real_part_all)
    Z_im = np.array(Z_imag_part_all)
    n_meas = Z_mag.shape[1]

    # ─── Per-channel sensitivity: magnitude vs phase ───────────────
    print(f"\n--- Sensitivity Analysis ({n_meas} channels) ---")

    # Linear fit dZ/dV for each channel
    sens_mag = np.zeros(n_meas)     # d|Z|/dV
    sens_phase = np.zeros(n_meas)   # d(phase)/dV
    sens_re = np.zeros(n_meas)      # d(Re(Z))/dV
    sens_im = np.zeros(n_meas)      # d(Im(Z))/dV

    for ch in range(n_meas):
        sens_mag[ch] = np.polyfit(volumes, Z_mag[:, ch], 1)[0]
        sens_phase[ch] = np.polyfit(volumes, Z_phase[:, ch], 1)[0]
        sens_re[ch] = np.polyfit(volumes, Z_re[:, ch], 1)[0]
        sens_im[ch] = np.polyfit(volumes, Z_im[:, ch], 1)[0]

    # Best channels
    best_mag_ch = np.argmax(np.abs(sens_mag))
    best_phase_ch = np.argmax(np.abs(sens_phase))
    best_re_ch = np.argmax(np.abs(sens_re))
    best_im_ch = np.argmax(np.abs(sens_im))

    print(f"\n  Component     Best channel  Sensitivity")
    print(f"  {'─'*48}")
    print(f"  |Z| (mag)     Ch {best_mag_ch:5d}     {sens_mag[best_mag_ch]*1e3:+.4f} mOhm/mL")
    print(f"  Re(Z)         Ch {best_re_ch:5d}     {sens_re[best_re_ch]*1e3:+.4f} mOhm/mL")
    print(f"  Im(Z)         Ch {best_im_ch:5d}     {sens_im[best_im_ch]*1e3:+.4f} mOhm/mL")
    print(f"  Phase(Z)      Ch {best_phase_ch:5d}     {sens_phase[best_phase_ch]*1e6:+.4f} udeg/mL")
    print(f"                                  = {np.degrees(sens_phase[best_phase_ch])*1e3:+.6f} mdeg/mL")

    # Ratio of imaginary to real sensitivity
    re_sens_best = np.max(np.abs(sens_re))
    im_sens_best = np.max(np.abs(sens_im))
    ratio = im_sens_best / re_sens_best if re_sens_best > 0 else 0

    print(f"\n  Ratio |Im sensitivity| / |Re sensitivity| = {ratio:.4f}")
    print(f"  → Imaginary part is {ratio*100:.1f}% of real part sensitivity")

    # ─── Phase sensitivity across frequencies ──────────────────────
    print(f"\n--- Phase Sensitivity vs Frequency ---")
    freqs = np.array([5, 10, 25, 50, 100, 200, 500])
    print(f"  {'Freq':>8s}  {'|dZ/dV|':>12s}  {'d(Re)/dV':>12s}  {'d(Im)/dV':>12s}  {'d(phase)/dV':>14s}  {'Im/Re ratio':>12s}")

    for freq in freqs:
        # Solve at two volumes
        Z_lo = []
        Z_hi = []
        for vol, store in [(100, Z_lo), (500, Z_hi)]:
            fmdl_f, img_f = build_pelvis_model(
                vol, mesh=mesh, freq_kHz=freq, n_per_ring=n_per_ring,
                ring_z=ring_z, stim_pattern="adjacent", use_complex=True,
            )
            sol_f = forward_solve(img_f)
            store.append(sol_f.meas)

        Z_lo = Z_lo[0]
        Z_hi = Z_hi[0]
        dZ = Z_hi - Z_lo
        dV = 400.0  # 500 - 100 mL

        # Best channel by magnitude
        best = np.argmax(np.abs(dZ))
        dz = dZ[best] / dV

        dz_mag = abs(dz) * 1e3
        dz_re = np.real(dz) * 1e3
        dz_im = np.imag(dz) * 1e3
        dz_phase = np.degrees(np.angle(Z_hi[best]) - np.angle(Z_lo[best])) / dV
        im_re = abs(dz_im / dz_re) if abs(dz_re) > 0 else 0

        print(f"  {freq:6.0f} kHz  {dz_mag:10.4f} mΩ  {dz_re:+10.4f} mΩ  {dz_im:+10.4f} mΩ  "
              f"{dz_phase*1e3:+12.6f} mdeg  {im_re:10.4f}")

    # ─── Summary ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  PHASE ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"")
    print(f"  At 50 kHz with 64 electrodes (adjacent drive):")
    print(f"  Best magnitude sensitivity: {np.max(np.abs(sens_mag))*1e3:.4f} mOhm/mL")
    print(f"  Best Re(Z) sensitivity:     {re_sens_best*1e3:.4f} mOhm/mL")
    print(f"  Best Im(Z) sensitivity:     {im_sens_best*1e3:.4f} mOhm/mL")
    print(f"  Im/Re ratio:                {ratio:.4f} ({ratio*100:.1f}%)")
    print(f"")

    if ratio < 0.1:
        print(f"  CONCLUSION: Phase/imaginary component provides <10% of the")
        print(f"  real-part sensitivity at bioimpedance frequencies (1-500 kHz).")
        print(f"  This is expected: at these frequencies, tissue permittivity")
        print(f"  contributes a small reactive component (omega*eps << sigma).")
        print(f"  Magnitude-only analysis captures >99% of the bladder signal.")
    elif ratio < 0.5:
        print(f"  CONCLUSION: Phase provides moderate additional sensitivity")
        print(f"  ({ratio*100:.0f}% of real part). Consider complex measurement")
        print(f"  for improved sensitivity, especially at higher frequencies")
        print(f"  where permittivity effects are more significant.")
    else:
        print(f"  CONCLUSION: Phase provides significant additional sensitivity")
        print(f"  ({ratio*100:.0f}% of real part). Complex measurement recommended.")

    print(f"\n  The AD5940 measures both magnitude and phase natively.")
    print(f"  Phase data can be used as a secondary indicator or for")
    print(f"  frequency-dependent tissue characterization (Cole model).")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_phase_analysis()
