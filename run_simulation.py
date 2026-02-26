#!/usr/bin/env python3
"""
Bladder Bioimpedance Simulation — Main Runner
==============================================

3D FEM simulation to determine whether bioimpedance can detect urine
output and to optimize electrode placement.

Pure Python implementation (numpy + scipy + matplotlib).
Male pelvis model with 15 tissue types, volume-dependent bladder wall,
Complete Electrode Model (CEM).

Usage:
    python run_simulation.py              # Full simulation
    python run_simulation.py --quick      # Quick validation only
    python run_simulation.py --no-freq    # Skip frequency sweep (slow)
    python run_simulation.py --no-mfbis   # Skip multi-frequency isolation

Dependencies:
    numpy, scipy, matplotlib
"""

import sys
import time
import argparse
import numpy as np

# Add parent directory to path for package import
sys.path.insert(0, ".")

from bladder_sim.model import build_pelvis_model, get_bladder_mask
from bladder_sim.fem import forward_solve, compute_transfer_impedance
from bladder_sim.analysis import (
    run_sensitivity_analysis,
    optimize_stim_patterns,
    frequency_sweep,
    multifreq_bladder_isolation,
)
from bladder_sim.figures import generate_publication_figures


def main():
    parser = argparse.ArgumentParser(description="Bladder Bioimpedance Simulation")
    parser.add_argument("--quick", action="store_true", help="Quick validation only")
    parser.add_argument("--no-freq", action="store_true", help="Skip frequency sweep")
    parser.add_argument("--no-mfbis", action="store_true", help="Skip multi-frequency isolation")
    parser.add_argument("--no-figs", action="store_true", help="Skip figure generation")
    parser.add_argument("--max-edge", type=float, default=1.0, help="Mesh edge size (cm)")
    parser.add_argument("--n-per-ring", type=int, default=16, help="Electrodes per ring")
    args = parser.parse_args()

    print("=" * 60)
    print("  BLADDER BIOIMPEDANCE SIMULATION")
    print("  3D FEM with Complete Electrode Model")
    print("=" * 60)

    # ====================================================================
    # STEP 0: Build the model
    # ====================================================================
    print("\n===== STEP 0: Build Model =====")
    t0 = time.time()
    fmdl, img = build_pelvis_model(
        bladder_volume_mL=300,
        freq_kHz=50.0,
        n_per_ring=args.n_per_ring,
        max_edge=args.max_edge,
        stim_pattern="cross_ring",
    )
    print(f"Model built in {time.time()-t0:.1f} seconds")

    mesh = fmdl.mesh
    n_elec = mesh.n_electrodes
    n_elem = mesh.n_elements
    n_nodes = mesh.n_nodes
    print(f"[MESH] {n_elem} elements, {n_nodes} nodes, {n_elec} electrodes, 3D")

    # ====================================================================
    # STEP 1: Validation
    # ====================================================================
    print("\n===== STEP 1: Validation =====")

    for vol in [100, 300, 500]:
        is_bl = get_bladder_mask(mesh, vol)
        print(f"[BLAD] {vol} mL: {np.sum(is_bl)} urine elements ({100*np.sum(is_bl)/n_elem:.1f}%)")
        assert np.sum(is_bl) > 0, f"No bladder elements at {vol} mL"

    is_bl300 = get_bladder_mask(mesh, 300)
    ec = mesh.element_centroids()
    bl_ctr = ec[is_bl300].mean(axis=0)
    print(f"[BLAD] Centre: ({bl_ctr[0]:.1f}, {bl_ctr[1]:.1f}, {bl_ctr[2]:.1f}) cm")
    assert bl_ctr[1] > 0, "Bladder should be anterior (y > 0)"

    print("\n[FWD] Forward solve at 300 mL...")
    t1 = time.time()
    data = forward_solve(img)
    n_meas = len(data.meas)
    print(f"[FWD] {n_meas} measurements in {time.time()-t1:.1f}s")
    print(f"[FWD] Voltage: median={np.median(data.meas)*1e3:.4e} mV, "
          f"range=[{np.min(data.meas)*1e3:.4e}, {np.max(data.meas)*1e3:.4e}] mV")
    assert np.all(np.isfinite(data.meas)), "Non-finite voltages!"

    print("\n[SENS] Quick sensitivity check (100 mL vs 500 mL)...")
    _, img_lo = build_pelvis_model(100, mesh=mesh, stim_pattern="none")
    img_lo.fwd_model = fmdl
    _, img_hi = build_pelvis_model(500, mesh=mesh, stim_pattern="none")
    img_hi.fwd_model = fmdl
    d_lo = forward_solve(img_lo)
    d_hi = forward_solve(img_hi)

    I_amp = np.max(np.abs(fmdl.stimulation[0].stim_pattern))
    dZ_ch = (d_hi.meas - d_lo.meas) / I_amp
    best_dZ_per_mL = np.max(np.abs(dZ_ch)) / 400.0
    total_dZ = np.max(np.abs(dZ_ch))
    print(f"[SENS] Best-channel |dZ/dVol| = {best_dZ_per_mL:.2e} Ohm/mL "
          f"({best_dZ_per_mL*1e3:.3f} mOhm/mL)")
    print(f"[SENS] Total best-channel dZ (400mL range) = {total_dZ*1e3:.3f} mOhm")
    print(f"[SENS] Literature range (3D anatomical models): 0.05 - 5 mOhm/mL")
    print(f"[SENS] Literature range (simplified/tank): 1 - 50 mOhm/mL")

    if best_dZ_per_mL >= 0.001:
        print("[SENS] PASS: Within literature range for 3D models")
    elif best_dZ_per_mL >= 0.00005:
        print(f"[SENS] OK: {best_dZ_per_mL*1e3:.3f} mOhm/mL — within range for realistic 3D anatomy")
    else:
        print(f"[SENS] LOW ({best_dZ_per_mL*1e3:.4f} mOhm/mL) — check model")

    print("\n===== VALIDATION COMPLETE =====")

    if args.quick:
        print("\n[Quick mode] Stopping after validation.")
        return

    # ====================================================================
    # STEP 2: Sensitivity Analysis
    # ====================================================================
    print("\n===== STEP 2: Sensitivity Analysis =====")
    sens_results = run_sensitivity_analysis(fmdl)

    # ====================================================================
    # STEP 3: Stimulation Pattern Optimization
    # ====================================================================
    print("\n===== STEP 3: Stimulation Pattern Optimization =====")
    pattern_results = optimize_stim_patterns(fmdl)

    # ====================================================================
    # STEP 4: Frequency Sweep
    # ====================================================================
    freq_results = None
    if not args.no_freq:
        print("\n===== STEP 4: Frequency Sweep =====")
        freqs = np.logspace(np.log10(1), np.log10(500), 12)
        freq_results = frequency_sweep(fmdl, freqs_kHz=freqs)
    else:
        print("\n===== STEP 4: Frequency Sweep (SKIPPED) =====")

    # ====================================================================
    # STEP 5: Multi-Frequency Bladder Isolation
    # ====================================================================
    mfbis_results = None
    if not args.no_mfbis:
        print("\n===== STEP 5: Multi-Frequency Bladder Isolation =====")
        mfbis_results = multifreq_bladder_isolation(
            fmdl,
            optimal_config=pattern_results["optimal"] if pattern_results else None,
        )
    else:
        print("\n===== STEP 5: Multi-Frequency Isolation (SKIPPED) =====")

    # ====================================================================
    # STEP 6: Publication Figures
    # ====================================================================
    if not args.no_figs:
        print("\n===== STEP 6: Publication Figures =====")
        generate_publication_figures(
            mesh=mesh,
            sensitivity_results=sens_results,
            pattern_results=pattern_results,
            freq_results=freq_results,
            multifreq_results=mfbis_results,
            fig_dir="figures",
        )

    print("\n===== SIMULATION COMPLETE =====")


if __name__ == "__main__":
    main()
