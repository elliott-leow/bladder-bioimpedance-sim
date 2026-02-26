"""
Analysis functions for bladder bioimpedance simulation.

Includes:
    - Sensitivity analysis (impedance vs volume, Jacobian maps)
    - Exhaustive 4-electrode pattern optimization
    - Multi-frequency sweep with optimal electrode search
    - Belt height optimization
"""

import numpy as np
from typing import Optional, Tuple, Dict, List
from .model import build_pelvis_model, get_bladder_mask, DEFAULT_RING_Z, DEFAULT_N_PER_RING
from .fem import (
    ForwardModel,
    Image,
    forward_solve,
    compute_transfer_impedance,
    compute_jacobian,
)
from .mesh import TorsoMesh
from .tissue_properties import measurement_noise_floor, get_contact_impedance


def run_sensitivity_analysis(
    fmdl: ForwardModel,
    volumes: Optional[np.ndarray] = None,
) -> Dict:
    """
    Impedance vs bladder volume + Jacobian sensitivity maps.

    Parameters
    ----------
    fmdl : ForwardModel
        Pre-built forward model (mesh reused across volumes).
    volumes : array-like, optional
        Bladder volumes to test (mL). Default: [50, 100, 200, 300, 400, 500].

    Returns
    -------
    dict with keys:
        'volumes': volume array
        'Z': (n_vol, n_meas) impedance matrix
        'sens_per_ch': (n_meas,) per-channel sensitivity (Ohm/mL)
        'best_channels': indices of most sensitive channels
        'Z_norm': (n_vol,) aggregate impedance norm
        'jacobian_sens': (n_elem,) total Jacobian sensitivity
        'sens_ratio': bladder-to-background sensitivity ratio
    """
    if volumes is None:
        volumes = np.array([50, 100, 200, 300, 400, 500])
    volumes = np.asarray(volumes)
    n_vol = len(volumes)

    print(f"\n===== SENSITIVITY ANALYSIS =====")

    # Forward solve at each volume
    V_meas = []
    for vi, vol in enumerate(volumes):
        print(f"  Forward solve at {vol:.0f} mL...")
        _, img = build_pelvis_model(vol, mesh=fmdl.mesh)
        img.fwd_model = fmdl
        data = forward_solve(img)
        V_meas.append(data.meas)

    n_meas = len(V_meas[0])
    V_meas = np.array(V_meas)  # (n_vol, n_meas)

    # Convert to impedance
    current = fmdl.stimulation[0].stim_pattern
    I_amp = np.max(np.abs(current))
    Z = V_meas / I_amp  # V/I -> Ohm

    # Per-channel linear sensitivity
    sens_per_ch = np.zeros(n_meas)
    for ch in range(n_meas):
        p = np.polyfit(volumes, Z[:, ch], 1)
        sens_per_ch[ch] = p[0]  # Ohm/mL

    rank = np.argsort(np.abs(sens_per_ch))[::-1]
    n_show = min(10, n_meas)
    best_chs = rank[:n_show]

    print(f"\n--- Per-Channel Sensitivity (top {n_show} of {n_meas}) ---")
    for ci in range(n_show):
        ch = best_chs[ci]
        print(
            f"  Ch {ch:4d}: dZ/dVol = {sens_per_ch[ch]:+.2e} Ohm/mL "
            f"({sens_per_ch[ch]*1e3:+.3f} mOhm/mL)"
        )

    Z_norm = np.sqrt(np.sum(Z ** 2, axis=1))
    p_agg = np.polyfit(volumes, Z_norm, 1)
    print(f"\n  Aggregate ||Z||: dZ/dVol = {p_agg[0]:.2e} Ohm/mL ({p_agg[0]*1e3:.3f} mOhm/mL)")

    # Jacobian at 300 mL
    print(f"\nComputing Jacobian at 300 mL...")
    _, img_ref = build_pelvis_model(300, mesh=fmdl.mesh, stim_pattern="none")
    img_ref.fwd_model = fmdl
    J = compute_jacobian(img_ref)
    total_sens = np.sum(np.abs(J), axis=0)  # (n_elem,)

    # Sensitivity ratio
    is_bl = get_bladder_mask(fmdl.mesh, 300)
    if np.any(is_bl):
        s_bl = np.mean(total_sens[is_bl])
        s_rest = np.mean(total_sens[~is_bl])
        ratio = s_bl / s_rest if s_rest > 0 else 0
        print(f"\n--- Jacobian Sensitivity Ratio ---")
        print(f"  Urine elements:      {np.sum(is_bl)} / {len(is_bl)}")
        print(f"  Mean |J| (urine):    {s_bl:.4e}")
        print(f"  Mean |J| (other):    {s_rest:.4e}")
        print(f"  Ratio (urine/other): {ratio:.3f}")
    else:
        ratio = 0.0
        print("  WARNING: No bladder elements found")

    # Literature comparison
    best_dZ = np.max(np.abs(sens_per_ch))
    total_dZ = best_dZ * (volumes[-1] - volumes[0])
    print(f"\n--- Literature Comparison ---")
    print(f"  Best-channel |dZ/dVol|: {best_dZ:.2e} Ohm/mL ({best_dZ*1e3:.3f} mOhm/mL)")
    print(f"  Total |dZ| over {volumes[0]:.0f}-{volumes[-1]:.0f} mL: {total_dZ*1e3:.2f} mOhm")
    print(f"  3D anatomical models:   0.05 - 5 mOhm/mL  (Leonhardt 2012, Schlebusch 2014)")
    print(f"  Simplified/tank models: 1 - 50 mOhm/mL    (Kim et al.)")
    if best_dZ >= 0.00005:
        print(f"  STATUS: Within expected range for realistic 3D anatomy.")
    else:
        print(f"  STATUS: Very low ({best_dZ*1e3:.4f} mOhm/mL). Check model.")

    return {
        "volumes": volumes,
        "Z": Z,
        "V_meas": V_meas,
        "sens_per_ch": sens_per_ch,
        "best_channels": best_chs,
        "Z_norm": Z_norm,
        "jacobian": J,
        "jacobian_sens": total_sens,
        "sens_ratio": ratio,
    }


def optimize_stim_patterns(
    fmdl: ForwardModel,
    freq_kHz: float = 50.0,
) -> Dict:
    """
    Exhaustive 4-electrode pattern optimization using transfer impedance.

    Evaluates ALL possible drive/sense combinations:
        - N-1 forward solves per volume (100 mL and 500 mL)
        - C(N,2) drive pairs x C(N-2,2) sense pairs

    Parameters
    ----------
    fmdl : ForwardModel
    freq_kHz : float

    Returns
    -------
    dict with optimization results.
    """
    mesh = fmdl.mesh
    n = mesh.n_electrodes
    # Detect actual electrodes per ring from electrode positions
    if n > 0:
        ez = np.array([e.center[2] for e in mesh.electrodes])
        ring_zs = np.unique(np.round(ez, 1))
        n_rings = len(ring_zs)
        n_per_ring = n // n_rings if n_rings > 0 else n
    else:
        n_per_ring = DEFAULT_N_PER_RING
        n_rings = 1
    n_drive = n * (n - 1) // 2

    print(f"\n=== Exhaustive 4-Electrode Pattern Optimization ===")
    print(f"Electrodes:     {n} ({n_rings} rings x {n_per_ring} per ring)")
    print(f"Drive pairs:    C({n},2) = {n_drive}")

    # Transfer impedance matrices
    print(f"\nComputing transfer impedance...")
    print(f"  100 mL: ", end="", flush=True)
    _, img_lo = build_pelvis_model(100, mesh=mesh, freq_kHz=freq_kHz, stim_pattern="none")
    img_lo.fwd_model = fmdl
    Z_lo = compute_transfer_impedance(fmdl, img_lo)
    print("done")

    print(f"  500 mL: ", end="", flush=True)
    _, img_hi = build_pelvis_model(500, mesh=mesh, freq_kHz=freq_kHz, stim_pattern="none")
    img_hi.fwd_model = fmdl
    Z_hi = compute_transfer_impedance(fmdl, img_hi)
    print("done")

    dZ = Z_hi - Z_lo
    dV_mL = 400.0

    # Evaluate all drive/sense combinations
    print(f"\nEvaluating all combinations...")
    drive_pairs = np.zeros((n_drive, 2), dtype=int)
    drive_best_dZ = np.zeros(n_drive)
    drive_best_sc = np.zeros((n_drive, 2), dtype=int)
    drive_all_rms = np.zeros(n_drive)

    di = 0
    for a in range(n):
        for b in range(a + 1, n):
            drive_pairs[di] = [a, b]

            # Voltage change at every electrode for drive a->b
            dV_elec = dZ[a, :] - dZ[b, :]  # superposition

            # Sense electrodes: all except a and b
            sense = [s for s in range(n) if s != a and s != b]
            n_s = len(sense)

            # All sense pairs - vectorized
            best_val = 0.0
            best_c, best_d = 0, 0
            dV_vals = []
            for ci_idx in range(n_s):
                for di_idx in range(ci_idx + 1, n_s):
                    c = sense[ci_idx]
                    d = sense[di_idx]
                    val = abs(dV_elec[c] - dV_elec[d]) / dV_mL
                    dV_vals.append(val)
                    if val > best_val:
                        best_val = val
                        best_c, best_d = c, d

            drive_best_dZ[di] = best_val
            drive_best_sc[di] = [best_c, best_d]
            if dV_vals:
                drive_all_rms[di] = np.sqrt(np.mean(np.array(dV_vals) ** 2))
            di += 1

    # Global optimum
    gi = np.argmax(drive_best_dZ)
    opt_a, opt_b = drive_pairs[gi]
    opt_c, opt_d = drive_best_sc[gi]
    global_best = drive_best_dZ[gi]

    ring_a = opt_a // n_per_ring
    ring_b = opt_b // n_per_ring
    ring_c = opt_c // n_per_ring
    ring_d = opt_d // n_per_ring

    print(f"\n{'='*56}")
    print(f"GLOBAL OPTIMUM:")
    print(f"  Drive:  elec {opt_a} (ring {ring_a}) -> elec {opt_b} (ring {ring_b})")
    print(f"  Sense:  elec {opt_c} (ring {ring_c}) -> elec {opt_d} (ring {ring_d})")
    print(f"  |dZ/dV| = {global_best:.2e} Ohm/mL ({global_best*1e3:.3f} mOhm/mL)")
    print(f"{'='*56}")

    # Classify drive types
    is_within = np.array(
        [drive_pairs[i, 0] // n_per_ring == drive_pairs[i, 1] // n_per_ring for i in range(n_drive)]
    )
    is_cross = ~is_within
    ring_dist = np.abs(
        drive_pairs[:, 0] // n_per_ring - drive_pairs[:, 1] // n_per_ring
    )

    if np.any(is_within):
        print(f"  Within-ring: best = {np.max(drive_best_dZ[is_within])*1e3:.3f} mOhm/mL")
    if np.any(is_cross):
        print(f"  Cross-ring:  best = {np.max(drive_best_dZ[is_cross])*1e3:.3f} mOhm/mL")

    # Top 20
    sorted_idx = np.argsort(drive_best_dZ)[::-1]
    print(f"\nTop 10 configurations:")
    for ri in range(min(10, n_drive)):
        idx = sorted_idx[ri]
        a, b = drive_pairs[idx]
        c, d = drive_best_sc[idx]
        print(
            f"  {ri+1:2d}. Drive {a:2d}->{b:2d}, Sense {c:2d}->{d:2d}: "
            f"{drive_best_dZ[idx]*1e3:.3f} mOhm/mL"
        )

    return {
        "drive_pairs": drive_pairs,
        "drive_best_dZ": drive_best_dZ,
        "drive_best_sc": drive_best_sc,
        "drive_all_rms": drive_all_rms,
        "is_within": is_within,
        "is_cross": is_cross,
        "ring_dist": ring_dist,
        "optimal": {
            "drive": (opt_a, opt_b),
            "sense": (opt_c, opt_d),
            "dZ_per_mL": global_best,
        },
        "Z_lo": Z_lo,
        "Z_hi": Z_hi,
    }


def frequency_sweep(
    fmdl: ForwardModel,
    freqs_kHz: Optional[np.ndarray] = None,
) -> Dict:
    """
    Multi-frequency 4-electrode optimization.

    At each frequency:
        1. Update conductivities (Gabriel 1996 models)
        2. Compute transfer impedance at 100 & 500 mL
        3. Find best 4-electrode combination
        4. Track overall best across frequencies

    Parameters
    ----------
    fmdl : ForwardModel
    freqs_kHz : array-like, optional
        Frequencies to test. Default: 32 log-spaced from 1 to 500 kHz.

    Returns
    -------
    dict with frequency sweep results.
    """
    if freqs_kHz is None:
        freqs_kHz = np.logspace(np.log10(1), np.log10(500), 32)
    freqs_kHz = np.asarray(freqs_kHz)
    n_freq = len(freqs_kHz)
    mesh = fmdl.mesh
    n_elec = mesh.n_electrodes
    dV_mL = 400.0

    print(f"\n=== Frequency Sweep: 4-Electrode Optimization ===")
    print(f"Electrodes:  {n_elec}")
    print(f"Frequencies: {n_freq} points, {freqs_kHz[0]:.1f} - {freqs_kHz[-1]:.0f} kHz")

    freq_best_abs_dZ = np.zeros(n_freq)
    freq_best_drive = np.zeros((n_freq, 2), dtype=int)
    freq_best_sense = np.zeros((n_freq, 2), dtype=int)
    z_contacts = np.zeros(n_freq)
    noise_floor = np.zeros(n_freq)

    # Electrode area for noise model
    if mesh.electrodes and mesh.electrodes[0].area > 0:
        elec_area = mesh.electrodes[0].area
    else:
        elec_area = 0.785  # default pi*0.5^2

    overall_best_dZ = 0
    overall_best_freq_idx = 0
    overall_best_drive = (0, 0)
    overall_best_sense = (0, 0)

    for fi, f in enumerate(freqs_kHz):
        print(f"  [{fi+1:2d}/{n_freq}] {f:7.1f} kHz ... ", end="", flush=True)

        _, img_lo = build_pelvis_model(100, mesh=mesh, freq_kHz=f, stim_pattern="none")
        img_lo.fwd_model = fmdl
        _, img_hi = build_pelvis_model(500, mesh=mesh, freq_kHz=f, stim_pattern="none")
        img_hi.fwd_model = fmdl

        z_contacts[fi] = get_contact_impedance(f)
        noise_floor[fi] = measurement_noise_floor(f, electrode_area_cm2=elec_area)

        Z_lo = compute_transfer_impedance(fmdl, img_lo)
        Z_hi = compute_transfer_impedance(fmdl, img_hi)
        dZ = Z_hi - Z_lo

        best_dZ = 0
        best_drv = (0, 0)
        best_sns = (0, 0)

        for a in range(n_elec):
            for b in range(a + 1, n_elec):
                dV_elec = dZ[a, :] - dZ[b, :]
                sense = [s for s in range(n_elec) if s != a and s != b]
                for ci in range(len(sense)):
                    for di in range(ci + 1, len(sense)):
                        c, d = sense[ci], sense[di]
                        val = abs(dV_elec[c] - dV_elec[d]) / dV_mL
                        if val > best_dZ:
                            best_dZ = val
                            best_drv = (a, b)
                            best_sns = (c, d)

        freq_best_abs_dZ[fi] = best_dZ
        freq_best_drive[fi] = best_drv
        freq_best_sense[fi] = best_sns

        if best_dZ > overall_best_dZ:
            overall_best_dZ = best_dZ
            overall_best_freq_idx = fi
            overall_best_drive = best_drv
            overall_best_sense = best_sns

        snr_val = best_dZ / noise_floor[fi] if noise_floor[fi] > 0 else 0
        print(f"|dZ/dV| = {best_dZ*1e3:.4f} mOhm/mL, "
              f"noise = {noise_floor[fi]*1e3:.4f} mOhm, "
              f"SNR/mL = {snr_val:.2f}")

    # SNR per frequency (best config at each freq)
    freq_snr = np.where(noise_floor > 0, freq_best_abs_dZ / noise_floor, 0)
    snr_peak_idx = np.argmax(freq_snr)

    # Compute frequency response of the overall SNR-optimal config
    # Use the config at the SNR-peak frequency as the optimal
    snr_opt_drive = tuple(freq_best_drive[snr_peak_idx])
    snr_opt_sense = tuple(freq_best_sense[snr_peak_idx])
    opt_a, opt_b = snr_opt_drive
    opt_c, opt_d = snr_opt_sense
    opt_abs_dZ = np.zeros(n_freq)
    opt_Z_base = np.zeros(n_freq)

    print(f"\nComputing frequency response for SNR-optimal config: "
          f"drive {opt_a}->{opt_b}, sense {opt_c}->{opt_d}")

    for fi, f in enumerate(freqs_kHz):
        _, img_lo = build_pelvis_model(100, mesh=mesh, freq_kHz=f, stim_pattern="none")
        img_lo.fwd_model = fmdl
        _, img_hi = build_pelvis_model(500, mesh=mesh, freq_kHz=f, stim_pattern="none")
        img_hi.fwd_model = fmdl
        _, img_300 = build_pelvis_model(300, mesh=mesh, freq_kHz=f, stim_pattern="none")
        img_300.fwd_model = fmdl

        Z_lo = compute_transfer_impedance(fmdl, img_lo)
        Z_hi = compute_transfer_impedance(fmdl, img_hi)
        Z_300 = compute_transfer_impedance(fmdl, img_300)

        dZ = Z_hi - Z_lo
        dV_opt = (dZ[opt_a, opt_c] - dZ[opt_b, opt_c]) - (
            dZ[opt_a, opt_d] - dZ[opt_b, opt_d]
        )
        opt_abs_dZ[fi] = abs(dV_opt) / dV_mL

        V_base = (Z_300[opt_a, opt_c] - Z_300[opt_b, opt_c]) - (
            Z_300[opt_a, opt_d] - Z_300[opt_b, opt_d]
        )
        opt_Z_base[fi] = abs(V_base)

    opt_rel_dZ = np.where(opt_Z_base > 0, opt_abs_dZ / opt_Z_base, 0)
    opt_snr = np.where(noise_floor > 0, opt_abs_dZ / noise_floor, 0)
    opt_snr_peak_idx = np.argmax(opt_snr)

    # Summary
    print(f"\n{'='*60}")
    print(f"  FREQUENCY SWEEP SUMMARY")
    print(f"{'='*60}")
    print(f"  SNR-optimal drive:  {opt_a} -> {opt_b}")
    print(f"  SNR-optimal sense:  {opt_c} -> {opt_d}")
    print(f"\n  --- Absolute Sensitivity ---")
    print(f"  Peak |dZ/dV|: {np.max(freq_best_abs_dZ)*1e3:.4f} mOhm/mL "
          f"at {freqs_kHz[np.argmax(freq_best_abs_dZ)]:.1f} kHz")
    print(f"\n  --- SNR (Signal / Measurement Noise) ---")
    print(f"  Peak SNR/mL: {freq_snr[snr_peak_idx]:.2f} at {freqs_kHz[snr_peak_idx]:.1f} kHz")
    print(f"  Noise at peak: {noise_floor[snr_peak_idx]*1e3:.4f} mOhm")
    print(f"  Signal at peak: {freq_best_abs_dZ[snr_peak_idx]*1e3:.4f} mOhm/mL")
    print(f"\n  --- Optimal Config Frequency Response ---")
    print(f"  Peak config SNR/mL: {opt_snr[opt_snr_peak_idx]:.2f} "
          f"at {freqs_kHz[opt_snr_peak_idx]:.1f} kHz")
    peak_rel_idx = np.argmax(opt_rel_dZ)
    print(f"  Peak |dZ/Z|: {opt_rel_dZ[peak_rel_idx]*1e6:.2f} ppm/mL "
          f"at {freqs_kHz[peak_rel_idx]:.1f} kHz")
    print(f"{'='*60}")

    return {
        "freqs_kHz": freqs_kHz,
        "freq_best_abs_dZ": freq_best_abs_dZ,
        "freq_best_drive": freq_best_drive,
        "freq_best_sense": freq_best_sense,
        "z_contacts": z_contacts,
        "noise_floor": noise_floor,
        "freq_snr": freq_snr,
        "opt_abs_dZ": opt_abs_dZ,
        "opt_Z_base": opt_Z_base,
        "opt_rel_dZ": opt_rel_dZ,
        "opt_snr": opt_snr,
        "optimal": {
            "drive": snr_opt_drive,
            "sense": snr_opt_sense,
            "freq_kHz": freqs_kHz[snr_peak_idx],
            "dZ_per_mL": freq_best_abs_dZ[snr_peak_idx],
            "snr_per_mL": freq_snr[snr_peak_idx],
        },
    }


def optimize_electrode_placement(
    belt_heights: Optional[np.ndarray] = None,
    n_elec: int = 16,
    max_edge: float = 1.0,
) -> Dict:
    """
    Sweep electrode belt height (single ring) to find optimal placement.

    Parameters
    ----------
    belt_heights : array-like, optional
        Z-positions to test. Default: [4, 5, 6, 7, 8, 9, 10].
    n_elec : int
        Electrodes per ring.
    max_edge : float
        Mesh element size.

    Returns
    -------
    dict with results.
    """
    if belt_heights is None:
        belt_heights = np.array([4, 5, 6, 7, 8, 9, 10])
    belt_heights = np.asarray(belt_heights)

    print(f"\n=== Belt Height Optimization ({n_elec} electrodes) ===")

    results = np.full((len(belt_heights), 3), np.nan)  # [z, sens_ratio, dZ_per_mL]

    for ci, ez in enumerate(belt_heights):
        print(f"  Config {ci+1}/{len(belt_heights)}: belt z = {ez:.0f} cm")
        try:
            fmdl, img = build_pelvis_model(
                300, ring_z=np.array([ez]), n_per_ring=n_elec,
                max_edge=max_edge, stim_pattern="adjacent"
            )

            # Jacobian sensitivity
            J = compute_jacobian(img)
            total_sens = np.sum(np.abs(J), axis=0)
            is_bl = get_bladder_mask(fmdl.mesh, 300)

            if not np.any(is_bl):
                print(f"    No bladder elements.")
                results[ci, 0] = ez
                continue

            sens_ratio = np.mean(total_sens[is_bl]) / max(np.mean(total_sens[~is_bl]), 1e-15)

            # dZ measurement
            _, img_lo = build_pelvis_model(100, mesh=fmdl.mesh, stim_pattern="none")
            img_lo.fwd_model = fmdl
            _, img_hi = build_pelvis_model(500, mesh=fmdl.mesh, stim_pattern="none")
            img_hi.fwd_model = fmdl

            data_lo = forward_solve(img_lo)
            data_hi = forward_solve(img_hi)

            I_amp = np.max(np.abs(fmdl.stimulation[0].stim_pattern))
            dZ_vec = (data_hi.meas - data_lo.meas) / I_amp
            dZ_per_mL = np.linalg.norm(dZ_vec) / 400.0

            results[ci] = [ez, sens_ratio, dZ_per_mL]
            print(
                f"    Sens ratio: {sens_ratio:.3f}, "
                f"||dZ||/dV: {dZ_per_mL:.2e} Ohm/mL ({dZ_per_mL*1e3:.3f} mOhm/mL)"
            )
        except Exception as e:
            print(f"    FAILED: {e}")
            results[ci, 0] = ez

    # Best
    valid = ~np.isnan(results[:, 1])
    if np.any(valid):
        vr = results[valid]
        best_idx = np.argmax(vr[:, 1])
        print(f"\n{'='*40}")
        print(f"OPTIMAL: belt z = {vr[best_idx, 0]:.0f} cm")
        print(f"  Sensitivity ratio: {vr[best_idx, 1]:.3f}")
        print(f"  ||dZ||/dV: {vr[best_idx, 2]:.2e} Ohm/mL ({vr[best_idx, 2]*1e3:.3f} mOhm/mL)")
        print(f"{'='*40}")

    return {"belt_heights": belt_heights, "results": results}


def get_angular_sep(a: int, b: int, n_per_ring: int) -> int:
    """Angular separation between two electrodes within their rings."""
    pos_a = a % n_per_ring
    pos_b = b % n_per_ring
    d = abs(pos_a - pos_b)
    return min(d, n_per_ring - d)


def multifreq_bladder_isolation(
    fmdl: ForwardModel,
    optimal_config: Optional[Dict] = None,
    freqs_kHz: Optional[np.ndarray] = None,
) -> Dict:
    """
    Multi-frequency bioimpedance spectroscopy for bladder signal isolation.

    Demonstrates that measuring at multiple frequencies allows separation of
    bladder volume changes from confounding tissue changes (respiratory motion,
    blood flow, bowel activity).

    Key principle: urine conductivity is frequency-INDEPENDENT (ionic conductor),
    while surrounding tissues show beta dispersion (conductivity increases with
    frequency as cell membranes become transparent to AC).  Therefore bladder and
    confounders have distinct spectral fingerprints, enabling dual-frequency
    subtraction.

    Parameters
    ----------
    fmdl : ForwardModel
    optimal_config : dict, optional
        {'drive': (a, b), 'sense': (c, d)} from optimize_stim_patterns().
    freqs_kHz : array-like, optional
        Default: [5, 10, 25, 50, 100, 200, 500].

    Returns
    -------
    dict with multi-frequency analysis results.
    """
    mesh = fmdl.mesh
    n_elec = mesh.n_electrodes

    if freqs_kHz is None:
        freqs_kHz = np.array([5.0, 10.0, 25.0, 50.0, 100.0, 200.0, 500.0])
    freqs_kHz = np.asarray(freqs_kHz)
    n_freq = len(freqs_kHz)

    # Detect electrode config
    ez = np.array([e.center[2] for e in mesh.electrodes])
    ring_zs = np.unique(np.round(ez, 1))
    n_rings = len(ring_zs)
    n_per_ring = n_elec // n_rings if n_rings > 0 else n_elec

    # Use optimal config or default (anterior-posterior on rings 2-3)
    if optimal_config:
        opt_a, opt_b = optimal_config["drive"]
        opt_c, opt_d = optimal_config["sense"]
    else:
        half = n_per_ring // 2
        opt_a = 2 * n_per_ring
        opt_b = 2 * n_per_ring + half
        opt_c = 3 * n_per_ring
        opt_d = 3 * n_per_ring + half

    volumes = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
    n_vol = len(volumes)

    # Tissue labels for respiratory artifact simulation
    from .model import get_tissue_labels
    labels = get_tissue_labels(mesh, 300)
    is_muscle = labels == 3

    # Electrode area for noise model
    if mesh.electrodes and mesh.electrodes[0].area > 0:
        elec_area = mesh.electrodes[0].area
    else:
        elec_area = 0.785  # default pi*0.5^2

    print(f"\n=== Multi-Frequency Bladder Isolation ===")
    print(f"Frequencies: {n_freq} points, {freqs_kHz[0]:.0f} - {freqs_kHz[-1]:.0f} kHz")
    print(f"Config: drive {opt_a}->{opt_b}, sense {opt_c}->{opt_d}")

    Z_vol_freq = np.zeros((n_vol, n_freq))
    Z_artifact_freq = np.zeros(n_freq)
    tissue_sigma = {}

    from .tissue_properties import get_conductivity as _get_sigma

    for fi, f in enumerate(freqs_kHz):
        print(f"  [{fi+1:2d}/{n_freq}] {f:6.0f} kHz: ", end="", flush=True)

        # Tissue conductivity spectra (first iteration only)
        if fi == 0:
            for tissue in ["urine", "muscle", "fat", "bone_avg", "background",
                           "skin", "bladder_wall", "bowel_eff"]:
                tissue_sigma[tissue] = np.array(
                    [_get_sigma(tissue, ff) for ff in freqs_kHz]
                )

        # Z at each bladder volume
        for vi, vol in enumerate(volumes):
            _, img = build_pelvis_model(vol, mesh=mesh, freq_kHz=f, stim_pattern="none")
            img.fwd_model = fmdl
            Z = compute_transfer_impedance(fmdl, img)
            Z_vol_freq[vi, fi] = (
                (Z[opt_a, opt_c] - Z[opt_b, opt_c])
                - (Z[opt_a, opt_d] - Z[opt_b, opt_d])
            )

        # Respiratory artifact: +5% muscle conductivity at 300 mL
        _, img_base = build_pelvis_model(300, mesh=mesh, freq_kHz=f, stim_pattern="none")
        img_base.fwd_model = fmdl
        img_artifact = Image(fwd_model=fmdl, elem_data=img_base.elem_data.copy())
        img_artifact.elem_data[is_muscle] *= 1.05

        Z_base = compute_transfer_impedance(fmdl, img_base)
        Z_art = compute_transfer_impedance(fmdl, img_artifact)

        def _z4e(Zm):
            return (Zm[opt_a, opt_c] - Zm[opt_b, opt_c]) - (
                Zm[opt_a, opt_d] - Zm[opt_b, opt_d]
            )

        Z_artifact_freq[fi] = _z4e(Z_art) - _z4e(Z_base)

        dZ_bl = (Z_vol_freq[-1, fi] - Z_vol_freq[0, fi]) / (volumes[-1] - volumes[0])
        print(f"dZ_bl={dZ_bl*1e3:.3f}, dZ_resp={Z_artifact_freq[fi]*1e3:.3f} mOhm")

    # Spectral sensitivity
    dZ_bladder_per_mL = np.zeros(n_freq)
    for fi in range(n_freq):
        p = np.polyfit(volumes, Z_vol_freq[:, fi], 1)
        dZ_bladder_per_mL[fi] = p[0]

    bl_shape = np.abs(dZ_bladder_per_mL) / np.max(np.abs(dZ_bladder_per_mL))
    art_shape = np.abs(Z_artifact_freq) / np.max(np.abs(Z_artifact_freq))

    # --- Noise at each frequency ---
    noise_per_freq = np.array([
        measurement_noise_floor(f, electrode_area_cm2=elec_area)
        for f in freqs_kHz
    ])

    # --- Single-frequency SNR (baseline comparison) ---
    single_snr = np.where(
        noise_per_freq > 0, np.abs(dZ_bladder_per_mL) / noise_per_freq, 0
    )
    best_single_idx = np.argmax(single_snr)

    # --- Find optimal dual-frequency pair ---
    # Maximize ISOLATED SNR = |isolated_signal| / isolated_noise
    # where:
    #   alpha = dZ_artifact(f1) / dZ_artifact(f2)   (cancels artifact)
    #   isolated_signal = dZ_bladder(f1) - alpha * dZ_bladder(f2)
    #   isolated_noise  = sqrt(noise(f1)^2 + alpha^2 * noise(f2)^2)
    best_iso_snr = 0.0
    best_f1, best_f2 = 0, 1
    best_alpha = 1.0

    for f1 in range(n_freq):
        for f2 in range(n_freq):
            if f1 == f2 or abs(Z_artifact_freq[f2]) < 1e-15:
                continue
            alpha = Z_artifact_freq[f1] / Z_artifact_freq[f2]
            iso_signal = abs(dZ_bladder_per_mL[f1] - alpha * dZ_bladder_per_mL[f2])
            iso_noise = np.sqrt(noise_per_freq[f1]**2 + alpha**2 * noise_per_freq[f2]**2)
            iso_snr = iso_signal / iso_noise if iso_noise > 0 else 0

            if iso_snr > best_iso_snr:
                best_iso_snr = iso_snr
                best_f1, best_f2 = f1, f2
                best_alpha = alpha

    # Isolated bladder measurement at optimal pair
    Z_isolated = Z_vol_freq[:, best_f1] - best_alpha * Z_vol_freq[:, best_f2]
    iso_sens = np.polyfit(volumes, Z_isolated, 1)[0]
    iso_noise = np.sqrt(
        noise_per_freq[best_f1]**2 + best_alpha**2 * noise_per_freq[best_f2]**2
    )
    # Artifact rejection
    art_residual = abs(
        Z_artifact_freq[best_f1] - best_alpha * Z_artifact_freq[best_f2]
    )
    art_rejection = abs(Z_artifact_freq[best_single_idx]) / max(art_residual, 1e-15)

    print(f"\n--- Single-Frequency Baseline ---")
    print(f"  Best single-freq SNR/mL: {single_snr[best_single_idx]:.2f} "
          f"at {freqs_kHz[best_single_idx]:.0f} kHz")
    print(f"  Signal: {np.abs(dZ_bladder_per_mL[best_single_idx])*1e3:.3f} mOhm/mL, "
          f"Noise: {noise_per_freq[best_single_idx]*1e3:.3f} mOhm")
    print(f"  BUT vulnerable to respiratory artifact: "
          f"{np.abs(Z_artifact_freq[best_single_idx])*1e3:.1f} mOhm")

    print(f"\n--- Noise-Optimal Dual-Frequency Pair ---")
    print(f"  f1 = {freqs_kHz[best_f1]:.0f} kHz, "
          f"f2 = {freqs_kHz[best_f2]:.0f} kHz, alpha = {best_alpha:.3f}")
    print(f"  Isolated signal: {abs(iso_sens)*1e3:.3f} mOhm/mL")
    print(f"  Isolated noise:  {iso_noise*1e3:.3f} mOhm")
    print(f"  Isolated SNR/mL: {best_iso_snr:.2f}")
    print(f"  Artifact rejection: {art_rejection:.0f}x vs single-freq")
    print(f"\n  Trade-off: SNR drops from {single_snr[best_single_idx]:.2f} "
          f"(single) to {best_iso_snr:.2f} (isolated)")
    print(f"  but artifact immunity gained.")

    return {
        "freqs_kHz": freqs_kHz,
        "volumes": volumes,
        "Z_vol_freq": Z_vol_freq,
        "Z_artifact_freq": Z_artifact_freq,
        "dZ_bladder_per_mL": dZ_bladder_per_mL,
        "tissue_sigma": tissue_sigma,
        "noise_per_freq": noise_per_freq,
        "single_snr": single_snr,
        "bl_spectral_shape": bl_shape,
        "art_spectral_shape": art_shape,
        "isolation": {
            "f1_kHz": freqs_kHz[best_f1],
            "f2_kHz": freqs_kHz[best_f2],
            "alpha": best_alpha,
            "Z_isolated": Z_isolated,
            "sensitivity": iso_sens,
            "iso_noise": iso_noise,
            "iso_snr": best_iso_snr,
            "art_rejection": art_rejection,
            "single_freq_kHz": freqs_kHz[best_single_idx],
            "single_snr": single_snr[best_single_idx],
        },
    }
