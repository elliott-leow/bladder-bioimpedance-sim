#!/usr/bin/env python3
"""
Sweep 200+ electrode configurations to find optimal placement.

Evaluates configurations varying:
- Number of electrode rings (1-4)
- Ring z-positions (3-12 cm above model origin)
- Electrodes per ring (4, 8, 16)

Phase 1: Quick sweep using forward solve with adjacent stim patterns.
Phase 2: Detailed transfer impedance analysis of top 10 configs.
"""

import sys
import time
import itertools
import io
import contextlib

import numpy as np

sys.path.insert(0, ".")

from bladder_sim.model import (
    build_pelvis_model, TORSO_RX, TORSO_RY, TORSO_H,
    DEFAULT_ELEC_RADIUS,
)
from bladder_sim.fem import (
    ForwardModel, forward_solve, compute_transfer_impedance,
    build_adjacent_stim_patterns,
)
from bladder_sim.mesh import (
    create_torso_mesh, compute_electrode_positions, _place_electrodes, TorsoMesh,
)
from bladder_sim.tissue_properties import get_contact_impedance

MAX_EDGE = 1.5  # coarser mesh for speed
FREQ_KHZ = 50.0


def generate_configs():
    """Generate 200+ electrode configurations."""
    configs = []

    # ---- 1-ring configs ----
    # n_per_ring in [4, 8, 16], z from 3 to 12 step 0.5
    for n in [4, 8, 16]:
        for z in np.arange(3.0, 12.5, 0.5):
            configs.append((n, [round(z, 1)]))

    # ---- 2-ring configs ----
    # n_per_ring in [4, 8], z1 from 3 to 10, z2 from z1+1 to 12
    for n in [4, 8]:
        for z1 in range(3, 11):
            for z2 in range(z1 + 1, 13):
                configs.append((n, [float(z1), float(z2)]))

    # ---- 3-ring configs ----
    three_ring = set()
    # Evenly spaced around various centers
    for c in np.arange(5, 11, 0.5):
        for s in [1, 1.5, 2, 2.5, 3]:
            z1, z3 = round(c - s, 1), round(c + s, 1)
            if z1 >= 3.0 and z3 <= 12.0:
                three_ring.add((z1, round(c, 1), z3))
    # Coarse-grid combinations
    for combo in itertools.combinations([3, 5, 7, 9, 11], 3):
        three_ring.add(tuple(float(z) for z in combo))
    for n in [4, 8]:
        for zs in sorted(three_ring):
            configs.append((n, list(zs)))

    # ---- 4-ring configs ----
    four_ring = set()
    for c in np.arange(6, 10, 0.5):
        for s in [1, 1.5, 2, 2.5]:
            zs = tuple(round(c + (i - 1.5) * s, 1) for i in range(4))
            if zs[0] >= 3.0 and zs[-1] <= 12.0:
                four_ring.add(zs)
    for combo in itertools.combinations([3, 5, 7, 9, 11], 4):
        four_ring.add(tuple(float(z) for z in combo))
    for n in [4, 8]:
        for zs in sorted(four_ring):
            configs.append((n, list(zs)))

    return configs


def evaluate_config_quick(base_mesh, n_per_ring, ring_z, z_c):
    """Phase 1: quick evaluation using adjacent-drive forward solve."""
    ring_z = np.asarray(ring_z)
    n_rings = len(ring_z)
    n_elec = n_per_ring * n_rings

    # Place electrodes on shared base mesh
    elec_pos = compute_electrode_positions(TORSO_RX, TORSO_RY, n_per_ring, ring_z)
    electrodes = _place_electrodes(base_mesh, elec_pos, DEFAULT_ELEC_RADIUS, z_c)
    mesh = TorsoMesh(
        nodes=base_mesh.nodes,
        elements=base_mesh.elements,
        boundary_faces=base_mesh.boundary_faces,
        boundary_normals=base_mesh.boundary_normals,
        electrodes=electrodes,
    )

    with contextlib.redirect_stdout(io.StringIO()):
        fmdl, _ = build_pelvis_model(
            300, mesh=mesh, freq_kHz=FREQ_KHZ, stim_pattern="adjacent",
            n_per_ring=n_per_ring, ring_z=ring_z,
        )
        _, img_lo = build_pelvis_model(
            100, mesh=mesh, stim_pattern="none",
            n_per_ring=n_per_ring, ring_z=ring_z,
        )
        img_lo.fwd_model = fmdl
        _, img_hi = build_pelvis_model(
            500, mesh=mesh, stim_pattern="none",
            n_per_ring=n_per_ring, ring_z=ring_z,
        )
        img_hi.fwd_model = fmdl

    data_lo = forward_solve(img_lo)
    data_hi = forward_solve(img_hi)

    I_amp = np.max(np.abs(fmdl.stimulation[0].stim_pattern))
    dZ = (data_hi.meas - data_lo.meas) / I_amp

    return {
        "max_dZ": np.max(np.abs(dZ)) / 400.0,
        "norm_dZ": np.linalg.norm(dZ) / 400.0,
        "n_meas": len(dZ),
        "n_elec": n_elec,
    }


def evaluate_config_detailed(base_mesh, n_per_ring, ring_z, z_c):
    """Phase 2: transfer impedance + exhaustive 4-electrode search."""
    ring_z = np.asarray(ring_z)
    n_elec = n_per_ring * len(ring_z)

    elec_pos = compute_electrode_positions(TORSO_RX, TORSO_RY, n_per_ring, ring_z)
    electrodes = _place_electrodes(base_mesh, elec_pos, DEFAULT_ELEC_RADIUS, z_c)
    mesh = TorsoMesh(
        nodes=base_mesh.nodes,
        elements=base_mesh.elements,
        boundary_faces=base_mesh.boundary_faces,
        boundary_normals=base_mesh.boundary_normals,
        electrodes=electrodes,
    )

    with contextlib.redirect_stdout(io.StringIO()):
        # Need a fmdl for the mesh (stim patterns don't matter for transfer Z)
        fmdl = ForwardModel(mesh=mesh)
        _, img_lo = build_pelvis_model(
            100, mesh=mesh, freq_kHz=FREQ_KHZ, stim_pattern="none",
            n_per_ring=n_per_ring, ring_z=ring_z,
        )
        img_lo.fwd_model = fmdl
        _, img_hi = build_pelvis_model(
            500, mesh=mesh, freq_kHz=FREQ_KHZ, stim_pattern="none",
            n_per_ring=n_per_ring, ring_z=ring_z,
        )
        img_hi.fwd_model = fmdl

    Z_lo = compute_transfer_impedance(fmdl, img_lo)
    Z_hi = compute_transfer_impedance(fmdl, img_hi)
    dZ = Z_hi - Z_lo

    # Exhaustive 4-electrode search
    best_val = 0.0
    best_quad = (0, 0, 0, 0)
    for a in range(n_elec):
        for b in range(a + 1, n_elec):
            dV = dZ[a, :] - dZ[b, :]
            sense = [s for s in range(n_elec) if s != a and s != b]
            for ci in range(len(sense)):
                for di in range(ci + 1, len(sense)):
                    c, d = sense[ci], sense[di]
                    val = abs(dV[c] - dV[d]) / 400.0
                    if val > best_val:
                        best_val = val
                        best_quad = (a, b, c, d)

    return best_val, best_quad


def main():
    configs = generate_configs()
    n_total = len(configs)
    print(f"{'=' * 72}")
    print(f"  ELECTRODE CONFIGURATION SWEEP")
    print(f"  {n_total} configurations, max_edge={MAX_EDGE} cm, freq={FREQ_KHZ} kHz")
    print(f"{'=' * 72}")

    # Build base mesh once (no electrodes)
    print("\nBuilding base mesh... ", end="", flush=True)
    base_mesh = create_torso_mesh(
        rx=TORSO_RX, ry=TORSO_RY, height=TORSO_H, max_edge=MAX_EDGE,
    )
    print(f"{base_mesh.n_elements} elements, {base_mesh.n_nodes} nodes\n")

    z_c = get_contact_impedance(FREQ_KHZ)
    results = []
    t_start = time.time()

    # ==================================================================
    # PHASE 1: Quick sweep
    # ==================================================================
    print("--- PHASE 1: Quick sweep (adjacent-drive forward solve) ---\n")

    for i, (n_per_ring, ring_z_list) in enumerate(configs):
        ring_z = np.array(ring_z_list)
        n_rings = len(ring_z)
        n_elec = n_per_ring * n_rings

        try:
            metrics = evaluate_config_quick(base_mesh, n_per_ring, ring_z, z_c)
            results.append({
                "n_per_ring": n_per_ring,
                "ring_z": ring_z_list,
                "n_rings": n_rings,
                "n_elec": n_elec,
                **metrics,
            })
        except Exception:
            pass

        if (i + 1) % 25 == 0 or i == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed if elapsed > 0 else 1
            eta = (n_total - i - 1) / rate
            last = results[-1] if results else None
            s = f"max={last['max_dZ']*1e3:.3f}" if last else "---"
            print(
                f"  [{i+1:3d}/{n_total}] {s} mOhm/mL  "
                f"({elapsed:.0f}s elapsed, ETA {eta/60:.1f} min)"
            )

    phase1_time = time.time() - t_start
    results.sort(key=lambda r: r["max_dZ"], reverse=True)

    print(f"\nPhase 1 complete: {len(results)} configs in {phase1_time/60:.1f} min\n")

    # ==================================================================
    # Display Phase 1 results
    # ==================================================================
    print(f"{'=' * 72}")
    print(f"  PHASE 1 RESULTS — TOP 20 (of {len(results)})")
    print(f"{'=' * 72}")
    header = (
        f"  {'#':>3s}  {'Config':22s}  {'Elec':>4s}  {'Meas':>5s}  "
        f"{'max|dZ/dV|':>11s}  {'||dZ||/dV':>11s}"
    )
    print(header)
    print(
        f"  {'':3s}  {'':22s}  {'':>4s}  {'':>5s}  "
        f"{'(mΩ/mL)':>11s}  {'(mΩ/mL)':>11s}"
    )
    print(f"  {'-' * 66}")
    for i, r in enumerate(results[:20]):
        ring_str = ",".join(f"{z:.0f}" for z in r["ring_z"])
        label = f"{r['n_rings']}R x {r['n_per_ring']:2d}e  z=[{ring_str}]"
        print(
            f"  {i+1:3d}  {label:22s}  {r['n_elec']:4d}  {r['n_meas']:5d}  "
            f"{r['max_dZ']*1e3:11.4f}  {r['norm_dZ']*1e3:11.4f}"
        )

    # Best by ring count
    print(f"\n  --- Best by ring count ---")
    for nr in [1, 2, 3, 4]:
        sub = [r for r in results if r["n_rings"] == nr]
        if sub:
            b = sub[0]
            rs = ",".join(f"{z:.0f}" for z in b["ring_z"])
            print(
                f"  {nr}-ring:  {b['n_per_ring']:2d}e/ring  z=[{rs:12s}]  "
                f"max={b['max_dZ']*1e3:.4f} mΩ/mL  ({b['n_elec']} elec)"
            )

    # Best by electrodes per ring
    print(f"\n  --- Best by electrodes per ring ---")
    for ne in [4, 8, 16]:
        sub = [r for r in results if r["n_per_ring"] == ne]
        if sub:
            b = sub[0]
            rs = ",".join(f"{z:.0f}" for z in b["ring_z"])
            print(
                f"  {ne:2d}e/ring:  {b['n_rings']}R  z=[{rs:12s}]  "
                f"max={b['max_dZ']*1e3:.4f} mΩ/mL  ({b['n_elec']} elec)"
            )

    # ==================================================================
    # PHASE 2: Detailed transfer impedance for top 10
    # ==================================================================
    top_n = min(10, len(results))
    print(f"\n{'=' * 72}")
    print(f"  PHASE 2: Transfer impedance analysis — top {top_n} configs")
    print(f"{'=' * 72}\n")

    t2_start = time.time()
    detailed = []

    for rank, r in enumerate(results[:top_n]):
        ring_z = np.array(r["ring_z"])
        ring_str = ",".join(f"{z:.0f}" for z in r["ring_z"])
        label = f"{r['n_rings']}R x {r['n_per_ring']}e  z=[{ring_str}]"
        print(f"  [{rank+1:2d}/{top_n}] {label:25s} ... ", end="", flush=True)

        t0 = time.time()
        best_dZ, best_quad = evaluate_config_detailed(
            base_mesh, r["n_per_ring"], ring_z, z_c,
        )
        dt = time.time() - t0

        detailed.append({
            **r,
            "tz_best_dZ": best_dZ,
            "tz_best_quad": best_quad,
        })
        a, b, c, d = best_quad
        print(
            f"|dZ/dV|={best_dZ*1e3:.4f} mΩ/mL  "
            f"drive {a}->{b}, sense {c}->{d}  ({dt:.1f}s)"
        )

    detailed.sort(key=lambda r: r["tz_best_dZ"], reverse=True)

    print(f"\n{'=' * 72}")
    print(f"  FINAL RANKING — Best 4-electrode sensitivity (transfer impedance)")
    print(f"{'=' * 72}")
    for i, r in enumerate(detailed):
        ring_str = ",".join(f"{z:.0f}" for z in r["ring_z"])
        label = f"{r['n_rings']}R x {r['n_per_ring']:2d}e  z=[{ring_str}]"
        a, b, c, d = r["tz_best_quad"]
        n_pr = r["n_per_ring"]
        ring_a = a // n_pr
        ring_b = b // n_pr
        ring_c = c // n_pr
        ring_d = d // n_pr
        print(
            f"  {i+1:2d}. {label:22s}  {r['n_elec']:3d} elec  "
            f"|dZ/dV|={r['tz_best_dZ']*1e3:.4f} mΩ/mL  "
            f"drive {a}(R{ring_a})->{b}(R{ring_b})  "
            f"sense {c}(R{ring_c})->{d}(R{ring_d})"
        )

    total_time = time.time() - t_start
    print(f"\n  Total time: {total_time/60:.1f} minutes")
    print(f"  Configs evaluated: {len(results)}")
    print(f"  Transfer impedance analyses: {len(detailed)}")


if __name__ == "__main__":
    main()
