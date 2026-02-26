"""
Finite Element Method solver with Complete Electrode Model (CEM).

Solves the forward problem for electrical impedance:
    ∇·(σ∇u) = 0  in Ω
with CEM boundary conditions (Somersalo et al. 1992).

Uses first-order tetrahedral elements with vectorized assembly.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve, factorized
from dataclasses import dataclass, field
from typing import List, Optional
from .mesh import TorsoMesh, Electrode


@dataclass
class StimPattern:
    """A stimulation/measurement pattern."""
    stim_pattern: np.ndarray   # (L,) current injection pattern (A)
    meas_pattern: np.ndarray   # (M, L) measurement pattern matrix


@dataclass
class ForwardModel:
    """Complete forward model for bioimpedance simulation."""
    mesh: TorsoMesh
    stimulation: List[StimPattern] = field(default_factory=list)

    @property
    def n_elec(self) -> int:
        return self.mesh.n_electrodes

    @property
    def n_elem(self) -> int:
        return self.mesh.n_elements

    @property
    def n_nodes(self) -> int:
        return self.mesh.n_nodes


@dataclass
class Image:
    """Conductivity image on a forward model."""
    fwd_model: ForwardModel
    elem_data: np.ndarray      # (M,) conductivity for each element (S/m)


@dataclass
class SolveData:
    """Result of a forward solve."""
    meas: np.ndarray
    node_voltages: Optional[np.ndarray] = None
    electrode_voltages: Optional[np.ndarray] = None


def _compute_element_gradients(nodes, elements):
    """
    Compute gradient matrices and volumes for all tetrahedra.

    Returns
    -------
    grad_phi : (M, 4, 3) gradient of basis functions
    vol : (M,) element volumes
    """
    p = nodes[elements]  # (M, 4, 3)
    d = p[:, 1:, :] - p[:, 0:1, :]  # (M, 3, 3) edge vectors

    det_d = np.linalg.det(d)
    vol = np.abs(det_d) / 6.0

    # Inverse of edge matrix for gradient coefficients.
    # d has edges as rows: d[e] = [e1; e2; e3] = J^T where J = [e1|e2|e3].
    # Mapping: x-p0 = J·ξ = d^T·ξ => ξ = (d^T)^{-1}·(x-p0) = d^{-T}·(x-p0)
    # ∇ξ_k = column k of d^{-1} (NOT row k!)
    # So grad_phi[e,k,:] should be column k of inv(d[e]).
    good = np.abs(det_d) > 1e-20
    inv_d = np.zeros_like(d)
    inv_d[good] = np.linalg.inv(d[good])

    # Transpose: grad_phi_123[e,k,:] = column k of inv_d = row k of inv_d^T
    grad_phi_123 = np.transpose(inv_d, (0, 2, 1))  # (M, 3, 3)
    grad_phi_0 = -grad_phi_123.sum(axis=1, keepdims=True)  # (M, 1, 3)
    grad_phi = np.concatenate([grad_phi_0, grad_phi_123], axis=1)  # (M, 4, 3)

    return grad_phi, vol


def assemble_stiffness_matrix(mesh: TorsoMesh, sigma: np.ndarray) -> sparse.csr_matrix:
    """
    Assemble the global FEM stiffness matrix.

    A_{ij} = Σ_e σ_e * V_e * (∇φ_i · ∇φ_j)
    """
    n_nodes = mesh.n_nodes
    elems = mesh.elements

    grad_phi, vol = _compute_element_gradients(mesh.nodes, elems)

    # Element stiffness: K_local[e,i,j] = sigma[e] * vol[e] * (grad_i · grad_j)
    sv = sigma * vol  # (M,)
    K_local = np.einsum("eik,ejk->eij", grad_phi, grad_phi)  # (M, 4, 4)
    # Use multiplication (not *=) to allow float->complex promotion
    K_local = K_local * sv[:, None, None]

    # Assemble into sparse matrix
    rows = elems[:, :, None].repeat(4, axis=2).ravel()
    cols = elems[:, None, :].repeat(4, axis=1).ravel()
    vals = K_local.ravel()

    A = sparse.coo_matrix((vals, (rows, cols)), shape=(n_nodes, n_nodes))
    return A.tocsr()


def assemble_cem_system(mesh: TorsoMesh, sigma: np.ndarray) -> sparse.csr_matrix:
    """
    Assemble the full CEM system matrix.

    [A + B,  -C ] [u]   [0]
    [-C^T,    D ] [U] = [I]

    Plus ground constraint: u_gnd = 0 (row/col for a well-connected interior node).
    """
    n_nodes = mesh.n_nodes
    n_elec = mesh.n_electrodes

    A = assemble_stiffness_matrix(mesh, sigma)

    # CEM surface integral terms
    B_rows, B_cols, B_vals = [], [], []
    C_rows, C_cols, C_vals = [], [], []
    D_diag = np.zeros(n_elec)

    for el_idx, electrode in enumerate(mesh.electrodes):
        z_c = max(electrode.z_contact, 1e-10)

        for face in electrode.faces:
            p = mesh.nodes[face]
            v1 = p[1] - p[0]
            v2 = p[2] - p[0]
            fa = 0.5 * np.linalg.norm(np.cross(v1, v2))
            if fa < 1e-15:
                continue

            # B: (1/z_c) * ∫ φ_i φ_j dS = (1/z_c) * A/12 * [2,1,1; 1,2,1; 1,1,2]
            mass = fa / (12.0 * z_c)
            for li in range(3):
                for lj in range(3):
                    w = 2.0 * mass if li == lj else mass
                    B_rows.append(face[li])
                    B_cols.append(face[lj])
                    B_vals.append(w)

            # C: (1/z_c) * ∫ φ_i dS = (1/z_c) * A/3
            c_val = fa / (3.0 * z_c)
            for li in range(3):
                C_rows.append(face[li])
                C_cols.append(el_idx)
                C_vals.append(c_val)

            # D: (1/z_c) * |e_l|
            D_diag[el_idx] += fa / z_c

    B = sparse.coo_matrix(
        (B_vals, (B_rows, B_cols)), shape=(n_nodes, n_nodes)
    ).tocsr() if B_vals else sparse.csr_matrix((n_nodes, n_nodes))

    C = sparse.coo_matrix(
        (C_vals, (C_rows, C_cols)), shape=(n_nodes, n_elec)
    ).tocsr() if C_vals else sparse.csr_matrix((n_nodes, n_elec))

    D = sparse.diags(D_diag, format="csr")

    # Full system
    top = sparse.hstack([A + B, -C], format="csr")
    bot = sparse.hstack([-C.T, D], format="csr")
    system = sparse.vstack([top, bot], format="csr")

    # Ground constraint: pin a well-connected interior node
    gnd = _find_ground_node(mesh)
    system = system.tolil()
    system[gnd, :] = 0
    system[:, gnd] = 0
    system[gnd, gnd] = 1.0
    system = system.tocsr()

    return system, gnd


def _find_ground_node(mesh: TorsoMesh) -> int:
    """Find a well-connected interior node for grounding."""
    # Count element membership for each node
    n_nodes = mesh.n_nodes
    count = np.zeros(n_nodes, dtype=int)
    for node_id in mesh.elements.ravel():
        count[node_id] += 1

    # Find node with most connections that's near the center
    centrality = np.zeros(n_nodes)
    coords = mesh.nodes
    cx = coords[:, 0].mean()
    cy = coords[:, 1].mean()
    cz = coords[:, 2].mean()
    dist_from_center = np.sqrt(
        (coords[:, 0] - cx)**2 + (coords[:, 1] - cy)**2 + (coords[:, 2] - cz)**2
    )
    max_dist = dist_from_center.max()
    centrality = count * (1 - dist_from_center / (max_dist + 1e-10))

    return int(np.argmax(centrality))


def forward_solve(img: Image) -> SolveData:
    """Solve the CEM forward problem for all stimulation patterns."""
    fmdl = img.fwd_model
    mesh = fmdl.mesh
    n_nodes = mesh.n_nodes
    n_elec = mesh.n_electrodes

    K, gnd = assemble_cem_system(mesh, img.elem_data)
    n_total = n_nodes + n_elec

    # Factor the system matrix once
    try:
        solve_fn = factorized(K.tocsc())
    except Exception:
        solve_fn = lambda b: spsolve(K, b)

    all_meas = []
    for stim in fmdl.stimulation:
        rhs = np.zeros(n_total)
        rhs[n_nodes:] = stim.stim_pattern
        rhs[gnd] = 0.0

        x = solve_fn(rhs)
        U = x[n_nodes:]  # electrode voltages
        meas = stim.meas_pattern @ U
        all_meas.append(meas)

    all_meas = np.concatenate(all_meas)
    return SolveData(meas=all_meas)


def compute_transfer_impedance(fmdl: ForwardModel, img: Image) -> np.ndarray:
    """
    Compute the full transfer impedance matrix.

    Z[i,j] = voltage at electrode j when 1A injected at electrode i,
             returned at electrode 0.
    """
    mesh = fmdl.mesh
    n_nodes = mesh.n_nodes
    n_elec = mesh.n_electrodes

    K, gnd = assemble_cem_system(mesh, img.elem_data)
    n_total = n_nodes + n_elec

    try:
        solve_fn = factorized(K.tocsc())
    except Exception:
        solve_fn = lambda b: spsolve(K, b)

    Z = np.zeros((n_elec, n_elec))
    for i in range(1, n_elec):
        rhs = np.zeros(n_total)
        rhs[n_nodes + i] = 1.0
        rhs[n_nodes + 0] = -1.0
        rhs[gnd] = 0.0

        x = solve_fn(rhs)
        Z[i, :] = x[n_nodes:]

    return Z


def compute_jacobian(img: Image) -> np.ndarray:
    """
    Compute the Jacobian (sensitivity matrix) using the adjoint method.

    J[m, e] = -v_m^T * (∂K/∂σ_e) * u = -vol[e] * (v·G)(u·G) per element
    """
    fmdl = img.fwd_model
    mesh = fmdl.mesh
    n_nodes = mesh.n_nodes
    n_elec = mesh.n_electrodes
    n_elem = mesh.n_elements
    n_total = n_nodes + n_elec

    K, gnd = assemble_cem_system(mesh, img.elem_data)

    try:
        solve_fn = factorized(K.tocsc())
    except Exception:
        solve_fn = lambda b: spsolve(K, b)

    grad_phi, vol = _compute_element_gradients(mesh.nodes, mesh.elements)

    # Forward solutions
    forward_u = []
    meas_info = []
    n_meas_total = 0

    for stim in fmdl.stimulation:
        rhs = np.zeros(n_total)
        rhs[n_nodes:] = stim.stim_pattern
        rhs[gnd] = 0.0
        x = solve_fn(rhs)
        forward_u.append(x[:n_nodes])
        n_m = stim.meas_pattern.shape[0]
        meas_info.append((n_meas_total, n_m, stim.meas_pattern))
        n_meas_total += n_m

    # Adjoint solutions
    adjoint_v = []
    for stim_idx, stim in enumerate(fmdl.stimulation):
        for m_row in stim.meas_pattern:
            rhs = np.zeros(n_total)
            rhs[n_nodes:] = m_row
            rhs[gnd] = 0.0
            v = solve_fn(rhs)
            adjoint_v.append(v[:n_nodes])

    # Compute Jacobian
    J = np.zeros((n_meas_total, n_elem))
    elems = mesh.elements

    m_idx = 0
    for stim_idx, stim in enumerate(fmdl.stimulation):
        u = forward_u[stim_idx]
        for mi in range(stim.meas_pattern.shape[0]):
            v = adjoint_v[m_idx]
            u_elem = u[elems]  # (M, 4)
            v_elem = v[elems]  # (M, 4)
            vG = np.einsum("ei,eid->ed", v_elem, grad_phi)
            uG = np.einsum("ei,eid->ed", u_elem, grad_phi)
            J[m_idx, :] = -vol * np.sum(vG * uG, axis=1)
            m_idx += 1

    return J


def build_adjacent_stim_patterns(
    n_per_ring: int,
    n_rings: int,
    current_A: float = 0.001,
) -> List[StimPattern]:
    """Build adjacent drive / adjacent measure stimulation patterns."""
    n_elec = n_per_ring * n_rings
    patterns = []

    for ring in range(n_rings):
        offset = ring * n_per_ring
        for i in range(n_per_ring):
            drv_p = offset + i
            drv_m = offset + (i + 1) % n_per_ring

            stim = np.zeros(n_elec)
            stim[drv_p] = current_A
            stim[drv_m] = -current_A

            meas_rows = []
            for mring in range(n_rings):
                m_off = mring * n_per_ring
                for mi in range(n_per_ring):
                    mp = m_off + mi
                    mn = m_off + (mi + 1) % n_per_ring
                    if mp in (drv_p, drv_m) or mn in (drv_p, drv_m):
                        continue
                    row = np.zeros(n_elec)
                    row[mp] = 1.0
                    row[mn] = -1.0
                    meas_rows.append(row)

            if meas_rows:
                patterns.append(StimPattern(
                    stim_pattern=stim,
                    meas_pattern=np.array(meas_rows),
                ))

    return patterns


def build_cross_ring_stim_patterns(
    n_per_ring: int,
    n_rings: int,
    current_A: float = 0.001,
) -> List[StimPattern]:
    """Build comprehensive cross-ring stimulation patterns."""
    n_elec = n_per_ring * n_rings
    half = n_per_ring // 2
    all_patterns = []

    # Part A: Within-ring opposite drive
    for ri in range(n_rings):
        off = ri * n_per_ring
        for ei in range(n_per_ring):
            dp = off + ei
            dm = off + (ei + half) % n_per_ring
            for rj in range(n_rings):
                m_off = rj * n_per_ring
                for mi in range(n_per_ring):
                    mp = m_off + mi
                    mn = m_off + (mi + 1) % n_per_ring
                    if mp in (dp, dm) or mn in (dp, dm):
                        continue
                    all_patterns.append([dp, dm, mp, mn])

    # Part B: Cross-ring adjacent
    for ri in range(n_rings - 1):
        off1 = ri * n_per_ring
        off2 = (ri + 1) * n_per_ring
        for ei in range(n_per_ring):
            dp = off1 + ei
            dm = off2 + ei
            for rj in range(n_rings):
                m_off = rj * n_per_ring
                for mi in range(n_per_ring):
                    mp = m_off + mi
                    mn = m_off + (mi + 1) % n_per_ring
                    if mp in (dp, dm) or mn in (dp, dm):
                        continue
                    all_patterns.append([dp, dm, mp, mn])

    # Part C: Long-range cross-ring
    if n_rings >= 3:
        off1 = 0
        offN = (n_rings - 1) * n_per_ring
        for ei in range(0, n_per_ring, max(1, n_per_ring // 4)):
            dp = off1 + ei
            dm = offN + ei
            for rj in range(n_rings):
                m_off = rj * n_per_ring
                for mi in range(n_per_ring):
                    mp = m_off + mi
                    mn = m_off + (mi + 1) % n_per_ring
                    if mp in (dp, dm) or mn in (dp, dm):
                        continue
                    all_patterns.append([dp, dm, mp, mn])

    return _patterns_list_to_stim(all_patterns, n_elec, current_A)


def _patterns_list_to_stim(patterns, n_elec, current_A):
    """Convert [drv+, drv-, meas+, meas-] list to StimPattern objects."""
    from collections import defaultdict
    drive_groups = defaultdict(list)
    for dp, dm, mp, mn in patterns:
        drive_groups[(dp, dm)].append((mp, mn))

    stim_list = []
    for (dp, dm), meas_pairs in drive_groups.items():
        stim = np.zeros(n_elec)
        stim[dp] = current_A
        stim[dm] = -current_A

        meas_rows = []
        seen = set()
        for mp, mn in meas_pairs:
            key = (mp, mn)
            if key not in seen:
                seen.add(key)
                row = np.zeros(n_elec)
                row[mp] = 1.0
                row[mn] = -1.0
                meas_rows.append(row)

        if meas_rows:
            stim_list.append(StimPattern(
                stim_pattern=stim,
                meas_pattern=np.array(meas_rows),
            ))

    return stim_list
