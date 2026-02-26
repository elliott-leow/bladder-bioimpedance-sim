"""
Structured tetrahedral mesh generation for ellipsoidal torso.

Creates a body-fitted mesh of an ellipsoidal cylinder with electrode
surface identification. Uses pure numpy -- no external mesh generator.

Approach:
    1. Regular Cartesian grid clipped to ellipsoidal cross-section
    2. Each hex cell split into 5 tetrahedra
    3. Orphan nodes removed; boundary faces extracted
    4. Electrode surfaces identified by proximity
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Electrode:
    """A surface electrode for the Complete Electrode Model."""
    nodes: np.ndarray          # indices of mesh nodes under this electrode
    faces: np.ndarray          # (F, 3) boundary face connectivity
    z_contact: float           # contact impedance (Ohm*cm^2)
    center: np.ndarray         # (3,) center position on surface
    area: float = 0.0         # total electrode area (cm^2)


@dataclass
class TorsoMesh:
    """Tetrahedral mesh of the ellipsoidal torso."""
    nodes: np.ndarray          # (N, 3) node coordinates in cm
    elements: np.ndarray       # (M, 4) tetrahedral connectivity (0-indexed)
    boundary_faces: np.ndarray  # (F, 3) boundary triangle connectivity
    boundary_normals: np.ndarray  # (F, 3) outward normals
    electrodes: List[Electrode] = field(default_factory=list)

    @property
    def n_nodes(self) -> int:
        return self.nodes.shape[0]

    @property
    def n_elements(self) -> int:
        return self.elements.shape[0]

    @property
    def n_electrodes(self) -> int:
        return len(self.electrodes)

    def element_centroids(self) -> np.ndarray:
        """Compute centroids of all elements. Returns (M, 3)."""
        return self.nodes[self.elements].mean(axis=1)

    def element_volumes(self) -> np.ndarray:
        """Compute volumes of all tetrahedra. Returns (M,)."""
        p = self.nodes[self.elements]  # (M, 4, 3)
        d = p[:, 1:, :] - p[:, 0:1, :]  # (M, 3, 3)
        return np.abs(np.linalg.det(d)) / 6.0

    def face_areas(self, faces: np.ndarray) -> np.ndarray:
        """Compute areas of triangular faces."""
        p = self.nodes[faces]  # (F, 3, 3)
        v1 = p[:, 1] - p[:, 0]
        v2 = p[:, 2] - p[:, 0]
        return 0.5 * np.linalg.norm(np.cross(v1, v2), axis=1)


def create_torso_mesh(
    rx: float = 15.0,
    ry: float = 10.0,
    height: float = 20.0,
    max_edge: float = 1.0,
    electrode_positions: Optional[np.ndarray] = None,
    electrode_radius: float = 0.5,
    z_contact: float = 5.0,
) -> TorsoMesh:
    """
    Create a structured tetrahedral mesh of an ellipsoidal cylinder.

    Parameters
    ----------
    rx, ry : float
        Semi-axes of the elliptical cross-section (cm).
    height : float
        Total height of the torso model (cm).
    max_edge : float
        Maximum element edge length (cm). Controls mesh density.
    electrode_positions : np.ndarray, optional
        (L, 3) positions of electrode centers on the surface.
    electrode_radius : float
        Radius of each circular electrode contact (cm).
    z_contact : float
        Default electrode-skin contact impedance (Ohm*cm^2).

    Returns
    -------
    TorsoMesh
    """
    h = max_edge
    nx = int(np.ceil(2 * rx / h)) + 1
    ny = int(np.ceil(2 * ry / h)) + 1
    nz = int(np.ceil(height / h)) + 1

    x = np.linspace(-rx, rx, nx)
    y = np.linspace(-ry, ry, ny)
    z = np.linspace(0, height, nz)

    # Generate all grid points
    X, Y, Z_grid = np.meshgrid(x, y, z, indexing="ij")
    grid_points = np.column_stack([X.ravel(), Y.ravel(), Z_grid.ravel()])

    # Grid point inside check
    n_grid = nx * ny * nz
    r_ellipse = np.sqrt(
        (grid_points[:, 0] / rx) ** 2 + (grid_points[:, 1] / ry) ** 2
    )
    inside = r_ellipse <= 1.001

    grid_to_node = -np.ones(n_grid, dtype=np.int64)
    kept = np.where(inside)[0]
    grid_to_node[kept] = np.arange(len(kept))
    all_nodes = grid_points[inside]

    # Build tetrahedra from hex cells.
    # 5-tet decomposition of a cube (consistent across shared faces):
    #   Corners: 0=(0,0,0), 1=(1,0,0), 2=(0,1,0), 3=(1,1,0),
    #            4=(0,0,1), 5=(1,0,1), 6=(0,1,1), 7=(1,1,1)
    tet5 = np.array([
        [0, 1, 3, 5],
        [0, 2, 3, 6],
        [0, 4, 5, 6],
        [3, 5, 6, 7],
        [0, 3, 5, 6],  # central tet
    ])

    elements_list = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            for k in range(nz - 1):
                # 8 corners of hex cell (i,j,k)
                c = np.array([
                    (i + di) * ny * nz + (j + dj) * nz + (k + dk)
                    for di, dj, dk in [
                        (0,0,0), (1,0,0), (0,1,0), (1,1,0),
                        (0,0,1), (1,0,1), (0,1,1), (1,1,1)
                    ]
                ])
                nids = grid_to_node[c]
                if np.any(nids < 0):
                    continue
                for t in tet5:
                    elements_list.append(nids[t])

    if not elements_list:
        raise RuntimeError("No elements created. Check mesh parameters.")

    elements = np.array(elements_list, dtype=np.int64)

    # Fix orientation: ensure positive volume for all tets
    p = all_nodes[elements]
    d = p[:, 1:, :] - p[:, 0:1, :]
    det_d = np.linalg.det(d)
    neg = det_d < 0
    elements[neg, 1], elements[neg, 2] = elements[neg, 2].copy(), elements[neg, 1].copy()

    # Remove degenerate tets
    vols = np.abs(det_d)
    valid = vols > 1e-14
    elements = elements[valid]

    # CRITICAL: Remove orphan nodes (nodes not in any element)
    used_nodes = np.unique(elements.ravel())
    old_to_new = -np.ones(len(all_nodes), dtype=np.int64)
    old_to_new[used_nodes] = np.arange(len(used_nodes))
    nodes = all_nodes[used_nodes]
    elements = old_to_new[elements]

    # Update electrode positions mapping if needed
    # (electrode positions are in world coords, so no mapping needed)

    # Extract boundary faces
    boundary_faces, boundary_normals = _extract_boundary(nodes, elements, rx, ry)

    mesh = TorsoMesh(
        nodes=nodes,
        elements=elements,
        boundary_faces=boundary_faces,
        boundary_normals=boundary_normals,
    )

    # Place electrodes
    if electrode_positions is not None:
        mesh.electrodes = _place_electrodes(
            mesh, electrode_positions, electrode_radius, z_contact
        )

    return mesh


def compute_electrode_positions(
    rx: float,
    ry: float,
    n_per_ring: int,
    ring_z: np.ndarray,
) -> np.ndarray:
    """Compute 3D positions of electrodes on the ellipsoidal surface."""
    positions = []
    for z_val in ring_z:
        for i in range(n_per_ring):
            theta = 2 * np.pi * i / n_per_ring
            positions.append([rx * np.cos(theta), ry * np.sin(theta), z_val])
    return np.array(positions)


def _extract_boundary(nodes, elements, rx, ry):
    """Extract boundary faces (faces belonging to exactly one tet)."""
    # Each tet has 4 triangular faces
    face_map = np.array([[1,2,3], [0,2,3], [0,1,3], [0,1,2]])

    n_tet = elements.shape[0]

    # Collect all faces as sorted tuples for counting
    from collections import Counter
    face_counter = Counter()
    face_original = {}  # sorted_tuple -> original face

    for fi in range(4):
        faces_fi = elements[:, face_map[fi]]
        faces_sorted = np.sort(faces_fi, axis=1)
        for idx in range(n_tet):
            key = tuple(faces_sorted[idx])
            face_counter[key] += 1
            if key not in face_original:
                face_original[key] = faces_fi[idx]

    # Boundary = faces appearing exactly once
    boundary_faces = np.array(
        [face_original[k] for k, v in face_counter.items() if v == 1],
        dtype=np.int64
    )

    # Compute outward normals
    if len(boundary_faces) == 0:
        return boundary_faces, np.zeros((0, 3))

    p = nodes[boundary_faces]
    v1 = p[:, 1] - p[:, 0]
    v2 = p[:, 2] - p[:, 0]
    normals = np.cross(v1, v2)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    normals = normals / norms

    # Orient outward: face centroid should move "away from center"
    centroids = p.mean(axis=1)
    # For lateral faces, radial direction is outward
    radial = centroids.copy()
    radial[:, 2] = 0
    dot = np.sum(normals * radial, axis=1)
    r_norm = np.sqrt((centroids[:, 0] / rx)**2 + (centroids[:, 1] / ry)**2)
    is_lateral = r_norm > 0.85

    flip = is_lateral & (dot < 0)
    normals[flip] *= -1
    boundary_faces[flip, 1], boundary_faces[flip, 2] = (
        boundary_faces[flip, 2].copy(), boundary_faces[flip, 1].copy()
    )

    # Top/bottom caps
    is_top = centroids[:, 2] > 0.9 * nodes[:, 2].max()
    normals[is_top & (normals[:, 2] < 0)] *= -1
    is_bot = centroids[:, 2] < 0.1 * nodes[:, 2].max() + nodes[:, 2].min()
    normals[is_bot & (normals[:, 2] > 0)] *= -1

    return boundary_faces, normals


def _place_electrodes(
    mesh: TorsoMesh,
    positions: np.ndarray,
    radius: float,
    z_contact: float,
) -> List[Electrode]:
    """Assign boundary faces and nodes to electrodes."""
    electrodes = []
    face_centroids = mesh.nodes[mesh.boundary_faces].mean(axis=1)

    for pos in positions:
        dist = np.linalg.norm(face_centroids - pos, axis=1)
        face_mask = dist < radius * 1.5  # slight expansion for coarse meshes

        if np.sum(face_mask) < 2:
            # Use nearest N faces
            n_min = max(3, int(np.pi * radius**2 / (mesh.element_volumes().mean()**(2/3))))
            n_min = min(n_min, len(dist))
            threshold = np.sort(dist)[n_min - 1]
            face_mask = dist <= threshold + 1e-10

        elec_faces = mesh.boundary_faces[face_mask]
        elec_nodes = np.unique(elec_faces.ravel())
        area = mesh.face_areas(elec_faces).sum()

        electrodes.append(Electrode(
            nodes=elec_nodes,
            faces=elec_faces,
            z_contact=z_contact,
            center=pos.copy(),
            area=max(area, 1e-6),
        ))

    return electrodes
