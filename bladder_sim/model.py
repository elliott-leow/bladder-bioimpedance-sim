"""
Anatomical pelvis model for bladder bioimpedance simulation.

Builds a 3D FEM model of the male pelvis with 15 tissue types:
    Outer shells: skin, subcutaneous fat, skeletal muscle
    Pelvic bone: iliac wings (L/R), sacrum/spine + sacral alae,
                 pubic symphysis, posterior pelvic ring
    Soft organs: bowel, rectum, iliac vessels, peritoneal fluid, prostate
    Bladder: detrusor wall (volume-dependent thickness), urine
    Background: connective tissue / fascia

All geometry in cm. Conductivities from Gabriel et al. 1996 + IT'IS v4.1.

Anatomical sources:
    Pelvimetry: StatPearls (pelvic inlet/outlet dimensions)
    Bladder wall: Oelke et al. 2006, Hakenberg et al. 2000 (sonographic BWT)
    Prostate: StatPearls, Kenhub (dimensions, position)
    Bone: sacral morphometric studies (ala dimensions)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from .mesh import TorsoMesh, create_torso_mesh, compute_electrode_positions
from .fem import (
    ForwardModel,
    Image,
    StimPattern,
    build_cross_ring_stim_patterns,
    build_adjacent_stim_patterns,
)
from .tissue_properties import get_conductivity, get_contact_impedance, get_complex_admittivity


# =====================================================================
# Default geometry (cm)
# =====================================================================
# The body core (skeleton + muscle) has fixed dimensions independent of BMI.
# Fat and skin are added OUTSIDE the core, so the torso surface grows with BMI.
#
# At the default fat thickness (1.5 cm), these yield the original
# TORSO_RX=15.0, TORSO_RY=10.0 for backward compatibility.
BODY_CORE_RX = 13.3   # lateral half-width of muscle outer boundary
BODY_CORE_RY = 8.3    # AP half-depth of muscle outer boundary
TORSO_H = 20.0        # total height (fixed)

SKIN_THICK = 0.2
FAT_THICK = 1.5       # default subcutaneous fat thickness
MUSCLE_THICK = 1.5


def torso_dimensions(fat_thick: float = FAT_THICK):
    """Compute outer torso radii from fat thickness.

    The torso surface = body core + muscle layer is fixed, then fat and
    skin are added on top. More fat → larger torso.

    Returns (rx, ry) in cm.
    """
    rx = BODY_CORE_RX + fat_thick + SKIN_THICK
    ry = BODY_CORE_RY + fat_thick + SKIN_THICK
    return rx, ry


# Default torso radii at FAT_THICK=1.5 (backward compatible: 15.0, 10.0)
TORSO_RX, TORSO_RY = torso_dimensions(FAT_THICK)

# Bladder geometry — volume-dependent anisotropic expansion
# Based on Glass Clark & Nagle et al. 2020 (PMC6538469), Nagle et al. 2018.
# The bladder expands preferentially in the craniocaudal (S-I) direction;
# at low volume it is oblate, becoming near-spherical when full.
# Reference semi-axes at 300 mL [lateral, AP, SI] in cm:
BLADDER_ASPECT_REF = np.array([5.0, 4.0, 5.3])
# Volume at which the reference aspect ratio applies:
BLADDER_REF_VOLUME = 300.0
BLADDER_CENTER_Y_EMPTY = 3.0   # Y at low volume (anterior, behind pubic symphysis)
BLADDER_CENTER_Y_FULL = 2.5    # Y at high volume (small posterior shift ~0.5 cm)
BLADDER_BASE_Z = 3.0           # base height above model origin (trigone, fixed)

# Wall thickness decreases as bladder distends (detrusor muscle stretching).
# Nonlinear: rapid decrease <250 mL, then plateau (Oelke et al. 2006,
# Ugwu et al. 2019).  Modeled with geometric surface-area scaling + floor.
BLADDER_WALL_THICK_REF = 0.28  # cm, BWT at 50 mL reference
BLADDER_WALL_REF_VOLUME = 50.0
BLADDER_WALL_THICK_MIN = 0.13  # cm, plateau DWT (Oelke 2006: men 1.4 mm)

# Default electrode configuration
DEFAULT_N_PER_RING = 16
DEFAULT_RING_Z = np.array([4.0, 6.0, 8.0, 10.0])
DEFAULT_ELEC_RADIUS = 0.5
DEFAULT_MAX_EDGE = 1.0
DEFAULT_CURRENT_A = 0.001

# Prostate geometry (male model)
PROSTATE_CENTER = np.array([0.0, 3.5, 2.0])  # inferior to bladder base, anterior
PROSTATE_SEMI = np.array([2.0, 1.5, 1.25])   # 4.0 x 3.0 x 2.5 cm total


def bladder_wall_thickness(volume_mL: float) -> float:
    """Volume-dependent bladder wall thickness in cm.

    Uses geometric surface-area scaling with an asymptotic minimum,
    matching the nonlinear pattern from sonographic data:
      - Rapid decrease <250 mL (rugae unfolding + stretching)
      - Plateau at ~1.3 mm for men (Oelke et al. 2006, Ugwu et al. 2019)

    BWT ≈ t_ref * (V_ref / V)^(2/3), floored at t_min.
    """
    V = max(volume_mL, BLADDER_WALL_REF_VOLUME)
    t_geometric = BLADDER_WALL_THICK_REF * (BLADDER_WALL_REF_VOLUME / V) ** (2.0 / 3.0)
    return max(t_geometric, BLADDER_WALL_THICK_MIN)


def bladder_semi_axes(volume_mL: float) -> Tuple[float, float, float]:
    """Volume-dependent bladder semi-axes (a_lat, b_ap, c_si) in cm.

    The bladder expands anisotropically (Glass Clark & Nagle et al. 2020):
      - At low volumes: oblate (wider than tall)
      - At high volumes: near-spherical (H:W ≈ 1.06)

    The S-I axis grows faster than lateral/AP axes.  Implemented by
    interpolating the SI:lateral ratio from ~0.85 (50 mL, oblate) to
    ~1.06 (500 mL, near-spherical), per ultrasound strain data.
    """
    # Compute isotropic scale factor from reference volume
    V_ref = (4.0 / 3.0) * np.pi * np.prod(BLADDER_ASPECT_REF)
    k = (volume_mL / V_ref) ** (1.0 / 3.0)

    # Volume-dependent SI:lateral ratio
    # At 50 mL  → 0.85 (oblate)
    # At 300 mL → 1.06 (reference, Glass Clark 2020)
    # At 500 mL → 1.06 (plateau)
    frac = np.clip((volume_mL - 50.0) / 250.0, 0.0, 1.0)
    si_lat_ratio = 0.85 + (1.06 - 0.85) * frac  # 0.85 → 1.06

    # Apply anisotropic correction: boost SI, reduce lateral to conserve volume
    # ratio = c_si / a_lat  →  c_si *= correction, a_lat *= 1/sqrt(correction)
    ref_ratio = BLADDER_ASPECT_REF[2] / BLADDER_ASPECT_REF[0]  # 5.3/5.0 = 1.06
    correction = si_lat_ratio / ref_ratio  # adjust relative to reference
    # Volume conservation: a * b * c = const → if c *= f, then a,b *= 1/sqrt(f)
    lat_corr = 1.0 / np.sqrt(correction)

    a = BLADDER_ASPECT_REF[0] * k * lat_corr
    b = BLADDER_ASPECT_REF[1] * k * lat_corr
    c = BLADDER_ASPECT_REF[2] * k * correction

    return a, b, c


def bladder_center_y(volume_mL: float) -> float:
    """Volume-dependent bladder center Y position in cm.

    The bladder base (trigone) is fixed at the pelvic floor.  At low
    volumes the bladder sits anteriorly, nestled behind the pubic
    symphysis.  As it fills, the dome expands posteriorly (the anterior
    wall is constrained by the pubis) and superiorly, so the geometric
    centre shifts backward.

    Returns the Y coordinate of the bladder center (cm).
    """
    frac = np.clip((volume_mL - 50.0) / 450.0, 0.0, 1.0)
    return BLADDER_CENTER_Y_EMPTY + (BLADDER_CENTER_Y_FULL - BLADDER_CENTER_Y_EMPTY) * frac


def build_pelvis_model(
    bladder_volume_mL: float = 300.0,
    mesh: Optional[TorsoMesh] = None,
    freq_kHz: float = 50.0,
    n_per_ring: int = DEFAULT_N_PER_RING,
    ring_z: Optional[np.ndarray] = None,
    max_edge: float = DEFAULT_MAX_EDGE,
    elec_radius: float = DEFAULT_ELEC_RADIUS,
    stim_pattern: str = "cross_ring",
    current_A: float = DEFAULT_CURRENT_A,
    fat_thick: Optional[float] = None,
    use_complex: bool = False,
) -> Tuple[ForwardModel, Image]:
    """
    Build a 3D pelvis model with multi-ring electrode array.

    Parameters
    ----------
    bladder_volume_mL : float
        Bladder fill volume in mL.
    mesh : TorsoMesh, optional
        Pre-built mesh to reuse (skips meshing).
    freq_kHz : float
        Excitation frequency in kHz. Conductivities are interpolated
        from Gabriel et al. 1996 parametric models.
    n_per_ring : int
        Electrodes per ring.
    ring_z : array-like
        Z-positions of each electrode ring (cm).
    max_edge : float
        Maximum mesh element edge length (cm).
    elec_radius : float
        Electrode contact radius (cm).
    stim_pattern : str
        'cross_ring', 'adjacent', or 'none'.
    current_A : float
        Stimulation current amplitude (A).
    fat_thick : float, optional
        Subcutaneous fat layer thickness in cm. Defaults to module-level
        FAT_THICK (1.5 cm). This is the dominant source of inter-subject
        variability in bioimpedance measurements (Leonhardt et al. 2025).
    use_complex : bool
        If True, assign complex admittivity (sigma + j*omega*eps_0*eps_r)
        instead of real conductivity.  This enables impedance phase
        analysis.

    Returns
    -------
    fmdl : ForwardModel
    img : Image
    """
    if fat_thick is None:
        fat_thick = FAT_THICK
    if ring_z is None:
        ring_z = DEFAULT_RING_Z.copy()
    ring_z = np.asarray(ring_z)
    n_rings = len(ring_z)
    n_elec = n_per_ring * n_rings

    # Compute BMI-dependent torso outer dimensions
    torso_rx, torso_ry = torso_dimensions(fat_thick)

    # --- Build or reuse mesh ---
    mesh_was_created = mesh is None
    if mesh is None:
        print(f"=== Multi-Ring Electrode Array Configuration ===")
        print(f"  Rings:            {n_rings} at z = {ring_z} cm")
        print(f"  Electrodes/ring:  {n_per_ring}")
        print(f"  Total electrodes: {n_elec}")
        print(f"  Electrode radius: {elec_radius:.2f} cm")
        print(f"  Max element edge: {max_edge:.2f} cm")
        print(f"  Torso radii:      {torso_rx:.1f} x {torso_ry:.1f} cm (fat={fat_thick:.1f} cm)")

        # Compute electrode positions on the ellipsoidal surface
        elec_pos = compute_electrode_positions(torso_rx, torso_ry, n_per_ring, ring_z)

        z_c = get_contact_impedance(freq_kHz)
        mesh = create_torso_mesh(
            rx=torso_rx,
            ry=torso_ry,
            height=TORSO_H,
            max_edge=max_edge,
            electrode_positions=elec_pos,
            electrode_radius=elec_radius,
            z_contact=z_c,
        )
        print(f"  Mesh: {mesh.n_elements} elements, {mesh.n_nodes} nodes")
        print(f"  Electrodes placed: {mesh.n_electrodes}")
    else:
        # Update contact impedance for this frequency
        z_c = get_contact_impedance(freq_kHz)
        for elec in mesh.electrodes:
            elec.z_contact = z_c

    # --- Build forward model ---
    fmdl = ForwardModel(mesh=mesh)

    # Build stimulation patterns
    if stim_pattern == "cross_ring":
        fmdl.stimulation = build_cross_ring_stim_patterns(
            n_per_ring, n_rings, current_A
        )
    elif stim_pattern == "adjacent":
        fmdl.stimulation = build_adjacent_stim_patterns(
            n_per_ring, n_rings, current_A
        )
    # else: no stimulation (user will set manually)

    # --- Assign conductivities ---
    # Only print verbose summary on first build (when mesh is created, not reused)
    img = _assign_conductivities(fmdl, bladder_volume_mL, freq_kHz,
                                 fat_thick=fat_thick,
                                 verbose=(mesh_was_created),
                                 use_complex=use_complex)

    return fmdl, img


def _assign_conductivities(
    fmdl: ForwardModel, bladder_volume_mL: float, freq_kHz: float,
    fat_thick: float = FAT_THICK, verbose: bool = True,
    use_complex: bool = False,
) -> Image:
    """
    Assign frequency-dependent tissue conductivities to mesh elements.

    Priority (last wins for overlapping regions):
        1. Background
        2. Concentric shells (skin, fat, muscle)
        3. Pelvic bone
        4. Soft organs (bowel, rectum, vessels, peritoneal)
        5. Bladder (wall + urine) — highest priority

    If use_complex is True, assigns complex admittivity
    (sigma + j*omega*eps_0*eps_r) instead of real conductivity.
    """
    mesh = fmdl.mesh
    ec = mesh.element_centroids()
    cx, cy, cz = ec[:, 0], ec[:, 1], ec[:, 2]
    n_elem = mesh.n_elements

    if use_complex:
        sigma = get_complex_admittivity
        dtype = complex
    else:
        sigma = get_conductivity
        dtype = float

    # Initialize with background
    elem_data = np.full(n_elem, sigma("background", freq_kHz), dtype=dtype)

    # ==================== CONCENTRIC SHELLS ====================
    # Outer surface grows with fat; internal body core is fixed.
    torso_rx, torso_ry = torso_dimensions(fat_thick)
    inner_skin_rx = torso_rx - SKIN_THICK
    inner_skin_ry = torso_ry - SKIN_THICK
    inner_fat_rx = inner_skin_rx - fat_thick
    inner_fat_ry = inner_skin_ry - fat_thick
    # inner_fat should equal BODY_CORE_RX, BODY_CORE_RY
    inner_muscle_rx = inner_fat_rx - MUSCLE_THICK
    inner_muscle_ry = inner_fat_ry - MUSCLE_THICK

    r_skin = np.sqrt((cx / inner_skin_rx) ** 2 + (cy / inner_skin_ry) ** 2)
    r_fat = np.sqrt((cx / inner_fat_rx) ** 2 + (cy / inner_fat_ry) ** 2)
    r_muscle = np.sqrt((cx / inner_muscle_rx) ** 2 + (cy / inner_muscle_ry) ** 2)

    is_skin = r_skin > 1.0
    is_fat = (r_fat > 1.0) & ~is_skin
    is_muscle = (r_muscle > 1.0) & ~is_skin & ~is_fat

    elem_data[is_muscle] = sigma("muscle", freq_kHz)
    elem_data[is_fat] = sigma("fat", freq_kHz)
    elem_data[is_skin] = sigma("skin", freq_kHz)

    # ==================== PELVIC BONE ====================
    # Left iliac wing
    r_iliac_L = np.sqrt(
        ((cx - (-8.0)) / 5.0) ** 2
        + ((cy - (-2.0)) / 3.5) ** 2
        + ((cz - 7.0) / 5.0) ** 2
    )
    # Right iliac wing
    r_iliac_R = np.sqrt(
        ((cx - 8.0) / 5.0) ** 2
        + ((cy - (-2.0)) / 3.5) ** 2
        + ((cz - 7.0) / 5.0) ** 2
    )
    # Sacrum / spine
    r_spine = np.sqrt(
        ((cx - 0.0) / 2.5) ** 2
        + ((cy - (-7.5)) / 2.0) ** 2
        + ((cz - 5.0) / 5.0) ** 2
    )
    # Pubic symphysis
    r_pubis = np.sqrt(
        ((cx - 0.0) / 1.5) ** 2
        + ((cy - 7.5) / 1.25) ** 2
        + ((cz - 2.5) / 2.0) ** 2
    )
    # Posterior pelvic ring (shell)
    pring_rx, pring_ry, pring_rz = 7.0, 5.5, 3.0
    pring_cz = 3.0
    pring_thick = 1.2
    avg_pring = (pring_rx + pring_ry) / 2.0
    inner_frac = 1.0 - pring_thick / avg_pring
    r_pring_outer = np.sqrt(
        (cx / pring_rx) ** 2
        + (cy / pring_ry) ** 2
        + ((cz - pring_cz) / pring_rz) ** 2
    )
    r_pring_inner = np.sqrt(
        (cx / (pring_rx * inner_frac)) ** 2
        + (cy / (pring_ry * inner_frac)) ** 2
        + ((cz - pring_cz) / pring_rz) ** 2
    )
    is_pelvic_ring = (r_pring_outer < 1) & (r_pring_inner > 1) & (cy < -1.0)

    # Sacral alae (lateral wings connecting sacrum to iliac bones)
    # Sacral width at S1 ~10 cm; each ala extends ~4 cm from midline
    r_ala_L = np.sqrt(
        ((cx - (-4.0)) / 3.0) ** 2
        + ((cy - (-6.0)) / 1.5) ** 2
        + ((cz - 4.0) / 2.5) ** 2
    )
    r_ala_R = np.sqrt(
        ((cx - 4.0) / 3.0) ** 2
        + ((cy - (-6.0)) / 1.5) ** 2
        + ((cz - 4.0) / 2.5) ** 2
    )

    is_bone = (
        (r_iliac_L < 1)
        | (r_iliac_R < 1)
        | (r_spine < 1)
        | (r_pubis < 1)
        | is_pelvic_ring
        | (r_ala_L < 1)
        | (r_ala_R < 1)
    )
    elem_data[is_bone] = sigma("bone_avg", freq_kHz)

    # ==================== SOFT ORGANS ====================
    # Bowel / intestines
    r_bowel = np.sqrt(
        ((cx - 0.0) / 8.0) ** 2
        + ((cy - 1.5) / 5.0) ** 2
        + ((cz - 12.0) / 5.0) ** 2
    )
    is_bowel = (r_bowel < 1) & ~is_bone & ~is_skin & ~is_fat
    elem_data[is_bowel] = sigma("bowel_eff", freq_kHz)

    # Rectum
    r_rectum = np.sqrt(
        ((cx - 0.0) / 1.5) ** 2
        + ((cy - (-3.0)) / 1.5) ** 2
        + ((cz - 4.0) / 4.0) ** 2
    )
    is_rectum = (r_rectum < 1) & ~is_bone
    elem_data[is_rectum] = sigma("rectum", freq_kHz)

    # Iliac blood vessels
    r_vessel_L = np.sqrt(
        ((cx - (-5.0)) / 1.0) ** 2
        + ((cy - 0.0) / 1.0) ** 2
        + ((cz - 5.0) / 4.0) ** 2
    )
    r_vessel_R = np.sqrt(
        ((cx - 5.0) / 1.0) ** 2
        + ((cy - 0.0) / 1.0) ** 2
        + ((cz - 5.0) / 4.0) ** 2
    )
    is_vessels = ((r_vessel_L < 1) | (r_vessel_R < 1)) & ~is_bone
    elem_data[is_vessels] = sigma("blood", freq_kHz)

    # Peritoneal fluid
    r_perit = np.sqrt(
        ((cx - 0.0) / 3.0) ** 2
        + ((cy - (-0.5)) / 1.0) ** 2
        + ((cz - 3.5) / 2.0) ** 2
    )
    is_peritoneal = (r_perit < 1) & ~is_bone & ~(r_rectum < 1)
    elem_data[is_peritoneal] = sigma("peritoneal", freq_kHz)

    # ==================== PROSTATE (male, inferior to bladder) ===========
    # ~4.0 x 3.0 x 2.5 cm, wraps proximal urethra below bladder neck
    # Position: inferior to bladder base, behind pubic symphysis
    r_prostate = np.sqrt(
        ((cx - PROSTATE_CENTER[0]) / PROSTATE_SEMI[0]) ** 2
        + ((cy - PROSTATE_CENTER[1]) / PROSTATE_SEMI[1]) ** 2
        + ((cz - PROSTATE_CENTER[2]) / PROSTATE_SEMI[2]) ** 2
    )
    is_prostate = (r_prostate < 1) & ~is_bone
    elem_data[is_prostate] = sigma("prostate", freq_kHz)

    # ==================== BLADDER (highest priority) ====================
    bl_a, bl_b, bl_c = bladder_semi_axes(bladder_volume_mL)
    bl_cy = bladder_center_y(bladder_volume_mL)
    bl_cz = BLADDER_BASE_Z + bl_c

    bx = cx / bl_a
    by = (cy - bl_cy) / bl_b
    bz = (cz - bl_cz) / bl_c
    r_bl = np.sqrt(bx ** 2 + by ** 2 + bz ** 2)
    avg_bl = (bl_a + bl_b + bl_c) / 3.0

    wall_thick = bladder_wall_thickness(bladder_volume_mL)
    is_urine = r_bl < (1.0 - wall_thick / avg_bl)
    is_wall = (r_bl < 1.0) & ~is_urine

    elem_data[is_wall] = sigma("bladder_wall", freq_kHz)
    elem_data[is_urine] = sigma("urine", freq_kHz)

    # --- Diagnostic output ---
    if verbose:
        n_total = n_elem
        n_bl = np.sum(is_urine | is_wall)
        print(f"\n=== Anatomical Model Summary (freq = {freq_kHz:.0f} kHz) ===")
        print(
            f"Bladder: {bladder_volume_mL:.0f} mL -> "
            f"{bl_a*2:.1f} x {bl_b*2:.1f} x {bl_c*2:.1f} cm, "
            f"wall = {wall_thick*10:.1f} mm, "
            f"{n_bl} elems ({100*n_bl/n_total:.1f}%), {np.sum(is_urine)} urine"
        )
        print(f"Fat layer: {fat_thick:.1f} cm")
        for name, mask, tissue in [
            ("Skin", is_skin, "skin"),
            ("Fat", is_fat, "fat"),
            ("Muscle", is_muscle, "muscle"),
            ("Bone", is_bone, "bone_avg"),
            ("Bowel", is_bowel, "bowel_eff"),
            ("Rectum", is_rectum, "rectum"),
            ("Vessels", is_vessels, "blood"),
            ("Perit.", is_peritoneal, "peritoneal"),
            ("Prostate", is_prostate, "prostate"),
            ("Bl.wall", is_wall, "bladder_wall"),
            ("Urine", is_urine, "urine"),
        ]:
            print(
                f"  {name:8s}: {np.sum(mask):6d} elems  "
                f"(sigma = {sigma(tissue, freq_kHz):.4f} S/m)"
            )

        n_assigned = np.sum(
            is_skin | is_fat | is_muscle | is_bone | is_bowel
            | is_rectum | is_vessels | is_peritoneal | is_prostate
            | is_wall | is_urine
        )
        print(
            f"  Backgnd:  {n_total - n_assigned:6d} elems  "
            f"(sigma = {sigma('background', freq_kHz):.4f} S/m)"
        )
        z_c = get_contact_impedance(freq_kHz)
        print(f"  z_contact: {z_c:.2f} Ohm*cm^2 at {freq_kHz:.0f} kHz")
        print(f"  Total: {n_total} elements\n")

    return Image(fwd_model=fmdl, elem_data=elem_data)


def get_bladder_mask(
    mesh: TorsoMesh, bladder_volume_mL: float = 300.0
) -> np.ndarray:
    """
    Get logical mask of URINE elements (excluding wall).

    Parameters
    ----------
    mesh : TorsoMesh
    bladder_volume_mL : float

    Returns
    -------
    np.ndarray
        (n_elem,) boolean mask.
    """
    ec = mesh.element_centroids()
    cx, cy, cz = ec[:, 0], ec[:, 1], ec[:, 2]

    bl_a, bl_b, bl_c = bladder_semi_axes(bladder_volume_mL)
    bl_cy = bladder_center_y(bladder_volume_mL)
    bl_cz = BLADDER_BASE_Z + bl_c

    bx = cx / bl_a
    by = (cy - bl_cy) / bl_b
    bz = (cz - bl_cz) / bl_c
    r_bl = np.sqrt(bx ** 2 + by ** 2 + bz ** 2)
    avg_bl = (bl_a + bl_b + bl_c) / 3.0

    wall_thick = bladder_wall_thickness(bladder_volume_mL)
    return r_bl < (1.0 - wall_thick / avg_bl)


def get_tissue_labels(
    mesh: TorsoMesh, bladder_volume_mL: float = 300.0,
    fat_thick: Optional[float] = None,
) -> np.ndarray:
    """
    Assign a tissue label index to each element.

    Returns
    -------
    np.ndarray
        (n_elem,) integer labels:
        0=background, 1=skin, 2=fat, 3=muscle, 4=bone,
        5=bowel, 6=rectum, 7=vessels, 8=peritoneal,
        9=bladder_wall, 10=urine
    """
    if fat_thick is None:
        fat_thick = FAT_THICK
    ec = mesh.element_centroids()
    cx, cy, cz = ec[:, 0], ec[:, 1], ec[:, 2]
    n_elem = mesh.n_elements

    labels = np.zeros(n_elem, dtype=np.int32)  # 0 = background

    # Shells — outer surface grows with fat, core is fixed
    torso_rx, torso_ry = torso_dimensions(fat_thick)
    inner_skin_rx = torso_rx - SKIN_THICK
    inner_skin_ry = torso_ry - SKIN_THICK
    inner_fat_rx = inner_skin_rx - fat_thick
    inner_fat_ry = inner_skin_ry - fat_thick
    inner_muscle_rx = inner_fat_rx - MUSCLE_THICK
    inner_muscle_ry = inner_fat_ry - MUSCLE_THICK

    is_skin = np.sqrt((cx / inner_skin_rx) ** 2 + (cy / inner_skin_ry) ** 2) > 1
    is_fat = (
        np.sqrt((cx / inner_fat_rx) ** 2 + (cy / inner_fat_ry) ** 2) > 1
    ) & ~is_skin
    is_muscle = (
        np.sqrt((cx / inner_muscle_rx) ** 2 + (cy / inner_muscle_ry) ** 2) > 1
    ) & ~is_skin & ~is_fat

    labels[is_muscle] = 3
    labels[is_fat] = 2
    labels[is_skin] = 1

    # Bone (same regions as _assign_conductivities)
    r_iliac_L = np.sqrt(
        ((cx + 8.0) / 5.0) ** 2 + ((cy + 2.0) / 3.5) ** 2 + ((cz - 7.0) / 5.0) ** 2
    )
    r_iliac_R = np.sqrt(
        ((cx - 8.0) / 5.0) ** 2 + ((cy + 2.0) / 3.5) ** 2 + ((cz - 7.0) / 5.0) ** 2
    )
    r_spine = np.sqrt(
        (cx / 2.5) ** 2 + ((cy + 7.5) / 2.0) ** 2 + ((cz - 5.0) / 5.0) ** 2
    )
    r_pubis = np.sqrt(
        (cx / 1.5) ** 2 + ((cy - 7.5) / 1.25) ** 2 + ((cz - 2.5) / 2.0) ** 2
    )
    # Posterior pelvic ring (shell)
    pring_rx, pring_ry, pring_rz = 7.0, 5.5, 3.0
    pring_cz = 3.0
    pring_thick = 1.2
    avg_pring = (pring_rx + pring_ry) / 2.0
    inner_frac = 1.0 - pring_thick / avg_pring
    r_pring_outer = np.sqrt(
        (cx / pring_rx) ** 2 + (cy / pring_ry) ** 2
        + ((cz - pring_cz) / pring_rz) ** 2
    )
    r_pring_inner = np.sqrt(
        (cx / (pring_rx * inner_frac)) ** 2
        + (cy / (pring_ry * inner_frac)) ** 2
        + ((cz - pring_cz) / pring_rz) ** 2
    )
    is_pelvic_ring = (r_pring_outer < 1) & (r_pring_inner > 1) & (cy < -1.0)
    # Sacral alae
    r_ala_L = np.sqrt(
        ((cx + 4.0) / 3.0) ** 2 + ((cy + 6.0) / 1.5) ** 2
        + ((cz - 4.0) / 2.5) ** 2
    )
    r_ala_R = np.sqrt(
        ((cx - 4.0) / 3.0) ** 2 + ((cy + 6.0) / 1.5) ** 2
        + ((cz - 4.0) / 2.5) ** 2
    )
    is_bone = (
        (r_iliac_L < 1) | (r_iliac_R < 1) | (r_spine < 1) | (r_pubis < 1)
        | is_pelvic_ring | (r_ala_L < 1) | (r_ala_R < 1)
    )
    labels[is_bone] = 4

    # Organs
    r_bowel = np.sqrt(
        (cx / 8.0) ** 2 + ((cy - 1.5) / 5.0) ** 2 + ((cz - 12.0) / 5.0) ** 2
    )
    labels[(r_bowel < 1) & ~is_bone & ~is_skin & ~is_fat] = 5

    r_rectum = np.sqrt(
        (cx / 1.5) ** 2 + ((cy + 3.0) / 1.5) ** 2 + ((cz - 4.0) / 4.0) ** 2
    )
    labels[(r_rectum < 1) & ~is_bone] = 6

    r_vessel_L = np.sqrt(
        ((cx + 5.0) / 1.0) ** 2 + (cy / 1.0) ** 2 + ((cz - 5.0) / 4.0) ** 2
    )
    r_vessel_R = np.sqrt(
        ((cx - 5.0) / 1.0) ** 2 + (cy / 1.0) ** 2 + ((cz - 5.0) / 4.0) ** 2
    )
    labels[((r_vessel_L < 1) | (r_vessel_R < 1)) & ~is_bone] = 7

    r_perit = np.sqrt(
        (cx / 3.0) ** 2 + ((cy + 0.5) / 1.0) ** 2 + ((cz - 3.5) / 2.0) ** 2
    )
    labels[(r_perit < 1) & ~is_bone & ~(r_rectum < 1)] = 8

    # Prostate (male)
    r_prostate = np.sqrt(
        ((cx - PROSTATE_CENTER[0]) / PROSTATE_SEMI[0]) ** 2
        + ((cy - PROSTATE_CENTER[1]) / PROSTATE_SEMI[1]) ** 2
        + ((cz - PROSTATE_CENTER[2]) / PROSTATE_SEMI[2]) ** 2
    )
    labels[(r_prostate < 1) & ~is_bone] = 11

    # Bladder (highest priority)
    bl_a, bl_b, bl_c = bladder_semi_axes(bladder_volume_mL)
    bl_cy = bladder_center_y(bladder_volume_mL)
    bl_cz = BLADDER_BASE_Z + bl_c

    bx = cx / bl_a
    by = (cy - bl_cy) / bl_b
    bz = (cz - bl_cz) / bl_c
    r_bl = np.sqrt(bx ** 2 + by ** 2 + bz ** 2)
    avg_bl = (bl_a + bl_b + bl_c) / 3.0

    wall_thick = bladder_wall_thickness(bladder_volume_mL)
    is_urine = r_bl < (1.0 - wall_thick / avg_bl)
    is_wall = (r_bl < 1.0) & ~is_urine

    labels[is_wall] = 9
    labels[is_urine] = 10

    return labels


TISSUE_NAMES = [
    "Background",    # 0
    "Skin",          # 1
    "Fat",           # 2
    "Muscle",        # 3
    "Bone",          # 4
    "Bowel",         # 5
    "Rectum",        # 6
    "Vessels",       # 7
    "Peritoneal",    # 8
    "Bladder Wall",  # 9
    "Urine",         # 10
    "Prostate",      # 11
]

TISSUE_COLORS = np.array(
    [
        [0.700, 0.600, 0.500],  # background (brown)
        [0.900, 0.700, 0.500],  # skin (tan)
        [1.000, 0.900, 0.400],  # fat (yellow)
        [0.600, 0.200, 0.200],  # muscle (dark red)
        [0.800, 0.800, 0.800],  # bone (grey)
        [0.600, 0.400, 0.800],  # bowel (purple)
        [0.800, 0.400, 0.600],  # rectum (pink)
        [0.800, 0.100, 0.100],  # blood (red)
        [0.300, 0.700, 0.900],  # peritoneal (cyan)
        [0.400, 0.600, 0.400],  # bladder wall (green)
        [1.000, 1.000, 0.200],  # urine (bright yellow)
        [0.850, 0.600, 0.400],  # prostate (tan-orange)
    ]
)
