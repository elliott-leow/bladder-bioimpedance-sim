"""
Tissue conductivity database for bioimpedance simulation.

All conductivity values in S/m at 37°C.

Sources:
    [G1] Gabriel C, Gabriel S, Corthout E. Phys. Med. Biol. 41:2231-2249, 1996.
    [G2] Gabriel S, Lau RW, Gabriel C. Phys. Med. Biol. 41:2251, 1996.
    [G3] Gabriel S, Lau RW, Gabriel C. Phys. Med. Biol. 41:2271, 1996.
    [IT] IT'IS Foundation, Zurich. Tissue Properties Database v4.1
    [NI] IFAC-CNR, Florence. Dielectric Properties of Body Tissues.
    [R]  Rosell et al. IEEE Trans. Biomed. Eng. 35(8):649-651, 1988.
    [M]  McAdams et al. Med. Biol. Eng. Comput. 34(5):397-408, 1996.
"""

import numpy as np
from scipy.interpolate import PchipInterpolator

# Frequency table (kHz) for all tissue conductivity data
FREQ_TABLE_KHZ = np.array([1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0, 500.0])

# =====================================================================
# Tissue conductivity tables (S/m)
# Rows correspond to FREQ_TABLE_KHZ
# =====================================================================

CONDUCTIVITY_DB = {
    # Skin (dry, stratum corneum barrier; rises with beta dispersion)
    "skin": np.array([0.002, 0.015, 0.040, 0.070, 0.100, 0.180, 0.320, 0.430]),

    # Subcutaneous fat (infiltrated; nearly freq-independent < 1 MHz)
    "fat": np.array([0.025, 0.030, 0.033, 0.036, 0.040, 0.044, 0.048, 0.055]),

    # Skeletal muscle (weighted avg of transverse 0.13 & longitudinal 0.56)
    "muscle": np.array([0.200, 0.240, 0.270, 0.310, 0.350, 0.400, 0.450, 0.550]),

    # Bladder wall (detrusor smooth muscle)
    "bladder_wall": np.array([0.150, 0.170, 0.185, 0.200, 0.210, 0.250, 0.310, 0.400]),

    # Urine (electrolyte solution; frequency-independent below 1 MHz)
    "urine": np.array([1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75]),

    # Cortical bone (Haversian canal bound water; very low, slow rise)
    "bone_cortical": np.array([0.008, 0.010, 0.012, 0.015, 0.020, 0.028, 0.038, 0.058]),

    # Cancellous bone (marrow-infiltrated; higher than cortical)
    "bone_cancellous": np.array([0.060, 0.070, 0.078, 0.085, 0.095, 0.120, 0.160, 0.220]),

    # Blood (ionic fluid; flat < 1 MHz, slight rise from RBC dispersion)
    "blood": np.array([0.700, 0.700, 0.700, 0.700, 0.700, 0.700, 0.720, 0.750]),

    # Colon / large intestine wall (smooth muscle + mucosa)
    "colon_wall": np.array([0.100, 0.150, 0.200, 0.280, 0.350, 0.420, 0.480, 0.550]),

    # Small intestine wall
    "si_wall": np.array([0.350, 0.400, 0.430, 0.470, 0.530, 0.600, 0.700, 0.860]),

    # Bowel lumen gas (air; negligible conductivity)
    "bowel_gas": np.array([0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001]),

    # Bowel lumen fluid (chyme / fecal fluid; moderate electrolyte)
    "bowel_fluid": np.array([0.500, 0.500, 0.500, 0.500, 0.500, 0.500, 0.500, 0.500]),

    # Rectum wall (similar to colon; slightly higher vascularity)
    "rectum": np.array([0.120, 0.170, 0.220, 0.300, 0.380, 0.450, 0.500, 0.570]),

    # Peritoneal fluid (transudate; high ionic content like interstitial)
    "peritoneal": np.array([1.50, 1.50, 1.50, 1.50, 1.50, 1.50, 1.50, 1.50]),

    # Prostate (glandular + fibromuscular stroma; well-vascularised)
    # IT'IS Foundation v4.1 + Gabriel parametric model for glandular tissue
    "prostate": np.array([0.300, 0.340, 0.370, 0.400, 0.420, 0.460, 0.500, 0.550]),

    # Background (connective tissue / fascia / unclassified soft tissue)
    "background": np.array([0.200, 0.230, 0.250, 0.275, 0.300, 0.340, 0.380, 0.440]),
}

# Weighted average bone: 40% cortical + 60% cancellous
CONDUCTIVITY_DB["bone_avg"] = (
    0.4 * CONDUCTIVITY_DB["bone_cortical"]
    + 0.6 * CONDUCTIVITY_DB["bone_cancellous"]
)

# Effective bowel: 30% gas + 30% fluid + 40% colon wall
CONDUCTIVITY_DB["bowel_eff"] = (
    0.30 * CONDUCTIVITY_DB["bowel_gas"]
    + 0.30 * CONDUCTIVITY_DB["bowel_fluid"]
    + 0.40 * CONDUCTIVITY_DB["colon_wall"]
)

# =====================================================================
# Relative permittivity tables (dimensionless)
# Gabriel S, Lau RW, Gabriel C. Phys. Med. Biol. 41:2271, 1996.
# IT'IS Foundation v4.1.  Rows correspond to FREQ_TABLE_KHZ.
# At bioimpedance frequencies (1-500 kHz), permittivity contributes
# a reactive (imaginary) component to the complex admittivity:
#   gamma = sigma + j * omega * epsilon_0 * epsilon_r
# =====================================================================

PERMITTIVITY_DB = {
    # Skin (dry): huge low-freq permittivity from alpha dispersion, drops fast
    "skin": np.array([1.1e5, 3.3e4, 1.5e4, 5.6e3, 2.8e3, 1.4e3, 7.0e2, 3.5e2]),

    # Fat: low permittivity, weak dispersion
    "fat": np.array([1.5e4, 6.0e3, 3.0e3, 1.2e3, 6.0e2, 3.0e2, 1.5e2, 8.0e1]),

    # Skeletal muscle: strong beta dispersion
    "muscle": np.array([4.0e5, 1.1e5, 4.0e4, 1.2e4, 6.0e3, 2.5e3, 1.0e3, 4.0e2]),

    # Bladder wall (smooth muscle): moderate beta dispersion
    "bladder_wall": np.array([2.5e5, 7.0e4, 2.8e4, 9.0e3, 4.0e3, 1.8e3, 8.0e2, 3.5e2]),

    # Urine: ionic solution — very low permittivity (near water ~80)
    "urine": np.array([80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0]),

    # Cortical bone
    "bone_cortical": np.array([5.0e3, 2.0e3, 1.2e3, 6.0e2, 3.5e2, 2.0e2, 1.2e2, 7.0e1]),

    # Cancellous bone
    "bone_cancellous": np.array([8.0e4, 2.5e4, 1.2e4, 4.0e3, 2.0e3, 9.0e2, 4.5e2, 2.0e2]),

    # Blood
    "blood": np.array([5.0e3, 3.5e3, 3.0e3, 2.5e3, 2.0e3, 1.5e3, 8.0e2, 4.0e2]),

    # Colon wall
    "colon_wall": np.array([3.0e5, 8.0e4, 3.5e4, 1.1e4, 5.0e3, 2.2e3, 1.0e3, 4.0e2]),

    # Small intestine wall
    "si_wall": np.array([4.0e5, 1.2e5, 5.0e4, 1.5e4, 7.0e3, 3.0e3, 1.2e3, 5.0e2]),

    # Bowel gas (air: permittivity ~1)
    "bowel_gas": np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),

    # Bowel fluid
    "bowel_fluid": np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]),

    # Rectum wall
    "rectum": np.array([3.5e5, 9.0e4, 3.8e4, 1.2e4, 5.5e3, 2.4e3, 1.1e3, 4.2e2]),

    # Peritoneal fluid (ionic: low permittivity)
    "peritoneal": np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]),

    # Prostate (glandular tissue)
    "prostate": np.array([3.0e5, 8.5e4, 3.5e4, 1.1e4, 5.0e3, 2.2e3, 1.0e3, 4.0e2]),

    # Background (connective tissue)
    "background": np.array([2.0e5, 6.0e4, 2.5e4, 8.0e3, 3.5e3, 1.5e3, 7.0e2, 3.0e2]),
}

# Weighted averages to match conductivity DB
PERMITTIVITY_DB["bone_avg"] = (
    0.4 * PERMITTIVITY_DB["bone_cortical"]
    + 0.6 * PERMITTIVITY_DB["bone_cancellous"]
)
PERMITTIVITY_DB["bowel_eff"] = (
    0.30 * PERMITTIVITY_DB["bowel_gas"]
    + 0.30 * PERMITTIVITY_DB["bowel_fluid"]
    + 0.40 * PERMITTIVITY_DB["colon_wall"]
)

EPSILON_0 = 8.854187817e-12  # vacuum permittivity (F/m)

_perm_interpolators = {}


def _get_perm_interpolator(tissue: str) -> PchipInterpolator:
    if tissue not in _perm_interpolators:
        if tissue not in PERMITTIVITY_DB:
            raise ValueError(f"Unknown tissue '{tissue}' for permittivity.")
        _perm_interpolators[tissue] = PchipInterpolator(
            FREQ_TABLE_KHZ, PERMITTIVITY_DB[tissue]
        )
    return _perm_interpolators[tissue]


def get_permittivity(tissue: str, freq_kHz: float) -> float:
    """Get relative permittivity at a given frequency."""
    return float(_get_perm_interpolator(tissue)(freq_kHz))


def get_complex_admittivity(tissue: str, freq_kHz: float) -> complex:
    """
    Get complex admittivity gamma = sigma + j*omega*epsilon_0*epsilon_r.

    Parameters
    ----------
    tissue : str
        Tissue name.
    freq_kHz : float
        Frequency in kHz.

    Returns
    -------
    complex
        Admittivity in S/m.
    """
    sigma = get_conductivity(tissue, freq_kHz)
    eps_r = get_permittivity(tissue, freq_kHz)
    omega = 2.0 * np.pi * freq_kHz * 1e3
    return complex(sigma, omega * EPSILON_0 * eps_r)


# Electrode-skin contact impedance (Ohm*cm^2)
# Rosell 1988, McAdams 1996, Yang 2017.
# Ag/AgCl electrodes with conductive gel on abdominal skin.
CONTACT_IMPEDANCE_TABLE = np.array([200.0, 60.0, 30.0, 12.0, 5.0, 3.0, 1.5, 0.8])

# Build PCHIP interpolators (monotonic, smooth)
_interpolators = {}


def _get_interpolator(tissue: str) -> PchipInterpolator:
    """Get or create a PCHIP interpolator for the given tissue."""
    if tissue not in _interpolators:
        if tissue not in CONDUCTIVITY_DB:
            raise ValueError(
                f"Unknown tissue '{tissue}'. Available: {list(CONDUCTIVITY_DB.keys())}"
            )
        _interpolators[tissue] = PchipInterpolator(
            FREQ_TABLE_KHZ, CONDUCTIVITY_DB[tissue]
        )
    return _interpolators[tissue]


_zc_interpolator = None


def get_conductivity(tissue: str, freq_kHz: float) -> float:
    """
    Get tissue conductivity at a given frequency.

    Parameters
    ----------
    tissue : str
        Tissue name (e.g. 'skin', 'urine', 'bone_avg').
    freq_kHz : float
        Frequency in kHz (1 to 500).

    Returns
    -------
    float
        Conductivity in S/m.
    """
    return float(_get_interpolator(tissue)(freq_kHz))


def get_contact_impedance(freq_kHz: float) -> float:
    """
    Get electrode-skin contact impedance at a given frequency.

    Parameters
    ----------
    freq_kHz : float
        Frequency in kHz.

    Returns
    -------
    float
        Contact impedance in Ohm*cm^2.
    """
    global _zc_interpolator
    if _zc_interpolator is None:
        _zc_interpolator = PchipInterpolator(FREQ_TABLE_KHZ, CONTACT_IMPEDANCE_TABLE)
    return float(_zc_interpolator(freq_kHz))


def get_all_conductivities(freq_kHz: float) -> dict:
    """Get conductivities for all tissues at a given frequency."""
    return {tissue: get_conductivity(tissue, freq_kHz) for tissue in CONDUCTIVITY_DB}


def measurement_noise_floor(
    freq_kHz: float,
    I_drive: float = 1e-3,
    electrode_area_cm2: float = 0.785,
    bw_Hz: float = 100.0,
) -> float:
    """
    Equivalent impedance noise floor for a tetrapolar bioimpedance measurement.

    Models four noise sources:
        1. Thermal noise from contact impedance at sense electrodes
        2. Instrumentation amplifier voltage noise
        3. Electrode polarization noise (1/f, dominant < 10 kHz)
        4. Common-mode artifact from contact impedance mismatch

    Parameters
    ----------
    freq_kHz : float
        Measurement frequency in kHz.
    I_drive : float
        Drive current amplitude in Amperes (default 1 mA).
    electrode_area_cm2 : float
        Electrode contact area in cm^2 (default pi*0.5^2 = 0.785).
    bw_Hz : float
        Demodulation bandwidth in Hz (default 100).

    Returns
    -------
    float
        Equivalent noise in Ohms (V_noise_rms / I_drive).

    References
    ----------
    Rosell et al. IEEE Trans. BME 35(8):649-651, 1988.
    McAdams et al. Med. Biol. Eng. Comput. 34(5):397-408, 1996.
    Grimnes & Martinsen, Bioimpedance and Bioelectricity Basics, 3rd ed., 2014.
    """
    kT = 1.381e-23 * 310.0  # Boltzmann * body temperature (37 C)
    f_Hz = freq_kHz * 1e3

    z_c = get_contact_impedance(freq_kHz)  # Ohm*cm^2
    z_per_elec = z_c / electrode_area_cm2  # Ohm per electrode

    # 1. Thermal noise from contact impedance (2 sense electrodes in series)
    V_thermal = np.sqrt(4.0 * kT * 2.0 * z_per_elec * bw_Hz)

    # 2. Instrumentation amplifier noise (~10 nV/sqrt(Hz), typical INA116/AD8429)
    V_amp = 10e-9 * np.sqrt(bw_Hz)

    # 3. Electrode polarization noise (dominant below ~10 kHz)
    #    Empirical: ~1 uV*sqrt(Hz) / f_Hz for Ag/AgCl with gel
    V_pol = 1e-6 / np.sqrt(f_Hz) * np.sqrt(bw_Hz)

    # 4. Common-mode artifact: V_cm = I_drive * Zc_mismatch / CMRR
    #    10% contact impedance mismatch, CMRR = 80 dB (10000:1)
    CMRR = 1e4
    Zc_mismatch = 0.1 * z_per_elec
    V_cm = I_drive * Zc_mismatch / CMRR

    V_total = np.sqrt(V_thermal**2 + V_amp**2 + V_pol**2 + V_cm**2)
    return float(V_total / I_drive)
