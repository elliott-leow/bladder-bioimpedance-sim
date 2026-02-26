"""
Bladder Bioimpedance Simulation
===============================

3D FEM simulation of bioimpedance for detecting bladder urine volume
and optimizing electrode placement.

Pure Python implementation (numpy/scipy/matplotlib).

Physics: Complete Electrode Model (CEM) on tetrahedral FEM mesh.
Tissue data: Gabriel et al. 1996 + IT'IS Foundation v4.1 database.

Modules
-------
tissue_properties : Frequency-dependent tissue conductivity database
mesh              : Structured tetrahedral mesh generation
fem               : FEM forward solver with CEM
model             : Anatomical pelvis model construction (male, 14+ tissue types)
analysis          : Sensitivity analysis, pattern & frequency optimization
figures           : Publication-quality visualization
"""

from .tissue_properties import get_conductivity, get_contact_impedance
from .mesh import create_torso_mesh
from .fem import ForwardModel, Image, forward_solve, compute_transfer_impedance
from .model import build_pelvis_model, get_bladder_mask, bladder_wall_thickness
from .analysis import (
    run_sensitivity_analysis,
    optimize_stim_patterns,
    frequency_sweep,
    optimize_electrode_placement,
)
from .figures import generate_publication_figures
