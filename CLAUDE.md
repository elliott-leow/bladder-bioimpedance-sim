# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

3D FEM simulation of bioimpedance for detecting bladder urine volume and optimizing electrode placement. Pure Python (numpy/scipy/matplotlib) implementation of the Complete Electrode Model (CEM) on a tetrahedral mesh with frequency-dependent tissue conductivities from Gabriel et al. 1996 and IT'IS Foundation v4.1.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Full simulation (several minutes)
python run_simulation.py

# Quick validation only (fast)
python run_simulation.py --quick

# Skip slow steps
python run_simulation.py --no-freq      # skip frequency sweep
python run_simulation.py --no-mfbis     # skip multi-frequency isolation
python run_simulation.py --no-figs      # skip figure generation

# Mesh control
python run_simulation.py --max-edge 0.5     # finer mesh (slower)
python run_simulation.py --n-per-ring 8     # fewer electrodes (faster)

# Generate AD5940 lab bench protocol
python test_protocol.py                     # outputs protocol.txt
```

No test suite exists. Validation is done by running `python run_simulation.py --quick` which checks bladder element counts, centroid position, forward solve finiteness, and sensitivity against literature ranges.

## Architecture

The simulation pipeline flows through 6 steps in `run_simulation.py`:

1. **Model construction** (`model.py`) - builds 3D male pelvis with 15 tissue types
2. **Forward solve validation** (`fem.py`) - CEM forward problem
3. **Sensitivity analysis** (`analysis.py`) - impedance vs bladder volume + Jacobian
4. **Pattern optimization** (`analysis.py`) - exhaustive 4-electrode search via transfer impedance
5. **Frequency sweep** (`analysis.py`) - multi-frequency optimization with SNR
6. **Figure generation** (`figures.py`) - 6 publication figures saved to `figures/`

### Core Package (`bladder_sim/`)

- **`tissue_properties.py`** - Conductivity database (PCHIP-interpolated lookup tables at 8 frequency points from 1-500 kHz). 16 tissue types including prostate. Also contains the measurement noise model (thermal + amplifier + polarization + CMRR). Entry points: `get_conductivity(tissue, freq_kHz)`, `get_contact_impedance(freq_kHz)`, `measurement_noise_floor(...)`.

- **`mesh.py`** - Structured tetrahedral mesh via Cartesian grid clipped to ellipsoidal cross-section, split into 5 tets per hex cell. Key types: `TorsoMesh` (nodes, elements, boundary_faces, electrodes), `Electrode` (nodes, faces, z_contact). Entry point: `create_torso_mesh(...)`.

- **`fem.py`** - FEM solver with CEM boundary conditions. Assembles stiffness matrix + CEM surface integrals into a single sparse system, uses `scipy.sparse.linalg.factorized` for efficient multi-RHS solves. Key functions: `forward_solve(img)`, `compute_transfer_impedance(fmdl, img)`, `compute_jacobian(img)` (adjoint method). Types: `ForwardModel`, `Image` (conductivity distribution), `StimPattern`.

- **`model.py`** - Anatomical geometry using nested ellipsoids. `build_pelvis_model(...)` is the main entry point: builds mesh, assigns conductivities by region priority (background < shells < bone < organs < prostate < bladder), returns `(ForwardModel, Image)`. Mesh can be reused across volumes/frequencies by passing the `mesh` parameter. Key anatomical features: volume-dependent bladder wall thickness, prostate gland, sacral alae, configurable fat layer thickness. `get_bladder_mask(mesh, volume)` returns boolean mask of urine elements.

- **`analysis.py`** - All analysis loops that call the forward solver repeatedly. Pattern optimization uses transfer impedance matrix (N-1 solves) rather than per-pattern forward solves. Multi-frequency isolation exploits urine's frequency-independent conductivity vs tissue beta dispersion.

- **`figures.py`** - Matplotlib with Agg backend. Uses Wong 2011 colorblind-friendly palette. Generates 6 PNG figures at 300 DPI.

### Key Design Patterns

- **Mesh reuse**: The mesh is built once and reused across all volume/frequency variations. Only `elem_data` (conductivity array) changes. Pass `mesh=existing_mesh` to `build_pelvis_model()`.
- **Tissue priority**: When assigning conductivities, regions are painted in order (background first, bladder last) so overlapping geometry resolves correctly. Prostate sits between organs and bladder in priority.
- **Volume-dependent wall thickness**: `bladder_wall_thickness(volume_mL)` returns BWT in cm. Detrusor thins from ~2.8 mm (empty) to ~1.5 mm (500 mL).
- **Fat thickness variability**: `FAT_THICK` module constant (default 1.5 cm) or `fat_thick` parameter in `build_pelvis_model()`. This is the dominant source of inter-subject variability.
- **All geometry in cm**, conductivities in S/m, currents in A, impedances in Ohm.
- **No external mesh generators** - pure numpy mesh construction.
