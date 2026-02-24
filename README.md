# Bladder Bioimpedance Simulation

EIDORS-based 3D FEM simulation for bladder volume estimation via bioimpedance measurements.

## Overview
Simulates a pelvis model with configurable electrode placement, tissue conductivities (skin, fat, muscle, bladder wall, urine), and multi-frequency sweeps. Uses Netgen for mesh generation.

## Requirements
- MATLAB R2025b+
- [EIDORS](http://eidors3d.sourceforge.net/) v3.12+
- Netgen (via Wine on macOS ARM)

## Quick Start
```matlab
run('/path/to/eidors/startup.m');
run_bladder_sim;
```

## Files
| File | Description |
|------|-------------|
| `run_bladder_sim.m` | Main entry point — runs all simulation steps |
| `build_pelvis_model.m` | Wrapper with default electrode positions |
| `build_pelvis_model_param.m` | Full 3D FEM model builder with configurable electrodes |
| `run_sensitivity_analysis.m` | Bladder volume sensitivity sweep |
| `frequency_sweep.m` | Multi-frequency impedance analysis |
| `optimize_electrode_placement.m` | Electrode placement optimization |
| `get_bladder_mask.m` | Bladder region element mask utility |
