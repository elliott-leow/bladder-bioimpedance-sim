# Bladder Bioimpedance Simulation

3D FEM simulation of bioimpedance for detecting bladder urine volume and optimizing electrode placement. Uses the Complete Electrode Model (CEM) on a tetrahedral mesh with frequency-dependent tissue conductivities from Gabriel et al. 1996 and IT'IS Foundation v4.1.

## Quick Start

```bash
# Clone
git clone https://github.com/elliott-leow/bladder-bioimpedance-sim.git
cd bladder-bioimpedance-sim

# Install dependencies
pip install -r requirements.txt

# Run quick validation
python run_simulation.py --quick

# Run full simulation (takes several minutes)
python run_simulation.py

# Generate lab bench protocol for AD5940 hardware
python test_protocol.py
```

## Requirements

- Python 3.9+
- numpy >= 1.24
- scipy >= 1.10
- matplotlib >= 3.7

Install with:
```bash
pip install -r requirements.txt
```

## Project Structure

```
├── bladder_sim/                 # Core simulation package
│   ├── __init__.py
│   ├── tissue_properties.py     # Gabriel 1996 + IT'IS v4.1 conductivity database + noise model
│   ├── mesh.py                  # Structured tetrahedral mesh generation
│   ├── fem.py                   # FEM forward solver (CEM), Jacobian, transfer impedance
│   ├── model.py                 # 3D anatomical male pelvis model (15 tissue types)
│   ├── analysis.py              # Sensitivity, pattern optimization, frequency sweep
│   └── figures.py               # Publication-quality figure generation
├── run_simulation.py            # Main simulation runner
├── test_protocol.py             # Generates AD5940 lab bench validation protocol
└── requirements.txt
```

## Usage

### Full Simulation

```bash
python run_simulation.py
```

This runs:
1. Model construction (3D male pelvis with 15 tissue types)
2. Forward solve validation
3. Sensitivity analysis (impedance vs. bladder volume)
4. Exhaustive 4-electrode stimulation pattern optimization
5. Multi-frequency sweep with SNR optimization
6. Dual-frequency respiratory artifact isolation
7. Publication figure generation (saved to `figures/`)

### Options

```
--quick       Stop after validation (fast check that the model works)
--no-freq     Skip frequency sweep (the slowest step)
--no-mfbis    Skip multi-frequency bladder isolation analysis
--no-figs     Skip figure generation
--max-edge N  Mesh element edge length in cm (default 1.0; smaller = finer mesh)
--n-per-ring N  Electrodes per ring (default 16)
```

### Lab Bench Protocol

```bash
python test_protocol.py
```

Generates `protocol.txt` — a complete lab bench protocol for validating simulation predictions on a human subject using an Analog Devices AD5940 evaluation board. Includes:
- AD5940 configuration (excitation, DFT, PGA, RTIA settings)
- Electrode placement with ASCII diagrams
- Three experiments: volume calibration, frequency sweep, respiratory artifact characterization
- Expected quantitative results from simulation
- Safety considerations (IEC 60601-1)

## Simulation Model

### Anatomy
The male pelvis model includes 15 tissue regions arranged in anatomically realistic geometry:
- **Outer shells**: skin (2 mm), subcutaneous fat (1.5 cm, configurable), skeletal muscle (1.5 cm)
- **Pelvic bone**: iliac wings, sacrum/spine + sacral alae, pubic symphysis, posterior ring
- **Soft organs**: bowel, rectum, iliac vessels, peritoneal fluid, prostate
- **Bladder**: detrusor wall (volume-dependent thickness) + urine (ellipsoidal, 50-500 mL)

Key anatomical features:
- **Volume-dependent bladder wall**: BWT decreases from ~2.8 mm (empty) to ~1.5 mm (500 mL), per sonographic data (Oelke et al. 2006)
- **Prostate gland**: 4.0 x 3.0 x 2.5 cm, inferior to bladder base
- **Sacral alae**: lateral wings connecting sacrum to iliac bones
- **Configurable fat thickness**: dominant source of inter-subject variability (Leonhardt et al. 2025)

### Physics
- Complete Electrode Model (CEM) on tetrahedral FEM mesh
- Frequency-dependent conductivities (1-500 kHz) from Gabriel et al. 1996
- Contact impedance model from Rosell 1988 / McAdams 1996
- Measurement noise model (thermal + amplifier + polarization + CMRR)

### Key Results
- **Sensitivity**: ~0.1-0.2 mOhm/mL (best channel, anterior-posterior drive)
- **Optimal frequency**: SNR peaks near 25-50 kHz
- **Artifact rejection**: Dual-frequency subtraction provides >10x respiratory artifact cancellation in simulation

## References

- Gabriel C, Gabriel S, Corthout E. *Phys. Med. Biol.* 41:2231-2249, 1996.
- Gabriel S, Lau RW, Gabriel C. *Phys. Med. Biol.* 41:2251, 1996.
- Gabriel S, Lau RW, Gabriel C. *Phys. Med. Biol.* 41:2271, 1996.
- IT'IS Foundation. *Tissue Properties Database v4.1*. Zurich.
- Somersalo E, Cheney M, Isaacson D. *SIAM J. Appl. Math.* 52(4):1023-1040, 1992.
- Adler A, Lionheart WRB. *Physiol. Meas.* 27:S25-S42, 2006.
- Rosell J et al. *IEEE Trans. Biomed. Eng.* 35(8):649-651, 1988.
- Schlebusch T et al. *Physiol. Meas.* 35(9):1813-1823, 2014.
- Oelke M et al. *Eur. Urol.* 2006. (Bladder wall thickness)
- Leonhardt S et al. *npj Digital Medicine* 2025. (Digital twin electrode optimization)
