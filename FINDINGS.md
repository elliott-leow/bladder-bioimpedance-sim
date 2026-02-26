# Simulation Findings

Results from 3D FEM bladder bioimpedance simulation. Male pelvis model with 15 tissue types, Complete Electrode Model, frequency-dependent conductivities (Gabriel et al. 1996 + IT'IS v4.1).

## Sensitivity Limits

The fundamental sensitivity of surface bioimpedance to bladder volume is constrained by anatomy — the bladder sits deep in the pelvis behind skin, fat, muscle, and bone. More electrodes don't help if you only use 4 at a time (tetrapolar). SVD-optimal multi-electrode drive patterns break through this ceiling by focusing current through the bladder.

| Configuration | Sensitivity (mOhm/mL) |
|---|---|
| 4-electrode tetrapolar (any count) | ~0.22 |
| 8 electrodes, SVD rank-1 | 0.47 |
| 16 electrodes, SVD rank-1 | 0.51 |
| 16 electrodes, SVD rank-3 | 0.63 |
| 32 electrodes, SVD rank-3 | 1.10 |
| 48 electrodes, SVD rank-3 | 1.55 |
| 128 electrodes, SVD rank-1 | 2.86 |

SVD-optimal means: drive multiple current patterns sequentially, linearly combine the voltage measurements with weights derived from the dominant singular vectors of the transfer impedance change matrix. The AD5940 can implement this by cycling through programmed drive pairs.

Fat thickness has a moderate effect (~2x across 0.5-2.5 cm), but drive strategy dominates.

## Noise Budget

Electronic noise is NOT the bottleneck. After 1 second of AD5940 averaging at 50 kHz (~100 measurements), electronic noise drops to ~13 uOhm. After 1 minute: ~1.6 uOhm. Sampling longer does not meaningfully improve resolution.

The dominant error sources are biological artifacts:

| Source | Magnitude | Averages away? |
|---|---|---|
| Electronic noise (single) | 127 uOhm | Yes (sqrt N) |
| Electronic noise (1 min) | 1.6 uOhm | - |
| Respiratory artifact | 20 mOhm peak-to-peak | No — coherent at 0.2-0.3 Hz |
| Electrode drift | ~5 uOhm/s (~0.3 mOhm/min) | No — slow trend |
| Posture/motion | ~2 mOhm per event | No — episodic |

Respiratory artifact is ~100x larger than the bladder signal per mL. This is the primary challenge.

## Mitigation Strategies

| Strategy | Residual noise | Implementation |
|---|---|---|
| Band-stop filter (0.15-0.4 Hz) | ~1.3 mOhm | Software DSP, trivial |
| Polynomial baseline detrending | reduces drift | Software |
| Dual-frequency subtraction (10 + 500 kHz) | ~0.02 mOhm respiratory, ~0.3 mOhm drift | Two AD5940 excitation frequencies, alpha-weighted subtraction |
| SVD-optimal drive patterns | Boosts signal 2-5x | Sequential drive pattern cycling |

Dual-frequency isolation exploits the fact that urine conductivity is frequency-independent (pure ionic, 1.75 S/m flat across 1-500 kHz), while surrounding tissues exhibit beta dispersion (conductivity increases with frequency as cell membranes become transparent). Subtracting Z(f1) - alpha*Z(f2) cancels tissue-change artifacts while preserving the bladder signal.

### Multi-Frequency Spectral Unmixing (7+ frequencies)

Tested: measure at K=7 frequencies (5-500 kHz), decompose into tissue templates via least-squares to separate bladder from respiratory, bowel gas, and drift simultaneously.

**Result: does NOT work well.** Template correlations are 0.999 between bladder and respiratory profiles. All tissue conductivities follow similar beta-dispersion curves (monotonically increasing), so the frequency templates are nearly colinear. Mixing matrix condition number ~475,000 — the system is too ill-conditioned for reliable source separation.

Multi-frequency gives marginal benefit over dual-frequency. The practical approach is layered mitigation: temporal filtering (bandstop) + dual-frequency subtraction + SVD spatial focusing.

## End-to-End SNR and Volume Resolution

| Tier | Config | Sensitivity | Total Noise | Resolution (1s) | Bottleneck |
|---|---|---|---|---|---|
| 1 | 8e, 1f, bandstop | 0.19 mOhm/mL | 1.9 mOhm | ~10 mL | Respiratory + bowel |
| 2 | 8e, 2f, bandstop | 0.07 mOhm/mL | 0.75 mOhm | ~11 mL | Bowel gas (99%) |
| 3 | 16e, 2f, SVD rank-3 | 0.24 mOhm/mL | 0.50 mOhm | ~2 mL | Bowel gas (99%) |
| 4 | 32e, 2f, SVD rank-3 | 0.41 mOhm/mL | 0.38 mOhm | ~0.9 mL | Bowel gas (99%) |

Electronic noise is never the bottleneck (0.013 mOhm after 1s averaging). The dominant noise source shifts from respiratory (Tier 1) to bowel gas (Tiers 2-4). Bowel gas is the fundamental limit because it's anatomically adjacent to the bladder, frequency-flat (insulator), and can only be rejected by spatial focusing (more electrodes + SVD).

### Target: 0.1 mL/kg/hr (= 7 mL at 70 kg)

Achievable with Tier 1 (simplest device):

- 8 electrodes (4/ring x 2 rings at z=9,10 cm)
- Single frequency (50 kHz)
- Band-stop respiratory filter in software
- Expected accuracy: +/- 5-6 mL (with 10s averaging: ~3 mL)
- SNR for 7 mL change: ~5 (Tier 3)

### Target: 0.015 mL/kg/hr (= 1.05 mL at 70 kg)

Requires Tier 3 or 4:

- 16 electrodes (8/ring x 2 rings)
- Dual-frequency excitation (10 kHz + 500 kHz)
- SVD rank-3 drive patterns (3 sequential measurements per cycle)
- Band-stop + dual-freq subtraction + polynomial detrend
- Expected accuracy: +/- 1-2 mL
- SNR for 1 mL change: ~3 (Tier 4, marginal)

## Optimal Electrode Placement

Swept 319 configurations (1-4 rings, 4/8/16 electrodes per ring, z-positions 3-12 cm).

Best by ring count (Phase 1 quick sweep, adjacent drive):

| Rings | Best config | Sensitivity | Electrodes |
|---|---|---|---|
| 1-ring | 16e at z=10 cm | 0.043 mOhm/mL | 16 |
| 2-ring | 8e at z=9,10 cm | 0.236 mOhm/mL | 16 |
| 3-ring | 8e at z=8,9,10 cm | 0.235 mOhm/mL | 24 |
| 4-ring | 8e at z=8,8,10,10 cm | 0.234 mOhm/mL | 32 |

Key findings:
- 2 rings is optimal — additional rings give <1% improvement while doubling electrode count
- z=9-10 cm (suprapubic region) is the sweet spot, directly above the bladder
- 8 electrodes/ring is sufficient (4e/ring is nearly as good)
- 1-ring configurations are ~5x worse, confirming axial separation is critical
- The best 4-electrode patterns use posterior drive electrodes (behind the bladder) with sense electrodes on an adjacent ring

### Anterior-Only vs Anterior-Posterior Placement

Exhaustive 4-electrode search across 64 electrodes (4 rings x 16/ring), classifying each configuration by electrode position (Y > 0 = anterior/abdomen, Y < 0 = posterior/back):

| Configuration | Sensitivity (mOhm/mL) | % of Global Best |
|---|---|---|
| Global best (= AP) | 0.190 | 100% |
| Anterior-posterior (mixed front+back) | 0.190 | 100% |
| Anterior-only (all on abdomen) | 0.165 | 87% |

Sensitivity loss from anterior-only: **only 13%**. A supra-pubic belt with all electrodes on the abdomen is entirely practical — no electrodes on the patient's back needed.

Best anterior-only: drive and sense pairs on rings at z=8 and z=10 cm, one electrode at the midline (directly anterior) and one offset laterally (~70 degrees), creating an asymmetric drive that still routes current through the bladder.

## Bladder Growth Model

Implemented anisotropic volume-dependent expansion based on ultrasound literature:

- At low volumes (50 mL): oblate shape, H:W ratio = 0.76
- At high volumes (300+ mL): near-spherical, H:W ratio = 1.06
- Based on Glass Clark & Nagle et al. 2020 (PMC6538469), Nagle et al. 2018

Growth direction: the bladder base (trigone) is fixed at the pelvic floor. As the bladder fills, the dome expands superiorly (upward) and posteriorly (backward). The anterior wall is constrained by the pubic symphysis, so expansion is preferentially posterior. The geometric center Y shifts from 3.0 cm (empty, anterior) to 2.0 cm (full, shifted ~1 cm posteriorly).

Wall thickness uses geometric surface-area scaling with asymptotic minimum:
- BWT = t_ref * (V_ref / V)^(2/3), floored at 1.3 mm
- Rapid decrease <250 mL (rugae unfolding), plateau after
- Based on Oelke et al. 2006, Ugwu et al. 2019

Bladder dimensions at key volumes:

| Volume (mL) | Lateral (cm) | AP (cm) | SI (cm) | H:W | BWT (mm) | Center Y (cm) |
|---|---|---|---|---|---|---|
| 50 | 5.4 | 4.3 | 4.1 | 0.76 | 2.8 | 3.0 |
| 100 | 6.6 | 5.3 | 5.4 | 0.82 | 1.8 | 2.9 |
| 200 | 8.0 | 6.4 | 7.5 | 0.94 | 1.3 | 2.7 |
| 300 | 8.8 | 7.0 | 9.3 | 1.06 | 1.3 | 2.4 |
| 500 | 10.4 | 8.3 | 11.0 | 1.06 | 1.3 | 2.0 |

## Phase (Complex Impedance) Analysis

At bioimpedance frequencies (1-500 kHz), tissue permittivity contributes a small reactive (imaginary) component to the complex admittivity: gamma = sigma + j*omega*eps_0*eps_r.

Urine has near-zero phase (~0.01 deg at 50 kHz, eps_r = 80), while surrounding tissues have measurable phase from cell membrane capacitance (muscle ~2.7 deg, skin ~4.5 deg at 50 kHz, eps_r = 3000-6000).

Phase sensitivity to bladder volume is small relative to magnitude:

| Component | Best sensitivity | Ratio to Re(Z) |
|---|---|---|
| Re(Z) (resistive) | 0.142 mOhm/mL | 1.00 |
| Im(Z) (reactive) | 0.007 mOhm/mL | 0.049 |
| Phase | ~39 mdeg/mL | N/A |

Im/Re ratio decreases with frequency (10.9% at 5 kHz, 4.9% at 50 kHz, 3.0% at 500 kHz) because omega*eps_0*eps_r grows linearly with frequency but eps_r drops (beta dispersion), partially cancelling.

Conclusion: magnitude-only analysis captures >95% of the bladder signal. Phase provides a small additional signal that could be used as a secondary indicator, but is not worth the added complexity for primary volume estimation. The AD5940 measures both natively anyway.

## Minimum Viable Device

For 0.1 mL/kg/hr clinical accuracy:

```
Hardware:
  - 8 Ag/AgCl gel electrodes (4 per ring)
  - 2 rings at z=9 cm and z=10 cm (suprapubic belt)
  - AD5940 evaluation board, 50 kHz excitation, 1 mA drive
  - Single tetrapolar or SVD rank-1 measurement

Software:
  - Band-stop filter at 0.15-0.4 Hz (respiratory rejection)
  - Polynomial baseline detrending (drift correction)
  - Calibration: linear fit of impedance vs known volumes

Expected resolution: 3-6 mL per reading
```

For 0.015 mL/kg/hr (aggressive target):

```
Hardware:
  - 16 electrodes (8 per ring, 2 rings)
  - AD5940 dual-frequency: 10 kHz + 500 kHz
  - SVD-optimal drive patterns (3 sequential per cycle)

Software:
  - Dual-frequency subtraction: Z_iso = Z(f1) - alpha*Z(f2)
  - Alpha optimized to cancel respiratory artifact
  - SVD-weighted measurement combination

Expected resolution: ~1 mL per reading
```

## Concrete Signal Processing Algorithm

For the optimal Tier 3 configuration (16 electrodes, dual-frequency, SVD):

```
REAL-TIME PROCESSING (per measurement cycle, ~70 ms):

1. Sweep frequencies: measure Z at 10 kHz and 500 kHz
   (3 SVD drive patterns x 2 frequencies = 6 measurements, ~10 ms each)

2. SVD combination: for each frequency fk:
   Z_combined(fk) = w1*Z_pattern1(fk) + w2*Z_pattern2(fk) + w3*Z_pattern3(fk)
   where w1,w2,w3 are SVD rank-3 weights from calibration

3. Dual-frequency subtraction:
   Z_isolated = Z_combined(10 kHz) - alpha * Z_combined(500 kHz)
   alpha = 1.80 (optimized to cancel respiratory artifact)

4. Temporal filtering (software, on accumulated samples):
   - Band-stop filter at 0.15-0.4 Hz (respiratory residual)
   - Low-pass filter at 0.5 Hz (cardiac rejection)
   - Polynomial detrending over 5-min window (drift)

5. Volume estimate:
   dV = (Z_isolated - Z_baseline) / sensitivity_cal
   where sensitivity_cal is from patient-specific calibration

CALIBRATION (once per session):
  1. Patient breathes normally 30s → extract alpha from periodic component
  2. Known void event → measure Z change per mL for patient-specific calibration
```

## Artifact Catalog

All known sources of error for pelvic bioimpedance bladder monitoring, ordered by severity.

### Biological Artifacts

| Source | Magnitude | Frequency | Ratio to signal | Mitigation |
|---|---|---|---|---|
| Respiratory | 20 mOhm pk-pk | 0.15-0.4 Hz | ~100x per mL | Band-stop filter (15x), dual-freq (1000x) |
| Posture change | 0.5-2 Ohm (trunk) | Step + 5-10 min drift | ~2000-10000x | Accelerometer gating, stabilization period |
| Food/drink intake | 4-15 Ohm (whole body) | 0.5-4 hr trend | Huge | Protocol: fast or note meal times |
| Cardiac (pelvic) | 1-5 mOhm | 0.8-2.0 Hz | ~5-25x | Low-pass filter below 0.5 Hz |
| Peristalsis/bowel | 0.5-5 mOhm | 0.05-0.2 Hz | ~2-25x | Temporal averaging, dual-freq |
| Temperature (1 degC) | ~430 mOhm | Very slow | ~2000x | Thermistor correction, ratiometric |
| Detrusor contractions | 1-10 mOhm | Episodic, 5-30s | ~5-50x | Moving median, outlier rejection |
| Fluid redistribution (supine) | 0.5-2 Ohm | 30-60 min drift | ~2000-10000x | Wait 10+ min, posture calibration |
| Menstrual cycle | <0.3 Ohm | ~28 day | Negligible | Ignorable for acute monitoring |

### Physiological Confounders (non-bladder pelvic changes)

| Source | Magnitude | Correctable? |
|---|---|---|
| Bowel gas | Potentially Ohms | Difficult; multi-freq may help (gas = freq-independent) |
| Fecal loading (rectum) | 0.5-5+ mOhm | Protocol: bowel evacuation; event logging |
| Urine conductivity variation | 2-4x sensitivity change | Multi-freq estimation; void-event calibration |
| Adipose attenuation | ~2x across 0.5-2.5 cm fat | Patient-specific calibration; SVD drive patterns |
| Blood pooling (standing) | 300-800 mL blood shift | Accelerometer posture detection; stabilization |
| Skin stretching (bladder distension) | Correlated with signal | Rigid belt; dual-freq subtraction |

### Electrode Artifacts

| Source | Magnitude | Mitigation |
|---|---|---|
| Skin-electrode drift | ~5 uOhm/s (0.3 mOhm/min) | Tetrapolar rejects; 10-15 min stabilization; detrending |
| Gel drying | Progressive increase after ~1 hr | Replace every 8-12 hrs; porous electrode structures |
| Sweat | Variable conductive bridge | Cool/dry environment; tetrapolar measurement |
| Motion artifact | ~2 mOhm per event | Higher freq (>50 kHz); accelerometer gating; active electrodes |
| Electrode polarization | Large at <1 kHz, negligible at >10 kHz | Operate at 50 kHz; Ag/AgCl electrodes; tetrapolar |

### Environmental/Technical

| Source | Magnitude | Mitigation |
|---|---|---|
| EMI (50/60 Hz) | uV-mV common mode | Synchronous demodulation at 50 kHz rejects completely |
| Cable triboelectric | uV-mV | Active shielding; short cables; active electrodes |
| Electronics temperature drift | 10-50 ppm/degC | AD5940 self-calibration (reduces error from 3.87% to 0.14%) |
| Stray capacitance | Systematic error >500 kHz | Operate below 500 kHz; compensation; Cole model fitting |

### Key Insight

The three hardest artifacts to handle are:
1. **Respiratory** (~20 mOhm) — solvable with dual-frequency subtraction
2. **Bowel gas/fecal loading** — anatomically adjacent to bladder, similar magnitude, hard to separate without spatial imaging (EIT)
3. **Posture changes** — orders of magnitude larger than signal, requires stabilization + gating

## Gelatin Phantom vs Anatomical Simulation: The ~700x Sensitivity Disparity

A gelatin phantom experiment typically shows ~100 mOhm/mL sensitivity, while this anatomical simulation gives ~0.19 mOhm/mL. This ~500-700x gap is entirely expected.

### The six factors

| Factor | Contribution | Mechanism |
|---|---|---|
| Electrode-to-bladder distance | ~10-50x | Phantom: 1-3 cm. Anatomy: 3-8 cm through skin+fat+muscle. Sensitivity drops as ~1/d^2 to 1/d^3 |
| Current shunting | ~10-30x | Muscle (0.35 S/m), blood (0.70 S/m), peritoneal fluid (1.50 S/m) provide massive parallel conductance paths that don't exist in uniform gelatin |
| Volume conductor size | ~3-10x | Bladder occupies 2-5% of pelvic cross-section vs 10-30% of a phantom tank |
| Pelvic bone shielding | ~2-5x | Low-conductivity bone ring partially encircles the bladder, deflecting current |
| Fat insulation layer | ~2-5x | Fat (0.04 S/m) is ~10x less conductive than muscle, acts as insulating barrier |
| Bladder wall attenuation | ~1.5-3x | Detrusor wall (0.21 S/m) partially masks the urine (1.75 S/m) |

Compound: 10 x 10 x 5 x 3 x 2 x 2 = ~6000x (factors overlap, actual ~500-700x).

The Jacobian sensitivity ratio from the simulation directly quantifies this: urine elements have only 1.3% of the average sensitivity of other elements. In gelatin, this ratio would be near 100%.

Published ranges: simplified/tank models 1-50 mOhm/mL, 3D anatomical models 0.05-5 mOhm/mL (Leonhardt 2012, Schlebusch 2014, Kim et al.).

## References

- Gabriel C, Gabriel S, Corthout E. Phys. Med. Biol. 41:2231-2249, 1996.
- Glass Clark SE, Nagle AS, et al. Female Pelvic Med Reconstr Surg. PMC6538469, 2020.
- Oelke M, et al. Neurourol Urodyn. PMID: 16652381, 2006.
- Ugwu AC, et al. J Diagnostic Medical Sonography. DOI: 10.1177/8756479318799295, 2019.
- Nagle AS, et al. Bladder. PMC5771657, 2018.
- Leonhardt S, et al. Sensors 25(24):7635, 2025.
- Schlebusch T, et al. Physiol. Meas. 35(9):1813-1823, 2014.
- Damaser MS, Lehman SL. J Biomech. PMID: 7601871, 1995.
- Bolton MP, et al. Physiol Meas. 19(2):235-245, 1998.
- Buendia R, et al. PLoS ONE. 11(6):e0156522, 2016.
- Keshtgar AS, et al. Physiol Meas. 19(4):527-534, 1998.
- Gonzalez-Araiza JR, et al. World J Gastrointest Pathophysiol. 3(1):10-18, 2012.
- Fernandez-Fuentes M, et al. Sensors. 24(18):5871, 2024.
- Scharfetter H, et al. Kidney Int. 51:1078-1087, 1997.
- Buxi D, et al. BioMed Eng OnLine. 13:149, 2014.
- Medrano G, et al. IEEE EMBC, 2023 (AD5940 self-calibration).
