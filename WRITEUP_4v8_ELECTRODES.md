# 4-Electrode vs 8-Electrode Bladder Bioimpedance: A Practical Comparison

## 1. Introduction

Bladder bioimpedance monitoring measures small changes in electrical impedance across the lower abdomen to track urine volume non-invasively. The bladder sits deep in the pelvis — behind skin, fat, muscle, and bone — so the impedance signal from bladder filling is tiny: roughly 0.2-1.0 mOhm per mL of urine, buried under respiratory artifacts 100x larger.

The number of electrodes and how you drive current through them determines whether this signal is detectable. This document compares two practical configurations using the same 8-electrode suprapubic belt:

- **4-electrode tetrapolar**: use 4 of the 8 electrodes in a standard 4-wire measurement
- **8-electrode SVD-optimal**: use all 8 electrodes with mathematically optimized drive patterns

The key finding: **SVD rank-1 drive patterns give 4.3x better sensitivity than tetrapolar, using the same hardware.** The improvement is purely in firmware — no additional electrodes or circuitry needed.

---

## 2. Electrode Placement

![Figure 1: Belt Placement](figures/writeup_fig1_belt_placement.png)

### The belt

Eight Ag/AgCl gel electrodes are arranged in two rings on a suprapubic belt:

| Ring | Height (z) | Electrodes | Angular positions |
|------|-----------|------------|-------------------|
| Ring 1 | 9 cm | 4 | 0, 90, 180, 270 degrees |
| Ring 2 | 10 cm | 4 | 0, 90, 180, 270 degrees |

The belt wraps around the lower abdomen at the suprapubic region, directly above the bladder dome. All electrodes are on the torso surface — **no electrodes on the back are needed** (anterior-only placement loses only 13% sensitivity vs anterior-posterior; see simulation results).

### Why z = 9-10 cm?

The bladder dome at 300 mL extends to approximately z = 12.6 cm above the pelvic floor. Electrode rings at z = 9-10 cm sit directly above the upper bladder, where the urine volume is largest and closest to the skin surface. The simulation swept 319 configurations and confirmed that 2 rings at z = 9-10 cm is optimal — additional rings give <1% improvement while increasing electrode count.

### Why 4 electrodes per ring?

The simulation shows that 4 electrodes/ring captures ~90% of the maximum sensitivity achievable with 16 electrodes/ring. The 4 positions (anterior, posterior, left lateral, right lateral) provide sufficient angular sampling for the dominant current patterns. Going from 4 to 8 per ring adds only ~10% sensitivity but doubles the hardware.

### Depth to bladder

From the anterior skin surface to the bladder:
- Skin: 0.2 cm
- Subcutaneous fat: 1.5 cm (varies with BMI)
- Abdominal muscle: 1.5 cm
- Interior tissue: ~3.5 cm
- **Total: ~6.7 cm**

This depth is why sensitivity is low — current must traverse 6+ cm of tissue to reach the bladder, and most of it takes the path of least resistance through muscle (0.35 S/m) rather than through the deep pelvis.

---

## 3. Why 4 Electrodes Hit a Ceiling

A tetrapolar (4-electrode) measurement uses one pair of electrodes to inject current and another pair to measure the resulting voltage:

```
Drive:  electrode A → electrode B  (inject 1 mA at 50 kHz)
Sense:  electrode C → electrode D  (measure voltage difference)
```

This eliminates contact impedance from the measurement (the voltage electrodes draw negligible current). But it constrains the current to a single path through the body — and that path is determined by anatomy, not by the bladder.

**The problem**: the pelvis contains massive parallel conductance paths. Skeletal muscle (0.35 S/m), blood vessels (0.70 S/m), and peritoneal fluid (1.50 S/m) all conduct current more readily than the path through fat and bone to reach the bladder. With a single drive pair, the Jacobian sensitivity analysis shows that urine elements have only **1.3% of the average sensitivity** of other elements. Most of the current bypasses the bladder entirely.

**The ceiling**: the best possible 4-electrode sensitivity with 8 electrodes (exhaustive search of all C(8,2) x C(6,2) = 420 configurations) is **0.22 mOhm/mL**. This is the same ceiling regardless of whether you have 8, 16, or 64 total electrodes — if you only use 4 at a time. A 10 mL bladder change produces just 2.2 mOhm of signal, while respiratory artifact is 20 mOhm. The per-reading SNR is roughly 0.1 — completely inadequate.

---

## 4. SVD-Optimal Drive Patterns

![Figure 2: SVD Explained](figures/writeup_fig2_svd_explained.png)

### The idea

Instead of driving current through one pair at a time, we drive multiple pairs in sequence and combine the measurements with optimal weights. The weights are computed by Singular Value Decomposition (SVD) of the transfer impedance change matrix — a matrix that captures how the impedance between every pair of electrodes changes when the bladder fills.

### How it works, step by step

**Step 1: Build the transfer impedance change matrix.**

For each pair of electrodes (i, j), compute the 4-electrode transfer impedance at two bladder volumes (100 mL and 500 mL). The difference, normalized by 400 mL, gives the sensitivity for that drive-sense combination. Arrange all C(8,2) = 28 drive pairs and 28 sense pairs into a 28 x 28 matrix **M**, where M[d, s] is the sensitivity when driving pair d and sensing pair s.

**Step 2: SVD decomposition.**

Compute M = U S V^T. The singular values S tell you how much sensitivity each "mode" contributes:

| Rank | Singular value | Sensitivity | Improvement over tetrapolar |
|------|---------------|-------------|----------------------------|
| 1 | 0.93 | 0.93 mOhm/mL | 4.3x |
| 2 | 0.91 | 0.91 mOhm/mL | — |
| 3 | 0.26 | 0.26 mOhm/mL | — |
| Cumulative rank-3 | — | 1.33 mOhm/mL | 6.1x |

**Step 3: Implement the rank-1 pattern.**

The first left singular vector U[:,0] gives the optimal drive weights, and V[0,:] gives the optimal sense weights. In practice:

1. **Measurement 1**: inject current with pattern proportional to U[:,0] — this means driving multiple pairs with specific current ratios
2. **Measurement 2**: repeat with the orthogonal "anti-pattern" (needed for robustness)
3. **Combine**: multiply the measured voltages by V[0,:] and sum

The AD5940 implements this by cycling through 2-3 programmed drive pair sequences per measurement cycle. Each sequence takes ~10 ms, so the total cycle time is ~20-30 ms.

### Why it works: current focusing

The SVD-optimal pattern achieves constructive interference of current paths through the bladder region. Where a single drive pair sends most current around the bladder through low-impedance muscle, the weighted combination of multiple drive pairs creates a virtual "focused beam" that preferentially passes through the bladder (Figure 2b). This is conceptually similar to beamforming in ultrasound — but with DC/AC current instead of sound waves.

### What "rank-1" means practically

"Rank-1" means we use only the first (most informative) singular vector — a single optimized drive-sense pattern. "Rank-3" means we use the top 3 patterns and combine them, extracting more information from the same electrodes. Each additional rank requires one more sequential measurement per cycle.

| Pattern | Measurements per cycle | Sensitivity | Hardware complexity |
|---------|----------------------|-------------|---------------------|
| Tetrapolar | 1 | 0.22 mOhm/mL | Trivial |
| SVD rank-1 | 2 | 0.93 mOhm/mL | Low (2 programmed patterns) |
| SVD rank-3 | 4 | 1.33 mOhm/mL | Moderate (4 programmed patterns) |

---

## 5. Artifact Rejection

![Figure 3: Signal Processing Chain](figures/writeup_fig3_signal_processing.png)

The raw bioimpedance signal is dominated by artifacts — primarily respiratory motion (20 mOhm peak-to-peak) compared to the bladder signal (0.94 mOhm/mL x 0.5 mL/min = 0.47 mOhm per minute of filling). The signal processing chain extracts the bladder trend from this noise:

### Stage 1: Band-stop filter (0.15-0.4 Hz)

Respiratory artifact is quasi-periodic at 12-20 breaths/minute (0.2-0.33 Hz). A 4th-order Butterworth band-stop filter centered on this band reduces the respiratory component by ~15x, from 20 mOhm to ~1.3 mOhm.

**Implementation**: IIR filter in firmware or post-processing. Adds ~10 ms latency. No hardware changes needed.

### Stage 2: Low-pass filter (< 0.08 Hz)

Removes cardiac artifact (~3 mOhm at 1 Hz) and any residual respiratory harmonics. The bladder signal changes over minutes, so frequencies above 0.08 Hz are irrelevant.

**Implementation**: 3rd-order Butterworth low-pass. Can be combined with Stage 1.

### Stage 3: Polynomial baseline detrending

Electrode drift (~5 uOhm/s = 0.3 mOhm/min) causes a slow baseline shift unrelated to bladder volume. A sliding-window polynomial fit (2-5 minute window) removes this trend while preserving the bladder filling rate.

**Implementation**: software, runs every few seconds on the buffered signal.

### Stage 4: Calibration and volume estimation

The detrended signal is divided by the patient-specific sensitivity calibration factor (mOhm/mL) to obtain volume change. Calibration is established from a known void event (impedance change / voided volume).

### Why NOT dual-frequency at 8 electrodes?

Dual-frequency subtraction (Z(10 kHz) - alpha * Z(500 kHz)) effectively eliminates respiratory artifact because respiratory tissue changes have a different frequency profile than the bladder signal. However, at 8 electrodes:

- **Sensitivity drops 82%**: the alpha subtraction cancels most of the bladder signal too (from 0.30 to 0.055 mOhm/mL)
- **Bowel gas is amplified**: the subtraction amplifies bowel gas artifacts because bowel tissue has a different frequency dispersion than muscle/background
- **Net result**: worse performance than single-frequency + bandstop

Dual-frequency becomes worthwhile only at 16+ electrodes where SVD rank-3 can compensate the sensitivity loss. For 8 electrodes, the software-only filtering approach (bandstop + detrend) is superior.

---

## 6. Head-to-Head Comparison

![Figure 4: Comparison](figures/writeup_fig4_comparison.png)

### Sensitivity

| Configuration | Sensitivity | Improvement |
|---|---|---|
| 4-electrode tetrapolar | 0.22 mOhm/mL | baseline |
| 8-electrode SVD rank-1 | 0.94 mOhm/mL | **4.3x** |
| 8-electrode SVD rank-3 | 1.33 mOhm/mL | **6.1x** |

The SVD improvement comes entirely from smarter current drive — same belt, same electrodes, different firmware.

### Noise budget

| Source | 4-elec tetrapolar | 8-elec SVD rank-1 |
|---|---|---|
| Electronic (1s avg) | 0.013 mOhm (0%) | 0.013 mOhm (0%) |
| Respiratory (after bandstop) | 1.6 mOhm (3%) | 1.6 mOhm (7%) |
| Bowel gas | 9.7 mOhm (96%) | 5.8 mOhm (93%) |
| Electrode drift (1 min) | 0.3 mOhm (0%) | 0.3 mOhm (0%) |
| **Total** | **9.8 mOhm** | **6.0 mOhm** |

Electronic noise is negligible after just 1 second of averaging (100 measurements at 50 kHz). The dominant noise source is bowel gas in both cases. SVD patterns reduce bowel gas sensitivity by ~40% through spatial focusing.

### Volume resolution

| Configuration | Resolution (1s) | Resolution (10s) | Meets 7 mL target? |
|---|---|---|---|
| 4-elec tetrapolar | 44 mL | 31 mL | No |
| 8-elec SVD rank-1 | 6.4 mL | 4.5 mL | **Yes** |
| 8-elec SVD rank-3 | 3.5 mL | 2.5 mL | **Yes** |

### Bottom line

| | 4-Electrode | 8-Electrode SVD |
|---|---|---|
| Hardware | Same belt | Same belt |
| Firmware | 1 drive pattern | 2-4 drive patterns |
| Sensitivity | 0.22 mOhm/mL | 0.94 mOhm/mL |
| Resolution | ~44 mL | ~6 mL |
| Clinical use | Not viable | **Meets 0.1 mL/kg/hr** |
| Complexity | Trivial | Low |

---

## 7. Practical Implementation

### Hardware

```
Belt:
  - 8 x Ag/AgCl gel electrodes (10 mm diameter, 0.785 cm^2)
  - 2 rings at z=9 cm and z=10 cm above pelvic floor
  - 4 electrodes per ring at 0, 90, 180, 270 degrees
  - Elastic strap with pre-positioned electrode pockets
  - All electrodes on anterior abdomen (no back electrodes needed)

Electronics:
  - AD5940 analog front-end (single chip)
  - 50 kHz excitation, 1 mA peak drive current
  - 8:1 analog multiplexer for electrode switching
  - Measurement rate: ~100 Hz (10 ms per drive pattern)
  - SVD patterns stored in firmware lookup table
```

### Firmware

```
Per measurement cycle (~20 ms):
  1. Set MUX to SVD drive pattern 1 → measure voltage at all sense pairs
  2. Set MUX to SVD drive pattern 2 → measure voltage at all sense pairs
  3. Apply SVD sense weights: V_combined = w1*V1 + w2*V2
  4. Store V_combined in circular buffer

Per processing cycle (every 100 ms):
  1. Apply band-stop filter (0.15-0.4 Hz IIR)
  2. Apply low-pass filter (< 0.08 Hz)
  3. Polynomial baseline removal (5-min sliding window)
  4. Volume_change = filtered_dZ / calibration_factor

Calibration (once per session):
  1. Patient breathes normally for 30 seconds
  2. Firmware extracts respiratory amplitude for quality check
  3. Known void event: record impedance before/after
  4. calibration_factor = dZ_void / void_volume (mOhm/mL)
```

### Expected performance

| Metric | Value |
|---|---|
| Sensitivity | 0.94 mOhm/mL |
| Volume resolution (1s) | ~6 mL |
| Volume resolution (10s) | ~4 mL |
| Minimum detectable change | ~5 mL (SNR > 1) |
| Clinical target (0.1 mL/kg/hr = 7 mL) | SNR ~1.1, detectable |
| Update rate | 0.1 Hz (every 10 seconds) |
| Belt application time | < 2 minutes |
| Battery life (AD5940, 1 mA drive) | > 24 hours on coin cell |

### Limitations

1. **Bowel gas** is the dominant noise source (93% of noise budget). Large gas pockets adjacent to the bladder can cause false readings of ~6-10 mL equivalent. Mitigation: use temporal averaging and flag rapid transients as likely artifact.

2. **Posture changes** produce impedance shifts of 2+ mOhm. Mitigation: accelerometer gating — pause volume tracking during movement, resume after 1-2 minutes of stabilization.

3. **Patient-specific calibration** is essential. Fat thickness varies from 0.5-4 cm across patients, changing sensitivity by ~2x. The void-event calibration accounts for this.

4. **The 1 mL target (0.015 mL/kg/hr) is NOT achievable with 8 electrodes.** This requires 16+ electrodes with SVD rank-3 and dual-frequency excitation.

---

## 8. Upgrading to 16 Electrodes

For applications requiring better than 6 mL resolution:

| Feature | 8-electrode | 16-electrode |
|---|---|---|
| Electrodes/ring | 4 | 8 |
| Rings | 2 | 2 |
| SVD patterns | rank-1 (2 seq.) | rank-3 (4 seq.) |
| Frequency | 50 kHz only | 10 kHz + 500 kHz |
| Sensitivity | 0.94 mOhm/mL | ~1.5 mOhm/mL (est.) |
| Respiratory rejection | Bandstop filter | Bandstop + dual-freq |
| Volume resolution | ~6 mL (1s) | ~1-2 mL (1s) |
| Target | 0.1 mL/kg/hr | 0.015 mL/kg/hr |

The 16-electrode upgrade adds dual-frequency excitation which becomes beneficial (rather than harmful) because SVD rank-3 compensates the sensitivity loss from alpha subtraction.

---

## References

- Gabriel C, Gabriel S, Corthout E. Phys. Med. Biol. 41:2231-2249, 1996. (Tissue conductivity data)
- Glass Clark SE, Nagle AS, et al. Female Pelvic Med Reconstr Surg. PMC6538469, 2020. (Bladder geometry)
- Oelke M, et al. Neurourol Urodyn. PMID: 16652381, 2006. (Bladder wall thickness)
- Leonhardt S, et al. Sensors 25(24):7635, 2025. (Bioimpedance bladder monitoring review)
- Schlebusch T, et al. Physiol. Meas. 35(9):1813-1823, 2014. (3D anatomical model sensitivity)
- Somersalo E, et al. Inverse Problems 8:919, 1992. (Complete Electrode Model)
