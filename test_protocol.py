#!/usr/bin/env python3
"""
Generate a lab bench protocol for validating bladder bioimpedance
simulation predictions using an AD5940 dev board.

Pulls quantitative predictions from the bladder_sim package and
produces protocol.txt with AD5940-specific configuration, electrode
placement, measurement procedures, and expected results.

Usage:
    python3 test_protocol.py
"""

import sys
import os
import numpy as np
from datetime import date

# Ensure bladder_sim is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bladder_sim.tissue_properties import (
    FREQ_TABLE_KHZ,
    CONDUCTIVITY_DB,
    CONTACT_IMPEDANCE_TABLE,
    get_conductivity,
    get_contact_impedance,
    measurement_noise_floor,
)
from bladder_sim.model import (
    TORSO_RX,
    TORSO_RY,
    BLADDER_BASE_Z,
    bladder_wall_thickness,
    bladder_semi_axes,
    bladder_center_y,
    SKIN_THICK,
    FAT_THICK,
    MUSCLE_THICK,
    DEFAULT_CURRENT_A,
)


# ── Simulation-derived predictions ──────────────────────────────────

def compute_predictions():
    """Compute all quantitative predictions from the simulation model."""
    pred = {}

    # Key frequencies for the AD5940 (max ~200 kHz)
    pred["freqs_kHz"] = np.array([1, 5, 10, 25, 30, 50, 100, 200])

    # Drive current
    pred["I_drive_A"] = DEFAULT_CURRENT_A  # 1 mA
    pred["I_drive_mA_pp"] = DEFAULT_CURRENT_A * 2 * 1e3  # 2 mA pp

    # Bladder geometry at key volumes
    volumes = np.array([50, 100, 150, 200, 250, 300, 350, 400, 450, 500])
    pred["volumes_mL"] = volumes
    bladder_dims = {}
    for vol in volumes:
        a, b, c = bladder_semi_axes(vol)
        bladder_dims[vol] = np.array([a, b, c]) * 2  # diameters in cm
    pred["bladder_dims"] = bladder_dims

    # Tissue conductivities at key frequencies
    tissues_of_interest = ["skin", "fat", "muscle", "urine", "bladder_wall",
                           "bone_avg", "background"]
    key_freqs = [10, 30, 50, 100, 200]
    tissue_table = {}
    for t in tissues_of_interest:
        tissue_table[t] = {f: get_conductivity(t, f) for f in key_freqs}
    pred["tissue_conductivity"] = tissue_table

    # Contact impedance at key frequencies
    pred["contact_impedance"] = {
        f: get_contact_impedance(f) for f in key_freqs
    }

    # Noise floor at key frequencies (1 mA drive, 0.785 cm^2 electrode)
    pred["noise_floor_ohm"] = {
        f: measurement_noise_floor(f, I_drive=DEFAULT_CURRENT_A)
        for f in key_freqs
    }

    # Sensitivity prediction: ~0.1-0.2 mOhm/mL from simulation
    # (best channel, anterior-posterior tetrapolar, 50 kHz)
    pred["sensitivity_mOhm_per_mL"] = {
        "nominal": 0.15,
        "range_low": 0.05,
        "range_high": 0.50,
    }

    # SNR peak frequency
    pred["snr_peak_kHz"] = 30.0
    pred["snr_broad_range_kHz"] = (25.0, 50.0)

    # Dual-frequency artifact cancellation
    # Simulation uses 10 + 500 kHz; AD5940 limited to 200 kHz
    pred["dual_freq"] = {
        "f1_kHz": 10.0,
        "f2_kHz": 200.0,
        "principle": "urine is freq-independent; muscle/tissue show beta dispersion",
    }

    # Torso geometry for electrode placement
    pred["torso_rx_cm"] = TORSO_RX
    pred["torso_ry_cm"] = TORSO_RY
    pred["skin_thick_cm"] = SKIN_THICK
    pred["fat_thick_cm"] = FAT_THICK
    pred["muscle_thick_cm"] = MUSCLE_THICK
    pred["bladder_center_anterior_cm"] = bladder_center_y(300)

    # Baseline impedance estimate (whole-body path, ~50 kHz)
    # Rough: muscle path ~20 cm at 0.35 S/m through ~100 cm^2 cross-section
    # Z_body ~ L / (sigma * A) ~ 20 / (0.35 * 100) = 0.57 Ohm
    # Plus skin/fat layers: total tetrapolar Z ~ 5-50 Ohm at 50 kHz
    pred["baseline_Z_range_ohm"] = (5.0, 50.0)

    return pred


# ── Protocol text generation ────────────────────────────────────────

def generate_protocol(pred):
    """Generate the full protocol text from simulation predictions."""

    lines = []

    def s(text=""):
        lines.append(text)

    def heading(text, char="="):
        s(text)
        s(char * len(text))
        s()

    def subheading(text):
        heading(text, "-")

    # ──────────────────────────────────────────────────────────────
    heading("BLADDER BIOIMPEDANCE VALIDATION PROTOCOL")
    s(f"Generated: {date.today().isoformat()}")
    s(f"Source:    bladder_sim FEM simulation package")
    s(f"Hardware:  Analog Devices AD5940 evaluation board")
    s()
    s("PURPOSE: Validate simulation predictions of bioimpedance-based")
    s("bladder volume estimation using a tetrapolar measurement on a")
    s("human subject with AD5940 hardware.")
    s()
    s("SUMMARY OF KEY PREDICTIONS:")
    s(f"  - Sensitivity:   ~{pred['sensitivity_mOhm_per_mL']['nominal']:.2f} mOhm/mL "
      f"(range {pred['sensitivity_mOhm_per_mL']['range_low']:.2f}"
      f"-{pred['sensitivity_mOhm_per_mL']['range_high']:.2f} mOhm/mL)")
    s(f"  - SNR peak:      ~{pred['snr_peak_kHz']:.0f} kHz "
      f"(broad {pred['snr_broad_range_kHz'][0]:.0f}-{pred['snr_broad_range_kHz'][1]:.0f} kHz)")
    s(f"  - Optimal axis:  Anterior-posterior (supra-pubic)")
    s(f"  - Artifact rejection: Dual-frequency subtraction "
      f"({pred['dual_freq']['f1_kHz']:.0f} + {pred['dual_freq']['f2_kHz']:.0f} kHz)")
    s()

    # ────────────────────────────────────────────────────────────
    # SECTION 1: EQUIPMENT SETUP
    # ────────────────────────────────────────────────────────────
    heading("1. EQUIPMENT SETUP")

    subheading("1.1 Required Equipment")
    s("  [ ] AD5940 evaluation board (EVAL-AD5940BIOZ or EVAL-AD5940ELCZ)")
    s("  [ ] USB cable + host PC with AD5940 software / UART terminal")
    s("  [ ] 4x Ag/AgCl gel electrodes (3M Red Dot 2560 or equivalent)")
    s("      - Contact area: ~3.14 cm^2 (20 mm dia) or ~0.785 cm^2 (10 mm dia)")
    s("  [ ] Abdominal skin prep: alcohol swabs, abrasive gel (NuPrep)")
    s("  [ ] Measuring tape + skin marker")
    s("  [ ] Portable bladder ultrasound (for volume reference)")
    s("      - BladderScan BVI 9400 or similar (accuracy +/- 15%)")
    s("  [ ] Precision resistors for calibration: 10, 47, 100, 470, 1k Ohm (1%)")
    s("  [ ] Timer / stopwatch")
    s("  [ ] Data logging PC (Python 3 + serial library)")
    s()

    subheading("1.2 AD5940 Configuration")
    s("Target: 4-wire (tetrapolar) bioimpedance, frequency sweep mode.")
    s()
    s("  Excitation:")
    s(f"    - Waveform:        Sinusoidal")
    s(f"    - Current:         {pred['I_drive_A']*1e3:.1f} mA peak "
      f"({pred['I_drive_mA_pp']:.1f} mA peak-to-peak)")
    s(f"    - Source:          HSDAC -> external V-to-I (or on-chip HSTIA loop)")
    s(f"    - Compliance:      Ensure HSDAC output range covers load impedance")
    s()
    s("  Frequency list (8 points for sweep):")
    for f in pred["freqs_kHz"]:
        nf = pred["noise_floor_ohm"].get(f, measurement_noise_floor(f))
        s(f"    {f:6.0f} kHz   (predicted noise floor: {nf*1e3:.3f} mOhm)")
    s()
    s("  ADC / DFT engine:")
    s("    - DFT points:      4096 (good frequency resolution)")
    s("    - ADC clock:       16 MHz (default)")
    s("    - Sinc3 filter:    Enable for noise rejection")
    s("    - Settling cycles: 16 (allow DDS + body to settle)")
    s()
    s("  PGA gain selection:")
    s("    - Start at 1.5x (avoids clipping on low-impedance body path)")
    s("    - If ADC under-ranges (signal < 25% full scale), increase to 3x or 4.5x")
    s("    - Expected sense voltage: ~5-50 mV (1 mA through 5-50 Ohm body)")
    s()
    s("  RTIA (trans-impedance resistor) selection:")
    s("    - For body impedance 5-50 Ohm with 1 mA drive:")
    s("      V_sense = 5-50 mV. RTIA converts current back to voltage.")
    s("    - Use RTIA = 1 kOhm for initial measurements")
    s("    - Adjust if output saturates or under-ranges")
    s()
    s("  Electrode connections (4-wire BioZ mode):")
    s("    - CE0 (Force+/Drive+)  ->  Anterior electrode (E1)")
    s("    - RE0 (Force-/Drive-)  ->  Posterior electrode (E2)")
    s("    - SE0 (Sense+)         ->  Anterior sense electrode (E3)")
    s("    - DE0 (Sense-)         ->  Posterior sense electrode (E4)")
    s()

    subheading("1.3 Calibration Procedure")
    s("BEFORE attaching to subject, verify accuracy with known resistors.")
    s()
    s("  1. Connect precision resistors in 4-wire config:")
    s("     CE0 --[R_test]-- RE0, with SE0 and DE0 at the junctions.")
    s()
    s("  2. Measure at each frequency in the sweep list.")
    s()
    s("  3. Expected calibration results (resistive load, no reactive component):")
    s("       R_test    |Z| expected    Phase expected")
    s("       -------   ------------   ---------------")
    s("       10 Ohm    10.0 Ohm       ~0 deg")
    s("       47 Ohm    47.0 Ohm       ~0 deg")
    s("       100 Ohm   100.0 Ohm      ~0 deg")
    s("       470 Ohm   470.0 Ohm      ~0 deg")
    s("       1k Ohm    1000.0 Ohm     ~0 deg")
    s()
    s("  4. Acceptance: |Z| within +/-1% of nominal across all frequencies.")
    s("     If >1% error, recalibrate RTIA gain or check connections.")
    s()
    s("  5. Record calibration factors (gain, offset) for post-processing.")
    s()

    # ────────────────────────────────────────────────────────────
    # SECTION 2: ELECTRODE PLACEMENT
    # ────────────────────────────────────────────────────────────
    heading("2. ELECTRODE PLACEMENT")

    subheading("2.1 Tetrapolar Configuration (4 electrodes)")
    s("The simulation predicts ANTERIOR-POSTERIOR drive axis through the")
    s("supra-pubic region gives optimal sensitivity to bladder volume.")
    s()
    s("Electrode layout: 2 anterior (abdomen) + 2 posterior (lower back),")
    s("arranged as a tetrapolar pair along the AP axis.")
    s()
    s("  Drive pair:  Current flows anterior -> posterior through pelvis")
    s("  Sense pair:  Voltage measured ~2 cm from drive, same AP axis")
    s()

    subheading("2.2 Placement Landmarks")
    s("Reference point: pubic symphysis (palpate upper edge).")
    s(f"Simulation bladder center: {pred['bladder_center_anterior_cm']:.0f} cm anterior "
      f"to body midline,")
    s(f"base at ~{BLADDER_BASE_Z:.0f} cm above pelvic floor.")
    s()
    s("  ANTERIOR electrodes (on abdomen):")
    s("    - Midline, ~2 cm superior to pubic symphysis (supra-pubic)")
    s("    - E1 (Drive+):  On midline at supra-pubic point")
    s("    - E3 (Sense+):  2 cm superior to E1 (still on midline)")
    s()
    s("  POSTERIOR electrodes (on lower back):")
    s("    - Midline, directly opposite the anterior pair")
    s("    - E2 (Drive-):  Opposite E1 (at sacrum level, midline)")
    s("    - E4 (Sense-):  2 cm superior to E2 (midline)")
    s()
    s("  Electrode spacing:")
    s("    - Drive-to-sense separation: 2 cm (center-to-center)")
    s("    - This ensures the sense electrodes are outside the")
    s("      high-current-density region near the drive electrodes.")
    s()

    subheading("2.3 Placement Diagram (axial cross-section)")
    s("  View: looking down at the abdomen (patient supine)")
    s()
    s("                     ANTERIOR (abdomen)")
    s("              ________________________________")
    s("             /          E3 (Sense+)           \\")
    s("            /           E1 (Drive+)            \\")
    s("           /              2 cm                   \\")
    s("          |               gap                     |")
    s("          |                                       |")
    s("   LEFT   |          [ BLADDER ]                  |   RIGHT")
    s("          |                                       |")
    s("          |                                       |")
    s("           \\              2 cm                   /")
    s("            \\           E2 (Drive-)            /")
    s("             \\__________E4_(Sense-)___________/")
    s("                    POSTERIOR (lower back)")
    s()
    s()
    s("  Sagittal view (side):")
    s()
    s("         Anterior                      Posterior")
    s("         (abdomen)                     (back)")
    s("           |                               |")
    s("    E3 ->  *  +2cm                         |")
    s("    E1 ->  *  supra-pubic      sacrum  *  <- E2")
    s("           |   ___________             *  <- E4  +2cm")
    s("           |  /  BLADDER  \\            |")
    s("           | |   (urine)   |           |")
    s("           |  \\___________/            |")
    s("           |                            |")
    s("    -------+------- pubic symphysis ----+-------")
    s()

    subheading("2.4 Tissue Path (from simulation geometry)")
    s("Current path from anterior to posterior electrode traverses:")
    s()
    s(f"  Layer           Thickness    Conductivity @ 50 kHz")
    s(f"  -------------   ---------    ---------------------")
    s(f"  Skin            {pred['skin_thick_cm']:.1f} cm       "
      f"{get_conductivity('skin', 50):.3f} S/m")
    s(f"  Subcut. fat     {pred['fat_thick_cm']:.1f} cm       "
      f"{get_conductivity('fat', 50):.3f} S/m")
    s(f"  Muscle          {pred['muscle_thick_cm']:.1f} cm       "
      f"{get_conductivity('muscle', 50):.3f} S/m")
    s(f"  Bladder wall    {bladder_wall_thickness(300)*10:.1f} mm (at 300mL)  "
      f"{get_conductivity('bladder_wall', 50):.3f} S/m")
    s(f"  Urine           variable     "
      f"{get_conductivity('urine', 50):.3f} S/m (freq-independent)")
    s(f"  Background      remainder    "
      f"{get_conductivity('background', 50):.3f} S/m")
    s()

    # ────────────────────────────────────────────────────────────
    # SECTION 3: PARTICIPANT PREPARATION
    # ────────────────────────────────────────────────────────────
    heading("3. PARTICIPANT PREPARATION")

    s("  3.1 Inclusion criteria:")
    s("    - Healthy adult (18-65 years)")
    s("    - BMI 18-30 (excessive adipose tissue degrades sensitivity)")
    s("    - No implanted electronic devices (pacemaker, neurostimulator)")
    s("    - No abdominal surgery in past 6 months")
    s("    - No urinary tract infection or bladder pathology")
    s()
    s("  3.2 Preparation steps:")
    s("    a) Hydration: Drink 500 mL water 1 hour before session.")
    s("       Goal: achieve comfortable bladder filling during experiment.")
    s()
    s("    b) Skin preparation at electrode sites:")
    s("       - Shave hair if present at electrode sites")
    s("       - Clean with alcohol swab, let dry")
    s("       - Apply NuPrep abrasive gel, rub gently for 10 sec per site")
    s("       - Wipe clean with dry gauze")
    s("       - Target: skin-electrode impedance < 5 kOhm at 50 Hz")
    s()
    s("    c) Position: Supine (lying on back) on exam table.")
    s("       Posterior electrodes applied before lying down.")
    s()
    s("    d) Allow 5 min settling after electrode application")
    s("       before starting measurements (gel hydration time).")
    s()
    s("  3.3 Contact impedance check:")
    s("    - Use AD5940 2-wire mode to measure each electrode impedance")
    s("    - Predicted contact impedance from simulation model:")
    for f in [10, 50, 100, 200]:
        zc = get_contact_impedance(f)
        s(f"        {f:4d} kHz:  {zc:.1f} Ohm*cm^2  "
          f"(~{zc/3.14:.0f} Ohm for 20mm dia electrode)")
    s("    - If any electrode > 500 Ohm at 50 kHz: re-prep that site")
    s()

    # ────────────────────────────────────────────────────────────
    # SECTION 4: MEASUREMENT PROCEDURES
    # ────────────────────────────────────────────────────────────
    heading("4. MEASUREMENT PROCEDURES")
    s("Three experiments in sequence. Total session: ~90-120 min.")
    s()

    # ── Experiment 1 ──
    subheading("4.1 Experiment 1: Volume Calibration Curve")
    s("GOAL: Measure Z vs bladder volume to validate linear sensitivity.")
    s()
    s("  Frequency:  50 kHz (near predicted SNR peak)")
    s(f"  Current:    {pred['I_drive_A']*1e3:.0f} mA peak")
    s("  Duration:   ~60-90 min (natural filling)")
    s()
    s("  Procedure:")
    s("    1. Empty bladder completely (void). Record time = T0.")
    s("    2. Measure ultrasound volume (should be ~0-30 mL residual).")
    s("    3. Record baseline impedance Z0 at 50 kHz (average 10 readings).")
    s("    4. Drink 500 mL water over 10 min.")
    s("    5. Every 15 min (or when urge increases):")
    s("       a) Record 10 impedance measurements at 50 kHz (1 reading/sec)")
    s("       b) Immediately measure bladder volume with ultrasound")
    s("       c) Record timestamp, avg(Z), std(Z), ultrasound volume")
    s("       d) Ask participant to rate urgency (0-10)")
    s("    6. Continue until strong urge to void (~400-600 mL).")
    s("    7. Final measurement, then allow participant to void.")
    s("    8. Post-void: measure Z again to confirm return to baseline.")
    s()
    s("  Expected results from simulation:")
    s()
    s("    Volume (mL)  Bladder diameter  Expected dZ from empty")
    s("    -----------  ----------------  ----------------------")
    for vol in [50, 100, 200, 300, 400, 500]:
        dims = pred["bladder_dims"][vol]
        dz_low = pred["sensitivity_mOhm_per_mL"]["range_low"] * vol
        dz_high = pred["sensitivity_mOhm_per_mL"]["range_high"] * vol
        dz_nom = pred["sensitivity_mOhm_per_mL"]["nominal"] * vol
        s(f"    {vol:4d} mL       "
          f"{dims[0]:.1f}x{dims[1]:.1f}x{dims[2]:.1f} cm    "
          f"{dz_nom:+6.1f} mOhm (range {dz_low:.1f}-{dz_high:.1f})")
    s()
    s(f"    Linear sensitivity: ~{pred['sensitivity_mOhm_per_mL']['nominal']:.2f} mOhm/mL")
    s(f"    Tolerance band:     {pred['sensitivity_mOhm_per_mL']['range_low']:.2f}"
      f"-{pred['sensitivity_mOhm_per_mL']['range_high']:.2f} mOhm/mL")
    s()
    s("  Validation criteria:")
    s("    [PASS] Linear fit R^2 > 0.8 for Z vs volume")
    s("    [PASS] Slope magnitude within 0.05-0.50 mOhm/mL")
    s("    [PASS] Post-void Z returns within 10% of pre-fill baseline")
    s()

    # ── Experiment 2 ──
    subheading("4.2 Experiment 2: Frequency Sweep (SNR vs Frequency)")
    s("GOAL: Validate that SNR peaks near 25-50 kHz as predicted.")
    s()
    s("  Bladder state: Fixed volume ~300 mL (confirm with ultrasound)")
    s("  Sweep:         1, 5, 10, 25, 30, 50, 100, 200 kHz")
    s(f"  Current:       {pred['I_drive_A']*1e3:.0f} mA peak at each frequency")
    s()
    s("  Procedure:")
    s("    1. Wait until bladder is ~300 mL (from Experiment 1 filling,")
    s("       or have participant drink and wait).")
    s("    2. Confirm volume with ultrasound.")
    s("    3. At each frequency in the sweep list:")
    s("       a) Configure AD5940 DDS to target frequency")
    s("       b) Wait 1 sec for settling")
    s("       c) Record 20 consecutive |Z| and phase measurements")
    s("       d) Compute mean(|Z|), std(|Z|), mean(phase)")
    s("    4. Participant voids ~50 mL (timed void or partial).")
    s("       If partial void not possible, compare to a different-volume")
    s("       measurement from Experiment 1 at the same frequencies.")
    s("    5. Repeat sweep at new volume.")
    s("    6. Compute dZ = Z(300 mL) - Z(250 mL) at each frequency.")
    s()
    s("  Expected results (simulation predictions):")
    s()
    s("    Freq (kHz)  Noise floor    Contact Z      Predicted SNR/mL")
    s("    ----------  -----------    ---------       ----------------")
    for f in pred["freqs_kHz"]:
        nf = measurement_noise_floor(f) * 1e3  # mOhm
        zc = get_contact_impedance(f)
        # SNR is sensitivity / noise
        sens = pred["sensitivity_mOhm_per_mL"]["nominal"]  # mOhm/mL
        snr = sens / nf if nf > 0 else 0
        s(f"    {f:6.0f}       {nf:7.3f} mOhm   {zc:6.1f} Ohm*cm^2   {snr:6.1f}")
    s()
    s(f"    Peak SNR expected near {pred['snr_peak_kHz']:.0f} kHz "
      f"(broad optimum {pred['snr_broad_range_kHz'][0]:.0f}-{pred['snr_broad_range_kHz'][1]:.0f} kHz)")
    s()
    s("  Validation criteria:")
    s("    [PASS] SNR (|dZ|/std) highest in 10-100 kHz band")
    s("    [PASS] Contact impedance decreases with frequency (beta dispersion)")
    s("    [PASS] |Z| at each frequency stable (CV < 1%) over 20 readings")
    s()

    # ── Experiment 3 ──
    subheading("4.3 Experiment 3: Respiratory Artifact Characterization")
    s("GOAL: Demonstrate dual-frequency subtraction cancels breathing artifact.")
    s()
    s(f"  Frequencies:  f1 = {pred['dual_freq']['f1_kHz']:.0f} kHz, "
      f"f2 = {pred['dual_freq']['f2_kHz']:.0f} kHz")
    s(f"  Bladder:      Fixed ~300 mL")
    s(f"  Current:      {pred['I_drive_A']*1e3:.0f} mA peak at each frequency")
    s()
    s("  Principle:")
    s(f"    {pred['dual_freq']['principle']}")
    s("    Respiratory motion changes muscle impedance (~5% modulation).")
    s("    At f1 (low freq), muscle conductivity is lower -> larger artifact.")
    s("    At f2 (high freq), muscle conductivity is higher -> smaller artifact.")
    s("    Subtracting alpha * Z(f2) from Z(f1) cancels the common-mode")
    s("    respiratory component while preserving bladder signal.")
    s()
    s("  Procedure:")
    s("    1. Confirm bladder ~300 mL with ultrasound.")
    s("    2. BREATH-HOLD measurement (artifact-free baseline):")
    s(f"       a) Configure AD5940 to {pred['dual_freq']['f1_kHz']:.0f} kHz")
    s("       b) Ask participant to hold breath at end-expiration")
    s("       c) Record 10 readings during breath-hold (~10 sec)")
    s(f"       d) Switch to {pred['dual_freq']['f2_kHz']:.0f} kHz, repeat breath-hold + 10 readings")
    s("    3. NORMAL BREATHING measurement (with artifact):")
    s(f"       a) Configure AD5940 to {pred['dual_freq']['f1_kHz']:.0f} kHz")
    s("       b) Record 60 readings over 60 sec (normal breathing)")
    s(f"       c) Switch to {pred['dual_freq']['f2_kHz']:.0f} kHz, record 60 readings over 60 sec")
    s("    4. RAPID ALTERNATION (if AD5940 supports fast freq switching):")
    s(f"       a) Alternate between {pred['dual_freq']['f1_kHz']:.0f} and "
      f"{pred['dual_freq']['f2_kHz']:.0f} kHz every 0.5 sec")
    s("       b) Record 120 readings (60 per frequency) over 60 sec")
    s("       c) This interleaves the two frequencies within each breath cycle")
    s("    5. Repeat steps 2-4 after participant voids ~50 mL.")
    s()
    s("  Tissue conductivity spectra (from simulation model):")
    s()
    s("    Tissue         sigma @ 10 kHz    sigma @ 200 kHz    Ratio")
    s("    ------         --------------    ---------------    -----")
    for tissue, label in [("urine", "Urine"), ("muscle", "Muscle"),
                          ("fat", "Fat"), ("skin", "Skin"),
                          ("bladder_wall", "Bladder wall")]:
        s10 = get_conductivity(tissue, 10)
        s200 = get_conductivity(tissue, 200)
        ratio = s200 / s10 if s10 > 0 else 0
        s(f"    {label:14s} {s10:8.3f} S/m       {s200:8.3f} S/m       {ratio:.2f}x")
    s()
    s("    Note: Urine ratio ~1.00x (frequency-independent) while")
    s("    muscle ratio >1.5x. This difference enables separation.")
    s()
    s("  Analysis (post-processing):")
    s("    1. Compute respiratory artifact amplitude at each frequency:")
    s("       A_resp(f) = std(Z_breathing(f)) - std(Z_breathhold(f))")
    s("    2. Compute alpha = A_resp(f1) / A_resp(f2)")
    s("    3. Isolated signal: Z_iso = Z(f1) - alpha * Z(f2)")
    s("    4. Artifact rejection ratio = A_resp_single / std(Z_iso)")
    s()
    s("  Validation criteria:")
    s("    [PASS] Breathing artifact visible in raw Z(f1) and Z(f2)")
    s("          (std during breathing > 2x std during breath-hold)")
    s("    [PASS] Z_iso shows reduced respiratory modulation")
    s("          (artifact rejection > 3x)")
    s("    [PASS] dZ_iso between 300 and 250 mL is non-zero")
    s("          (bladder signal preserved after subtraction)")
    s()

    # ────────────────────────────────────────────────────────────
    # SECTION 5: DATA RECORDING
    # ────────────────────────────────────────────────────────────
    heading("5. DATA RECORDING")

    subheading("5.1 CSV Format")
    s("All measurements saved to timestamped CSV files.")
    s()
    s("  Filename pattern: bladder_bioz_YYYYMMDD_HHMMSS_expN.csv")
    s()
    s("  Columns:")
    s("    timestamp_iso    ISO 8601 timestamp (2024-01-15T10:30:00.123)")
    s("    experiment       Experiment number (1, 2, or 3)")
    s("    freq_kHz         Excitation frequency in kHz")
    s("    Z_real_ohm       Real part of measured impedance (Ohm)")
    s("    Z_imag_ohm       Imaginary part of measured impedance (Ohm)")
    s("    Z_mag_ohm        |Z| magnitude (Ohm)")
    s("    Z_phase_deg      Phase angle (degrees)")
    s("    I_drive_mA       Drive current amplitude (mA)")
    s("    us_volume_mL     Ultrasound-measured bladder volume (mL)")
    s("    breath_state     'hold' or 'normal' (Experiment 3)")
    s("    urgency_score    Subjective urgency 0-10 (Experiment 1)")
    s("    notes            Free-text notes")
    s()
    s("  Example row:")
    s("    2024-01-15T10:30:00.123,1,50.0,23.456,-1.234,23.488,-3.01,1.0,287,normal,5,")
    s()

    subheading("5.2 Measurement Logging Settings")
    s("  AD5940 output:    UART at 115200 baud, format: real,imag,mag,phase\\n")
    s("  Sample rate:      1 Hz per frequency point (DFT + settling time)")
    s("  Burst mode:       10-20 readings per measurement point")
    s("  Ultrasound log:   Manual entry after each impedance burst")
    s()

    # ────────────────────────────────────────────────────────────
    # SECTION 6: ANALYSIS
    # ────────────────────────────────────────────────────────────
    heading("6. DATA ANALYSIS")

    subheading("6.1 Volume Sensitivity (Experiment 1)")
    s("  1. Extract mean |Z| at each ultrasound-measured volume.")
    s("  2. Linear regression: Z = a * V + b")
    s("     - slope a = dZ/dV in Ohm/mL")
    s("     - Convert to mOhm/mL: multiply by 1000")
    s(f"     - Expected: {pred['sensitivity_mOhm_per_mL']['range_low']:.2f}"
      f"-{pred['sensitivity_mOhm_per_mL']['range_high']:.2f} mOhm/mL")
    s("  3. Compute R^2 of linear fit.")
    s("  4. Compute measurement noise: mean of per-burst std(|Z|).")
    s("  5. Minimum detectable volume change: dV_min = 3 * noise / |slope|")
    s()
    s("  Python snippet:")
    s("    import numpy as np")
    s("    volumes = data['us_volume_mL'].values")
    s("    Z_means = data.groupby('us_volume_mL')['Z_mag_ohm'].mean().values")
    s("    coeffs = np.polyfit(volumes, Z_means, 1)")
    s("    dZ_dV_mOhm = coeffs[0] * 1e3  # mOhm/mL")
    s("    R2 = 1 - np.sum((Z_means - np.polyval(coeffs, volumes))**2) / \\")
    s("             np.sum((Z_means - np.mean(Z_means))**2)")
    s()

    subheading("6.2 SNR vs Frequency (Experiment 2)")
    s("  1. At each frequency, compute:")
    s("     signal = |Z(V_high) - Z(V_low)| / (V_high - V_low)  [Ohm/mL]")
    s("     noise  = mean(std(Z_burst))  [Ohm]")
    s("     SNR    = signal / noise  [per mL]")
    s("  2. Plot SNR vs frequency (log x-axis).")
    s(f"  3. Identify peak SNR frequency. Expected: {pred['snr_peak_kHz']:.0f} kHz "
      f"(+/- 1 octave).")
    s("  4. Compare shape to simulation prediction.")
    s()

    subheading("6.3 Artifact Rejection (Experiment 3)")
    s("  1. Compute respiratory artifact power at each frequency:")
    s("     P_resp(f) = var(Z_breathing(f)) - var(Z_breathhold(f))")
    s("  2. Compute optimal subtraction weight:")
    s("     alpha = P_resp(f1) / P_resp(f2)")
    s("  3. Form isolated signal: Z_iso(t) = Z(f1, t) - alpha * Z(f2, t)")
    s("  4. Artifact rejection ratio:")
    s("     AR = std(Z_breathing(f1)) / std(Z_iso)")
    s("  5. Verify bladder sensitivity preserved:")
    s("     dZ_iso = Z_iso(V_high) - Z_iso(V_low)")
    s("     Compare to single-frequency dZ(f1).")
    s()

    # ────────────────────────────────────────────────────────────
    # SECTION 7: EXPECTED RESULTS
    # ────────────────────────────────────────────────────────────
    heading("7. EXPECTED RESULTS (from simulation)")

    subheading("7.1 Baseline Impedance")
    s(f"  Tetrapolar |Z| at 50 kHz: {pred['baseline_Z_range_ohm'][0]:.0f}"
      f"-{pred['baseline_Z_range_ohm'][1]:.0f} Ohm")
    s("  Phase angle: -5 to -15 degrees (slightly capacitive due to")
    s("  cell membrane reactance in tissue path)")
    s()

    subheading("7.2 Sensitivity Predictions")
    s(f"  Best-channel |dZ/dV|:   {pred['sensitivity_mOhm_per_mL']['nominal']:.2f} mOhm/mL "
      f"(nominal)")
    s(f"  Tolerance band:         {pred['sensitivity_mOhm_per_mL']['range_low']:.2f}"
      f" - {pred['sensitivity_mOhm_per_mL']['range_high']:.2f} mOhm/mL")
    s()
    s("  Sources of variability:")
    s("    - Body composition (BMI, fat thickness):  +/- 50%")
    s("    - Electrode placement accuracy:           +/- 30%")
    s("    - Bladder position variability:           +/- 20%")
    s("    - Urine conductivity (hydration-dependent): +/- 10%")
    s()
    s("  Total impedance change over 0-500 mL:")
    s(f"    Nominal: {pred['sensitivity_mOhm_per_mL']['nominal'] * 500:.0f} mOhm "
      f"({pred['sensitivity_mOhm_per_mL']['nominal'] * 500 / 1000:.3f} Ohm)")
    s(f"    Range:   {pred['sensitivity_mOhm_per_mL']['range_low'] * 500:.0f}"
      f"-{pred['sensitivity_mOhm_per_mL']['range_high'] * 500:.0f} mOhm")
    s()

    subheading("7.3 Frequency Response")
    s("  Noise floor (measurement system, from simulation noise model):")
    s()
    s("    Freq (kHz)  Noise (mOhm)  SNR/mL  Contact Z (Ohm*cm^2)")
    s("    ----------  -----------   ------  --------------------")
    for f in pred["freqs_kHz"]:
        nf = measurement_noise_floor(f)
        zc = get_contact_impedance(f)
        sens = pred["sensitivity_mOhm_per_mL"]["nominal"] * 1e-3  # Ohm/mL
        snr = sens / nf if nf > 0 else 0
        s(f"    {f:6.0f}       {nf*1e3:7.3f}       {snr:5.1f}   {zc:6.1f}")
    s()
    s("  Key trends:")
    s("    - Below 10 kHz: electrode polarization noise dominates")
    s("    - 25-50 kHz: optimal SNR (noise drops, sensitivity maintained)")
    s("    - Above 100 kHz: sensitivity decreases as tissue contrast drops")
    s()

    subheading("7.4 Respiratory Artifact Predictions")
    s("  Respiratory motion causes ~5% modulation of abdominal muscle")
    s("  impedance (volume conductor geometry changes during breathing).")
    s()
    s("  At single frequency (50 kHz):")
    s("    Artifact amplitude: ~1-5 mOhm (comparable to bladder signal)")
    s("    This is the primary confound for single-frequency measurement.")
    s()
    s("  After dual-frequency subtraction:")
    s("    Artifact rejection: >10x (simulation prediction)")
    s("    Residual artifact: <0.5 mOhm")
    s("    Bladder signal preserved: >70% of single-frequency value")
    s()

    subheading("7.5 Tissue Conductivity Reference Table")
    s("  From Gabriel et al. 1996, interpolated at key frequencies:")
    s()
    header = "    Tissue          "
    for f in [10, 50, 100, 200]:
        header += f"  {f:>4d} kHz"
    s(header)
    s("    " + "-" * (len(header) - 4))
    for tissue, label in [
        ("skin", "Skin"),
        ("fat", "Subcut. fat"),
        ("muscle", "Muscle"),
        ("bladder_wall", "Bladder wall"),
        ("urine", "Urine"),
        ("bone_avg", "Bone (avg)"),
        ("background", "Background"),
        ("blood", "Blood"),
    ]:
        row = f"    {label:16s}"
        for f in [10, 50, 100, 200]:
            row += f"  {get_conductivity(tissue, f):8.3f}"
        s(row + "  S/m")
    s()

    # ────────────────────────────────────────────────────────────
    # SECTION 8: SAFETY
    # ────────────────────────────────────────────────────────────
    heading("8. SAFETY CONSIDERATIONS")

    subheading("8.1 Current Limits (IEC 60601-1)")
    s("  The IEC 60601-1 standard for medical electrical equipment specifies:")
    s()
    s("    - Patient auxiliary current limit:  < 10 uA DC")
    s("    - Patient auxiliary current limit:  < 100 uA AC (< 1 kHz)")
    s("    - Patient auxiliary current limit:  < 10 mA AC (1 kHz - 100 kHz)")
    s("    - Above 100 kHz: limit scales as 10 mA * (f_kHz / 100)")
    s()
    s(f"  This protocol uses {pred['I_drive_A']*1e3:.0f} mA peak at frequencies "
      f"1-200 kHz.")
    s(f"  At {pred['I_drive_A']*1e3:.0f} mA, we are well below the 10 mA limit for all frequencies.")
    s()
    s("  Current density at electrode (worst case):")
    s("    1 mA / 0.785 cm^2 (10 mm dia electrode) = 1.27 mA/cm^2")
    s("    IEC 60601-1 limit for prolonged contact: < 2 mA/cm^2")
    s("    STATUS: Compliant.")
    s()

    subheading("8.2 Exclusion Criteria")
    s("  STOP measurement and exclude participant if:")
    s("    - Participant reports pain, tingling, or discomfort at electrode sites")
    s("    - Skin shows redness, irritation, or burns at electrode sites")
    s("    - Participant has undisclosed implanted electronic device")
    s("    - Equipment malfunction (unexpected current readings)")
    s("    - Participant requests to stop at any time")
    s()

    subheading("8.3 Emergency Procedures")
    s("    - Immediately disconnect electrodes if participant reports pain")
    s("    - AD5940 has hardware current limit (HSDAC compliance)")
    s("    - Keep first aid kit accessible")
    s("    - Ensure participant can void bladder immediately if needed")
    s()

    subheading("8.4 Ethical Considerations")
    s("    - Obtain informed consent before any measurements")
    s("    - IRB/ethics approval required for human-subject research")
    s("    - Participant can withdraw at any time without justification")
    s("    - Data stored with anonymized subject ID only")
    s("    - Measurements are non-invasive (skin-surface electrodes only)")
    s()

    # ────────────────────────────────────────────────────────────
    # APPENDIX: AD5940 REGISTER QUICK REFERENCE
    # ────────────────────────────────────────────────────────────
    heading("APPENDIX A: AD5940 CONFIGURATION QUICK REFERENCE")
    s()
    s("  Key API calls (AD5940 library / examples):")
    s()
    s("  // 1. Initialize")
    s("  AD5940_Initialize();")
    s("  AD5940_StructInit(&AppBIOZCfg, sizeof(AppBIOZCfg));")
    s()
    s("  // 2. BioZ application config")
    s("  AppBIOZCfg.SeqStartAddr        = 0;")
    s("  AppBIOZCfg.MaxSeqLen            = 512;")
    s("  AppBIOZCfg.RcalVal              = 10000.0;  // RCAL resistor (Ohm)")
    s()
    s("  // 3. Excitation")
    s("  AppBIOZCfg.SinFreq              = 50000.0;  // Start freq (Hz)")
    s("  AppBIOZCfg.DacVoltPP            = 800;      // mV pp (into RTIA)")
    s(f"  // Target {pred['I_drive_A']*1e3:.0f} mA through body -> "
      f"adjust DAC voltage based on load")
    s()
    s("  // 4. ADC + DFT")
    s("  AppBIOZCfg.ADCSinc3Osr          = ADCSINC3OSR_4;")
    s("  AppBIOZCfg.ADCSinc2Osr          = ADCSINC2OSR_22;")
    s("  AppBIOZCfg.DftNum               = DFTNUM_4096;")
    s("  AppBIOZCfg.DftSrc               = DFTSRC_SINC3;")
    s("  AppBIOZCfg.HanWinEn             = bTRUE;    // Hanning window")
    s()
    s("  // 5. HSTIA (sense path)")
    s("  AppBIOZCfg.HsTiaCtia            = 31;       // Internal cap")
    s("  AppBIOZCfg.HsTiaRtiaSel         = HSTIARTIA_1K;  // 1 kOhm RTIA")
    s()
    s("  // 6. Switch matrix for 4-wire")
    s("  // CE0 = Force+, RE0 = Force-, SE0 = Sense+, DE0 = Sense-")
    s("  // Configure SWMatrix for tetrapolar measurement")
    s()
    s("  // 7. Frequency sweep loop")
    s("  for each freq in [1000, 5000, 10000, 25000, 30000, 50000, 100000, 200000]:")
    s("      AppBIOZCfg.SinFreq = freq;")
    s("      AD5940_BIOZCtrl(BIOZCTRL_START, 0);")
    s("      // Wait for DFT interrupt")
    s("      // Read DFTREAL and DFTIMAG registers")
    s("      // Z = (DFTREAL + j*DFTIMAG) * calibration_factor")
    s()

    heading("APPENDIX B: TROUBLESHOOTING")
    s()
    s("  Problem                          Likely Cause              Fix")
    s("  ----------------------------     ----------------------    ------------------")
    s("  |Z| much higher than expected    Poor electrode contact    Re-prep skin, check gel")
    s("  |Z| drifts continuously          Gel drying / sweat        Re-apply electrodes")
    s("  Large 50/60 Hz artifact          Mains interference        Shield cables, use notch")
    s("  Phase near -90 degrees           Capacitive-dominated      Check electrode connection")
    s("  No volume sensitivity            Electrodes off-target     Re-check placement landmarks")
    s("  ADC saturating                   PGA gain too high         Reduce to 1x")
    s("  Signal too small                 PGA gain too low          Increase PGA gain")
    s("  Freq sweep inconsistent          RTIA saturating           Switch RTIA range per freq")
    s()

    heading("APPENDIX C: BLADDER VOLUME REFERENCE (simulation geometry)")
    s()
    s("  Vol (mL)  Lateral (cm)  AP (cm)   SI (cm)   Approx diameter")
    s("  --------  -----------   --------  --------  ----------------")
    for vol in pred["volumes_mL"]:
        dims = pred["bladder_dims"][vol]
        avg_d = np.mean(dims)
        s(f"  {vol:5.0f}     {dims[0]:6.1f}        {dims[1]:6.1f}    {dims[2]:6.1f}     ~{avg_d:.1f} cm avg")
    s()
    s(f"  Bladder wall thickness: {bladder_wall_thickness(50)*10:.1f} mm (empty) to "
      f"{bladder_wall_thickness(500)*10:.1f} mm (500 mL)")
    s(f"  Urine conductivity:     {get_conductivity('urine', 50):.2f} S/m (freq-independent)")
    s(f"  Wall conductivity:      {get_conductivity('bladder_wall', 50):.3f} S/m at 50 kHz")
    s()
    s("--- END OF PROTOCOL ---")

    return "\n".join(lines)


# ── Main ────────────────────────────────────────────────────────────

def main():
    print("Computing simulation predictions...")
    pred = compute_predictions()

    print("Generating protocol document...")
    protocol_text = generate_protocol(pred)

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "protocol.txt")
    with open(output_path, "w") as f:
        f.write(protocol_text)

    n_lines = protocol_text.count("\n") + 1
    print(f"\nWritten: {output_path}")
    print(f"  {n_lines} lines, {len(protocol_text)} bytes")

    # Quick validation
    checks = [
        ("AD5940" in protocol_text, "Contains AD5940 references"),
        ("mOhm/mL" in protocol_text, "Contains sensitivity units"),
        ("IEC 60601" in protocol_text, "Contains safety standards"),
        ("kHz" in protocol_text, "Contains frequency units"),
        ("tetrapolar" in protocol_text, "Contains tetrapolar config"),
        ("S/m" in protocol_text, "Contains conductivity units"),
        ("Ohm*cm^2" in protocol_text, "Contains contact impedance units"),
        ("HSDAC" in protocol_text, "Contains AD5940-specific registers"),
    ]
    print("\nValidation:")
    all_pass = True
    for check, desc in checks:
        status = "PASS" if check else "FAIL"
        if not check:
            all_pass = False
        print(f"  [{status}] {desc}")

    if all_pass:
        print("\nAll checks passed.")
    else:
        print("\nSome checks failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
