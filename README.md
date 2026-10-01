# Guided-wave damage detection under temperature variation (Open Guided Waves, dataset 2)

Version française : [README.fr.md](README.fr.md)

Personal project: detect damage on an instrumented composite plate despite
temperature variations from 20 to 60 °C, with three approaches compared
without data leakage (a classical baseline method, a supervised convolutional
network, and anomaly detection trained on healthy measurements only), then
compress and export the models for embedded targets (ONNX, int8
quantization, C++ generation with Eclipse Aidge).

**Main result.** A supervised convolutional network trained on three damage
positions **does not detect** a fourth, never-seen position (AUC from 0.25
to 1.00 depending on the position). A principal component analysis with 5
components, trained on only 79 **healthy** measurements, detects damage at
all four positions (AUC 1.000), generalizes to a temperature range absent
from training, and fits in 337 KB with int8 weights, versus 34.85 MB for the
measurement library of the baseline method.

## Data
Dataset 2 of the Open Guided Waves platform: Moll J., Kexel C., Pötzsch S.,
Rennoch M., Herrmann A. S., *Temperature affected guided wave propagation in a
composite plate complementing the Open Guided Waves Platform*, Scientific Data
6, 191 (2019), <https://doi.org/10.1038/s41597-019-0208-1>. Data:
<https://doi.org/10.6084/m9.figshare.c.4488089.v1> (CC0 license, about 160 GB,
not included in this repository).

- CFRP plate of 500 × 500 × 2 mm, 12 piezoelectric transducers, hence
  **66 transmitter-receiver paths** (numbered 0 to 11 as in the HDF5 file).
- Reversible damage: an aluminium disc attached to the surface with a
  repositionable adhesive (according to the publication), at 4 positions
  (D04, D12, D16, D24).
- **966 measurements**: 322 healthy (two cycles of 161) and 161 per damage
  position. Each cycle: 20 → 60 → 20 °C in steps of 0.5 °C (setpoint).
- 12 excitation frequencies (40 to 260 kHz), 5-cycle tone bursts; signals of 13,108
  points at 10 MHz (1.31 ms).
- As stated in the publication, the effect of temperature on the signal
  exceeds that of damage: this is the central difficulty of the problem.

## Data and preprocessing
1. **Centering** of each signal (DC offset of 7 % of the RMS at 40 kHz).
2. **Decimation** by 16 with an anti-aliasing filter: 625 kHz, **820 points**.
3. **Band-pass** [f/2, 2f], i.e. [20, 80] kHz.
4. **Per-path scaling**, estimated on the training measurements only.
5. **Quality control**: 11 (measurement, path) pairs are nearly empty (all
   on path 8-9); they are treated as sensor failures, not as damage.

**Temperature.** The chamber setpoint is not the plate temperature: the mean
offset depends on the acquisition run (+1.11 °C for the first healthy cycle,
+0.55 °C for the second, +0.26 to +0.34 °C for the damaged runs). All
temperature bands refer to the temperature measured on the plate.

## Frequency choice
Ratio between the damage residual and the variability between two healthy
cycles (centered and filtered signals, baseline matched on the measured
temperature; `tools/frequency_contrast.py`, full table in
`results/step01_exploration/frequency_contrast.md`):

| kHz | 40 | 60 | 80 | 100 | 120 | 200 | 260 |
|---|---|---|---|---|---|---|---|
| median over paths | **2.28** | 1.72 | 1.39 | 1.31 | 1.32 | 1.46 | 1.32 |
| maximum over paths | **9.01** | 6.50 | 5.29 | 4.03 | 2.23 | 3.57 | 4.27 |

Selected frequency: **40 kHz**.

## Protocol: splits without data leakage
The unit is the **measurement** (its 66 paths always stay together).
- **Main, by position (4 folds)**: test = one never-seen damage position +
  the second healthy cycle (161 + 161); training (580 measurements) and
  validation (64, band 38 to 42 °C) on the three other positions and the
  first healthy cycle.
- **By temperature**: test = band 48 to 52 °C, all classes.
- **Drift control**: first versus second healthy cycle (all healthy
  measurements precede the damaged ones; the model must not learn the
  acquisition date).
- Normalization, baselines and thresholds learned on training only;
  configuration frozen on validation; **test opened only once per method**.
  One exception: the acquisition quality check of the baseline (Step 2) was
  added after examining two false alarms of the test set (healthy cycle 2,
  sensor fault on path 8-9). The same fault is present in the training data,
  so a check run on the training data alone would also have revealed it.
- **Nested validation** for the supervised network (one training position
  held out, external test set aside): it predicted the failure before the
  test was opened.

## Results (test AUC)
| Split | Baseline (no training) | Supervised CNN (5 seeds) | PCA, k = 5 (healthy only) |
|---|---|---|---|
| never-seen position D04 | 1.000 | 1.000 ± 0.000 | 1.000 |
| never-seen position D12 | 1.000 | 0.581 ± 0.113 | 1.000 |
| never-seen position D16 | 1.000 | 0.247 ± 0.073 | 1.000 |
| never-seen position D24 | 1.000 | 0.446 ± 0.122 | 1.000 |
| never-seen temperature | 0.851 | 1.000 ± 0.000 | 1.000 |
| drift control | | 0.555 ± 0.124 | |

- **Baseline**: for each measurement, the most similar healthy measurement of
  the library (optimal baseline selection), residual normalized per path,
  maximum over paths. At the threshold calibrated without the test set: 100 %
  detection, 1.2 % false alarms on the positions, but **64.7 %** on the
  never-seen temperature (no baseline at these temperatures).
- **Supervised CNN** (65,601 parameters, 4 Conv1d/BatchNorm/ReLU/MaxPool
  blocks, global max): it recognizes the positions seen in training; on D16,
  it classifies the damage as clearly healthy (median logit −9.5, versus
  +8.8 for the seen positions). Regularization changes nothing (nested
  validation 0.647 versus 0.644).
- **PCA**: projection onto 5 components learned on 79 healthy measurements
  (heating ramp of the first cycle, minus 2 measurements with an empty path);
  indicator = projection residual, maximum over paths. At the calibrated
  threshold: 100 % detection, 7.5 % false alarms (all between 20.6 and
  24.0 °C, at the cold edge of the learned domain), 0 % on the never-seen
  temperature.

## Damage localization
With 4 aligned damage positions, a supervised model cannot learn to locate a
new position: ridge regression or a multilayer perceptron trained on 3
positions and evaluated on the 4th gives a median error of 184 to 193 mm
(except the intermediate position D16, interpolated at 24 mm). Localization
therefore comes from the geometry (coordinates from tables 1 and 2 of the
publication), using the residuals of the healthy model (PCA), without any
damage position to tune the parameters:

| Median error (mm) | D04 | D12 | D16 | D24 | all |
|---|---|---|---|---|---|
| RAPID (amplitude per path) | 369 | 24 | 24 | 133 | 128 |
| Delay and sum (time of flight, v = 1.284 mm/µs calibrated on healthy measurements) | 45 | 16 | 22 | 46 | 40 |

RAPID cannot place damage along an edge path (D04, D24); time of flight can.
The estimates at the edges remain biased outwards (cause not established;
the velocity is not the cause, ±5 % gives 38 to 55 mm).

## Other frequencies
Same protocol at the 12 frequencies (40 to 260 kHz): the pipeline only works
at low frequency. PCA detection: AUC 1.000 at 40 kHz, 0.996 to 1.000 at
60 kHz, close to chance above 100 kHz (healthy residual floor insensitive to
the number of components: a signal-to-noise ratio limit). Localization:
40 mm at 40 kHz, 194 mm or more elsewhere (the first-arrival velocity
calibration picks up another mode above 40 kHz; at 60 kHz the fit fails
outright, with a negative velocity of -5.99 mm/µs, and at 260 kHz t0 is
negative). Fusing the 12 frequencies degrades localization (205 mm). On the
temperature split, the PCA AUC is erratic above 60 kHz (0.28 at 140 kHz,
0.99 at 200 kHz), without a consistent trend. The choice of 40 kHz,
initially made with the damage at all 4 positions, was checked by a
leakage-free selection (validation of each fold only): 40 kHz has the best
validation AUC in 3 folds; in the D16 fold, 40 and 60 kHz tie (validation
AUC 1.000). Details: `results/step07_frequencies/`.

## Deep autoencoders
Same protocol as PCA (training on the 79 healthy measurements, latent size
chosen on validation, 3 seeds). A dense autoencoder (6,981,866 parameters,
27.93 MB) matches PCA on the positions (AUC 1.000); a compact convolutional
autoencoder (85,900 parameters) detects less well (AUC from 0.763 to 1.000;
not converged after 3,000 epochs). On the never-seen temperature, their
thresholds transfer poorly (42 % and 100 % false alarms, versus 0 % for
PCA). PCA, a linear autoencoder, remains the best trade-off.

## Embedded: ONNX and quantization
| Model | ONNX file | Latency (1 thread, batch of 1) | Detection |
|---|---|---|---|
| CNN float32 | 285 KB | 0.24 ms | see above |
| CNN static int8 (per channel) | 99 KB | 0.07 ms | AUC difference ≤ 0.018 per model, ≤ 0.006 on average over 5 seeds |
| PCA float32 | 1,285 KB | 0.10 ms | AUC 1.000 |
| PCA, int8 weights | 337 KB | 0.12 ms | AUC 1.000, 6.2 % false alarms |

Dynamic int8 quantization of the PCA is discarded: larger file (1,549 KB, a
float32 copy of the components remains) and a threshold that does not
transfer (94 % false alarms on the never-seen temperature). Latencies
measured on a workstation processor: relative orders of magnitude. Not counted:
preprocessing (decimating from 10 MHz costs more than the models; an
embedded system would sample directly at a lower rate).

## C++ export with Aidge
With Eclipse Aidge 0.10.1, the generated float32 C++ is faithful: difference
of 8e-6 on the CNN logit, relative difference of 7e-4 on the PCA indicator
(64 test measurements). Four exact rewrites were needed: normalization
folded into the first convolution, explicit data format (the automatic
format produces a program that crashes), 2D rewrite of the CNN (quantization
does not support MaxPooling1D), PCA without boolean output.
**Int8 C++**: with the original configuration, the generated program
collapses the output, whereas the graph quantized by Aidge is correct in
Python (correlation 0.989 with the float model) and ONNX Runtime quantizes
the same graph without loss (correlation 0.995). A minimal case (one
convolution + ReLU, 15 sizes) shows that **an int8 C++ convolution is exact
only if its numbers of input and output channels are multiples of 16**.
Exact workaround: input padded from 66 to 80 channels with zeros, output
from 1 to 16; the int8 C++ CNN then becomes faithful (correlation 0.989,
68 KB of parameters). Details: `results/step05_aidge_export/`,
`results/step09_aidge_int8_case/`, `step05_aidge_export/`,
`step09_aidge_int8_case/`.

## On a microcontroller (emulated Cortex-M7)
Compilation for Cortex-M7 (arm-none-eabi-gcc 15.2.1, `-O3`) and execution
under QEMU (mps2-an500 board); **no physical board**. Sizes read from the
binary; instructions counted with `-icount` and calibrated on a loop of
known length; durations **estimated** at 480 MHz (one instruction per cycle).

| Detector | Flash | Static RAM + input | Instructions per measurement | ≈ ms at 480 MHz |
|---|---|---|---|---|
| PCA k = 5, int8 weights (hand-written C) | 353 KB | 3 KB + 211 KB | 3.25 M | 6.8 |
| CNN int8 (Aidge, padded channels) | 104 KB | 311 KB + 64 KB | 120.65 M | 251 |
| CNN float32 (Aidge) | 291 KB | 317 KB + 211 KB | 107.89 M | 225 |

Outputs checked: PCA within 1.9e-5 of Python; networks identical to the host
program. Without vectorized kernels (CMSIS-NN, not tested), int8 saves
memory but not time.

## Limitations
A single plate, an artificial reversible defect, temperature as the only
nuisance, a single measurement campaign; false alarms at the cold edge of
the learned domain; emulated microcontroller only (no physical board).
Optimal baseline selection and temperature compensation are known
techniques; the contribution of this project is the controlled comparison
of the three approaches and the transfer to embedded targets.

## Reproduce
```
# torch==2.14.0+cpu comes from the PyTorch CPU index
python -m venv .venv && .venv/bin/pip install -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -e . --no-build-isolation
# data: download the 5 zip files of the figshare collection into data/raw/
.venv/bin/python tools/build_index.py
.venv/bin/python tools/extract_frequency.py 40
# all 12 frequencies (about 3.3 GB each), needed by Step 7
for f in 60 80 100 120 140 160 180 200 220 240 260; do .venv/bin/python tools/extract_frequency.py $f; done
.venv/bin/python -m pytest -q
.venv/bin/python step01_exploration/exploration.py
.venv/bin/python tools/frequency_contrast.py > results/step01_exploration/frequency_contrast.md   # frequency choice table
.venv/bin/python step02_baseline/evaluate_baseline.py
.venv/bin/python step03_cnn_vs_pca/train_cnn.py --phase validation
.venv/bin/python step03_cnn_vs_pca/train_cnn.py --phase validation --config regularized
.venv/bin/python step03_cnn_vs_pca/train_cnn.py --phase test
.venv/bin/python step03_cnn_vs_pca/evaluate_pca.py --phase validation
.venv/bin/python step03_cnn_vs_pca/evaluate_pca.py --phase test
.venv/bin/python step03_cnn_vs_pca/summary.py
.venv/bin/python step04_embedded_export/embedded_export.py
# Aidge: separate environment
python -m venv .venv-aidge && .venv-aidge/bin/pip install -r requirements-aidge.txt
.venv/bin/python step05_aidge_export/prepare_data.py
.venv-aidge/bin/python step05_aidge_export/export_aidge.py
.venv-aidge/bin/python step05_aidge_export/diagnostic_int8.py
.venv/bin/python step05_aidge_export/check_onnxruntime.py
.venv/bin/python step06_localization/localize.py
.venv/bin/python step07_frequencies/frequencies.py
.venv-aidge/bin/python step09_aidge_int8_case/minimal_case.py
.venv/bin/python step08_cortex_m7/build.py    # xPack arm-none-eabi-gcc and QEMU tools in arm_tools/
.venv/bin/python step10_autoencoders/autoencoders.py
```

## Layout
`src/ogw/` (`data`, `splits`, `baseline`, `models`, `training`, `anomaly`
(PCA), `embedded`, `localization`, `autoencoder`); `tests/`;
`step01_exploration/` to `step10_autoencoders/` (scripts for Steps 1 to 10);
`tools/` (extraction, diagnostics); `results/` (figures, tables, metrics).

## Note
The code was written with the assistance of an AI tool (Claude, Anthropic).

## License
Code under the MIT License (see LICENSE). The Open Guided Waves data are
distributed under CC0 by their authors and are not included.
