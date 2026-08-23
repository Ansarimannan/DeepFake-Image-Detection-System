# CONTEXT

Running log of what changed, why, and what the measured state of the project is.
Newest entry first. Append a new entry when you change behaviour; do not rewrite
old ones.

Companion documents: [README.md](README.md) for how to run it,
[ARCHITECTURE.md](ARCHITECTURE.md) for how it fits together,
[DECISIONS.md](DECISIONS.md) for why each choice was made.

---

## 2026-08-17 - v2.1.0, standalone notebook, and four ablations that all failed to help

### What changed

1. **`Deepfake Detection.ipynb` at the project root: a self-contained notebook.**
   38 cells, 22 of them code, every function defined inline. It imports nothing from
   `src/`, needs no `PYTHONPATH` and no CLI. Open it, Run All, and it goes data to
   saved model in one pass. It is committed **already executed**, so the outputs and
   the seven figures are visible before running anything.

   Reason for the change: the v2.0.0 deliverable was a package plus a thin driver
   notebook, which meant reading `src/` to understand what any cell did. The request
   was for the notebook to *be* the thing, and that was the right call for the way
   this project is actually used.

   The thin driver `notebooks/walkthrough.ipynb` was deleted. Two notebooks is a
   choice nobody should have to make.

2. **Ablation tooling**: `experiments/derive.py` (generate a config variant by dotted
   override) and `experiments/compare.py` (table of every run, read only from
   `metrics.json`, sorted by threshold-free AUC).

3. **Optional additions behind config flags**: `evaluation.tta_hflip` (average the
   prediction over the image and its mirror) and `model.frequency_branch: srm` (a
   second stream over the SRM high-pass noise residual, fused before the head).

4. **`open.bat` and `requirements-notebook.txt`** restored, since Jupyter was
   removed from the core requirements in v2.0.0 and the notebook needs it.

### Measured result: the notebook's own run

Sealed test split, 306 images:

| Metric | Value |
|---|---|
| Accuracy | 0.6340 (95 % CI 0.582 to 0.686) |
| Majority baseline | 0.5294 |
| Lift | **+0.1046** |
| Balanced accuracy | 0.6296 |
| ROC AUC | 0.6865 (95 % CI 0.629 to 0.742) |
| Brier | 0.2280 |
| Threshold | 0.5616, Youden's J on validation |
| Abstain band | 105 of 306 (34.3 %) routed to review |

Slightly below the v2.0.0 CLI run (0.6405 / 0.6969). That gap is run-to-run noise,
not a regression: the two use the same method and the confidence intervals overlap
almost completely. It is a useful reminder of how wide the error bars are at this
sample size.

### Measured result: four configurations, none better than the baseline

All on the same sealed test split, via `python experiments/compare.py`:

| run | accuracy | 95 % CI | ROC AUC | Brier |
|---|---|---|---|---|
| unfreeze_all (154 layers, not 40) | 0.6176 | [0.565, 0.670] | 0.6974 | 0.2270 |
| **baseline** (224 px, top 40) | **0.6405** | [0.588, 0.693] | 0.6969 | **0.2242** |
| res384 (384 px input) | 0.5817 | [0.526, 0.637] | 0.6864 | 0.2276 |
| srm (frequency branch) | 0.6373 | [0.585, 0.690] | 0.6743 | 0.2334 |

**Every interval overlaps every other interval.** No difference here is established.
The correct summary is that on 1,429 training images these four are the same model
within noise, and the baseline configuration is kept because it is the simplest and
the cheapest, not because it won.

Note that the CONTEXT.md entry below predicted higher input resolution would be
"the change most likely to move the number". It was measured, and it was the
**worst** of the four on accuracy while costing 3x the training time. That
prediction was wrong and is left standing above rather than edited out. See
DECISIONS.md D-021.

### Bugs found and fixed while doing this

| Bug | Fix |
|---|---|
| `cli train --config x.yaml` failed with `unrecognized arguments`; `--config` was top-level only. Two experiment launches exited 2 and produced nothing | `--config` accepted on both sides of the subcommand; `tests/test_cli.py` pins it |
| argparse re-parses a subcommand in its own namespace and copies it over the outer one, so a default on either side clobbered a value given on the other | both `--config` actions use `SUPPRESS`; `parse_args()` fills the gaps |
| Notebook cell 2 called `keras.__version__`, which the TF 2.15 Keras shim does not expose | `getattr(keras, "__version__", tf.__version__)` |
| `compare.py` crashed with `KeyError: 'test'` on the notebook's flatter `metrics.json` | non-CLI runs are reported and skipped |

### Verification performed

| Check | Result |
|---|---|
| `pytest` | 57 passed (was 48) |
| Notebook executed headless, in place | 38 cells, **0 errors**, 22/22 code cells with outputs, 7 embedded figures |
| `experiments/compare.py` | table above, 4 runs |
| CLI still works | `smoke`, `split`, `train`, `evaluate`, `gradcam`, `predict` all exercised |

### Follow-ups

The ranked list in the entry below still stands, minus resolution, which has now
been tested and rejected for this dataset. The remaining untested ideas are face
cropping, region-mask auxiliary supervision, JPEG-recompression augmentation and
cross-dataset evaluation. Given that four changes in a row landed inside the noise,
the logical next step is not another architecture tweak: it is **k-fold
cross-validation**, so that differences of a few points can be distinguished from
sampling noise at all.

---

## 2026-08-17 - v2.0.0, rebuilt as a tested package

The project was rebuilt bottom-up as a tested, config-driven Python package with
a single CLI entry point, replacing the original prototype notebook. The goal of
the rebuild was measurement integrity: a sealed test split, explicit callback
directions, preprocessing and augmentation inside the model, calibrated
probabilities, and every headline number reported with a baseline and a
confidence interval.

### Measured result of this rewrite

Sealed test split, 306 images never seen during training or threshold selection:

| Metric | Value | Notes |
|---|---|---|
| Accuracy | **0.6405** (95 % CI 0.588 to 0.693) | Majority-class baseline 0.5294, so **lift +0.111** |
| Balanced accuracy | 0.6424 | Both classes contribute equally |
| ROC AUC | **0.6969** (95 % CI 0.639 to 0.753) | Chance is 0.5 |
| PR AUC (fake) | 0.6827 | |
| PR AUC (real) | 0.7017 | |
| Log loss | 0.6461 | |
| Brier score | 0.2242 | |
| Expected calibration error | 0.0662 | Mildly over-confident |
| Operating threshold | 0.6116 | Chosen on validation by Youden's J |

Confusion matrix, rows actual, columns predicted, order fake then real:

```
            pred fake   pred real
fake             97          47      recall 0.674, precision 0.606
real             63          99      recall 0.611, precision 0.678
```

Accuracy by CIPLAB difficulty tier of the fake images:

| Tier | n | Accuracy |
|---|---|---|
| easy | 32 | 0.469 |
| mid | 81 | 0.741 |
| hard | 31 | 0.710 |
| none (real images) | 162 | 0.611 |

Worth noting: the model is **worst on the "easy" tier**. "Easy" in this dataset
means easy for a *human* to spot, which is not the same thing as easy for a CNN,
and the tiers have small sample sizes (n = 31 to 32), so these differences are
not strongly significant. It is still the kind of breakdown that a single
accuracy number hides entirely.

With the abstain band enabled (margin 0.10 around the threshold), 98 of 306 test
images (32.0 %) are routed to human review and accuracy on the remaining 208
rises to **0.697**.

### Methodology established by this rewrite

Each point maps to a decision entry and to at least one test.

| Guarantee | Where |
|---|---|
| Three-way stratified split; the test set is sealed until `evaluate` runs | D-003, `test_splits_are_disjoint` |
| Only the training split is shuffled; val/test keep file order | `test_validation_and_test_are_not_shuffled` |
| Preprocessing (`Rescaling(1/127.5, offset=-1)`) is a layer inside the model, so training and serving cannot disagree | D-004, `test_mobilenet_preprocessing_maps_to_minus_one_to_one` |
| Augmentation is a model layer, so caching the pipeline cannot freeze it | D-005, `test_pipeline_yields_raw_pixels_of_the_right_shape` |
| Callback directions (`mode`) are required config keys, never inferred | D-009, `test_callback_mode_must_be_explicit` |
| Exactly one learning-rate controller, selected by config | D-010 |
| BatchNorm stays frozen and in inference mode throughout fine-tuning | D-007, `test_unfreeze_keeps_batchnorm_frozen` |
| The predicted class name is always derived, never a string literal | D-012, `test_class_and_label_never_disagree` |
| Only measured classification metrics are reported; an AST check bans `np.random.uniform` and `r2_score` from every module | D-013, `test_no_random_or_regression_metric_calls` |
| Every headline number ships with the majority baseline, lift and a bootstrap CI | D-014 |
| Calibration is measured (reliability curve, ECE, Brier) and an abstain band is available | D-016 |
| The model is saved as `.keras`, never pickled | D-017 |
| Conservative, forensics-aware augmentation | D-006 |
| No hardcoded paths; everything derives from config or `Path(__file__)` | D-001 |
| Grad-CAM differentiates the logit, not a saturated sigmoid | D-008 |

### Structural changes

- **Added**: `config.yaml`, `src/deepfake/{config,data,model,train,evaluate,predict,gradcam,cli,utils}.py`,
  `tests/` (48 tests), `pytest.ini`, `.gitignore`, `ARCHITECTURE.md`, `DECISIONS.md`, this file.
- **Rewritten**: `README.md`, `requirements.txt`, `setup.bat`, `launcher.bat`.
- **Archived**: the earlier prototype notebook and its helper scripts and
  checkpoints were moved to `legacy/`, which nothing in the project reads.
- **Moved**: the project study and interview guide PDF into `docs/`.
- **Dependencies dropped**: `opencv-python`, `seaborn`, `jupyter`, `nbconvert`,
  `joblib`. Added: `PyYAML`, `pytest`.

### Verification performed

| Check | Result |
|---|---|
| `pytest` | 48 passed |
| `python -m deepfake.cli smoke` | OK, prediction contract asserted |
| `python -m deepfake.cli split` | 2,041 images into 1,429 / 306 / 306, stratified |
| `python -m deepfake.cli train` | 35 epochs, 308 s on CPU, stage 2 selected on val AUC 0.6842 |
| `python -m deepfake.cli evaluate` | Test metrics above, `metrics.json` and `evaluation.png` written |
| `python -m deepfake.cli gradcam` | `gradcam.png` written |

The training log at `artifacts/train_log.txt` reads
`val_auc_roc improved from 0.49837 to 0.50482`, confirming the checkpoint
callback maximises the monitored metric as configured.

### Known limitations, unchanged by this rewrite

1. **The model is weak in absolute terms.** AUC 0.70 is well above chance
   and not deployable. The cause is architectural, not a bug: forgery cues are
   high-frequency and local, and 600 x 600 images downsampled to 224 then global-
   average-pooled lose them. See "Next" below.
2. **The split is stratified by class, not by source identity.** CIPLAB fakes are
   built from the real photographs in the same collection, so identity overlap
   across splits is possible and the numbers may be slightly optimistic.
3. **One dataset, one manipulation family** (Photoshop composites, not GAN
   output). No cross-dataset evaluation has been run, and detectors are known to
   generalise badly across generators.
4. **Full run-to-run determinism is not achieved.** Seeds fix the split, the
   initialisation and the shuffle, but multi-threaded `tf.data` and
   non-deterministic CPU kernels still move the last digits.
5. **Redundant dataset copies remain on disk** (`data/real_fake`,
   `data/real_and_fake_face_detection`), roughly 2,000 duplicated images each.
   Nothing reads them. Delete them by hand if you want the space back; this code
   will not delete a user's images on its own (D-019).

### Next, in the order that should pay off

1. **Raise the input resolution** to 299 or 384. One line in `config.yaml`
   (`image.size`). This is the change most likely to move the number, because it
   directly addresses limitation 1.
2. **Crop to the aligned face** with MTCNN, RetinaFace or MediaPipe, and classify
   the crop at native resolution instead of the whole downsampled frame.
3. **Add a frequency branch**: DCT coefficients, an SRM high-pass filter bank or
   error-level analysis, concatenated with the RGB stream. This is the
   domain-specific move that separates a forensics model from a generic classifier.
4. **Use the region masks already parsed into the manifest** as auxiliary
   multi-label supervision (left eye, right eye, nose, mouth). Cheap regularisation,
   and it makes the Grad-CAM story measurable rather than decorative.
5. **Random JPEG re-compression as augmentation**, which is the forensically
   correct kind of noise to be robust to.
6. **Cross-dataset evaluation** against FaceForensics++ or Celeb-DF. Expect a
   large drop and report it; that is the finding, not a failure.
7. **Stratified k-fold** instead of a single split, given only 2,041 images.

Each of these should be run as a single-variable ablation: derive a variant with
`experiments/derive.py`, set a new `project.run_name`, and compare `metrics.json`.
The run directory keeps `config.used.yaml`, so the comparison is always traceable.

---

## How to keep this file useful

Add an entry when you change behaviour. Keep it to five headings:

```markdown
## YYYY-MM-DD - short title

### What changed
### Why
### Measured result       (numbers from artifacts/<run>/metrics.json, not from memory)
### Verification performed (the exact commands, and their outcomes, including failures)
### Follow-ups
```

Two rules that matter more than the format:

- **Never write a number you did not read out of an artifact file.**
- **Record failures as failures.** A run that got worse is information; a run
  that got worse and was quietly not written down destroys the value of the log.
