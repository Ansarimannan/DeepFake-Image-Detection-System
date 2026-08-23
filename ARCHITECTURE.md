# Architecture

How the pieces fit together, and where each responsibility lives. If you are
looking for *why* a choice was made, read [DECISIONS.md](DECISIONS.md).

## 1. Shape of the system

```
config.yaml
     |
     v
config.py  ── validates, resolves paths, exposes Config ──────────────┐
                                                                      |
data.py                model.py              train.py                 |
  discover images        backbone              stage 1 (frozen)       |
  stratified split  ->   augmentation layer -> stage 2 (fine-tune) ───┤
  manifest.csv           preprocess layer     stage selection         |
  tf.data pipeline       linear logit head     model.keras            |
                                                label_map.json        |
                                                                      v
                       evaluate.py  ── threshold on val, metrics on test ──> metrics.json
                       predict.py   ── loads model + label_map ───────────> decisions
                       gradcam.py   ── logit gradients ───────────────────> gradcam.png
                                                                      ^
                       cli.py  ── one entry point for all of it ───────┘
```

One rule governs the whole system: **config.yaml is the only place a number
lives.** No module hardcodes a learning rate, a path, a threshold or a split
fraction. Any run can be reproduced from `artifacts/<run>/config.used.yaml`.

## 2. Data flow

### 2.1 Discovery and split (`data.py`)

`build_manifest` walks the two class directories once and writes a CSV:

| column | meaning |
|---|---|
| `path` | absolute path to the image |
| `class_name` | `fake` or `real` |
| `label` | **0 = fake, 1 = real**, the training target |
| `difficulty` | `easy` / `mid` / `hard` / `none`, parsed from the CIPLAB filename |
| `region_mask` | 4-bit mask: left eye, right eye, nose, mouth |
| `split` | `train` / `val` / `test` |

The split is stratified per class and seeded, then written to disk. Materialising
it matters: training, evaluation, Grad-CAM and any later analysis all read the
same file, so they cannot disagree about which images are in which split. The
manifest is checked for disjointness on creation.

The `test` split is opened by exactly one module, `evaluate.py`, and only after
the threshold has been chosen on `val`.

### 2.2 Input pipeline (`data.make_dataset`)

```
from_tensor_slices(paths, labels)
  -> map(read + decode + resize)        expensive, deterministic
  -> cache()                            so the decode happens once
  -> shuffle()                          train only; val and test keep file order
  -> batch()
  -> prefetch()
```

The pipeline emits **raw float32 pixels in [0, 255]**. It never normalises and
never augments. Both of those are layers inside the model (see 3.1), which is
what makes the cache safe and the serving path identical to the training path.

## 3. The model (`model.py`)

```
Input (H, W, 3), pixels in [0, 255]
  |
  |-- augmentation      Sequential(RandomFlip, RandomRotation, RandomZoom,
  |                                RandomTranslation, RandomContrast)
  |                     active only when training=True
  |
  |-- preprocess        Rescaling(scale, offset) chosen per backbone
  |                     MobileNetV2:   1/127.5, -1   ->  [-1, 1]
  |                     EfficientNetB0: 1, 0         ->  [0, 255] (it normalises internally)
  |
  |-- backbone(x, training=False)       ImageNet weights, include_top=False
  |                                     training=False pins BatchNorm to inference mode
  |                                     for the whole life of the model
  |-- GlobalAveragePooling2D
  |-- Dropout
  |-- [Dense(u) + Dropout] * len(model.head_units)
  |-- Dense(1, activation=None)         a LOGIT
```

### 3.1 Why preprocessing and augmentation are layers

Keeping them in the pipeline invites two classic failure modes, both of which
this design makes structurally impossible:

- Normalisation drift. If the pipeline rescales to `[0, 1]` while MobileNetV2
  expects `[-1, 1]`, and the inference helper carries its own copy of the
  arithmetic, the three can silently disagree. As a layer there is exactly one
  implementation and the saved model carries it.
- Frozen augmentation. A `cache()` placed after an augmentation `map` stores the
  first epoch's random transforms and replays them for every later epoch. As
  layers they run per batch, per epoch, and switch off automatically at
  inference.

### 3.2 Why the output is a logit

`BinaryCrossentropy(from_logits=True)` is numerically stable (it fuses the
sigmoid into the loss), and Grad-CAM differentiates the logit rather than a
saturated sigmoid whose gradients vanish at high confidence. The cost is that
the raw model output is not a probability, so `predict.py` applies the sigmoid,
and `label_map.json` states `"output": "logit"` so nothing can double-apply it.

## 4. Training (`train.py`)

| Stage | Backbone | Learning rate | Purpose |
|---|---|---|---|
| 1 | frozen | 1e-4 | Fit the head without letting large random gradients touch pre-trained weights |
| 2 | top N layers unfrozen, BatchNorm still frozen | 1e-5 | Adapt the most task-specific features without destroying them |

Recompiling between stages is mandatory: a trainability change only takes effect
when the training function is rebuilt. `initial_epoch` carries the epoch counter
forward so the history and any schedule stay continuous.

**Callback directions are always explicit.** `training.early_stopping.mode` and
`training.checkpoint.mode` are required config keys and `config.py` refuses to
load a file that omits them. Keras' `mode='auto'` infers the direction from the
metric *name* and resolves `val_auc_roc` to *minimise*, which would silently
checkpoint and restore the worst epoch instead of the best.

**Exactly one learning-rate controller runs.** `training.lr_schedule` picks
`reduce_on_plateau`, `cosine` or `none`. Running a scheduler and a plateau
callback together means the scheduler overwrites every reduction at the start of
the next epoch.

**Each stage checkpoints to its own file.** A shared file would be wrong: the
stage-2 callback starts with a fresh "best so far", so a stage-2 epoch worse than
stage 1's best would silently overwrite it. `_select_stage` compares the two on
the monitored metric and copies the genuine winner to `model.keras`.

## 5. Evaluation (`evaluate.py`)

1. Predict on **val**. Choose the operating threshold there (`f1`, `youden` or
   `fixed`).
2. Predict on **test**. Apply that threshold unchanged. Compute everything.
3. Write the threshold into `label_map.json`, because the operating point is
   part of the deployment contract, not an evaluation detail.

Reported for the test split: accuracy against the majority-class baseline with a
bootstrap 95 % interval, balanced accuracy, ROC AUC, PR AUC for each class, log
loss, Brier score, a calibration curve with expected calibration error, the
confusion matrix, per-class precision/recall/F1, accuracy by difficulty tier, and
abstain-band behaviour. Plus `worst_mistakes.csv`, the most confident errors, for
failure analysis.

## 6. Inference (`predict.py`)

`Predictor` loads `model.keras` **and** `label_map.json` together. The contract
carries the class order, image size, colour order, the preprocessing description
and the threshold. The predicted name is always
`class_names[int(probability >= threshold)]`; no class name is ever written as a
literal in the decision path. That is the structural fix for the polarity bug,
and `tests/test_predict.py` asserts `predicted_class == label` across a range of
logits.

## 7. Artifacts

```
artifacts/
  splits/manifest.csv          the split, shared by every command
  <run_name>/
    config.used.yaml           exactly what produced this run
    model.keras                the selected model
    model_stage1.keras         per-stage checkpoints, kept for comparison
    model_stage2.keras
    history_stage1.csv         per-epoch metrics, from CSVLogger
    history_stage2.csv
    train_report.json          parameters, class weights, stage selection, timings
    label_map.json             the deployment contract
    metrics.json               validation and sealed-test results
    evaluation.png             ROC, PR, confusion matrix, calibration
    gradcam.png                attention figure
    worst_mistakes.csv         most confident errors on the test split
```

## 8. Testing

57 tests, no GPU and no dataset required: `conftest.py` writes a synthetic
26-image dataset into a temporary directory and points a temporary `Config` at
it. Each file pins behavioural guarantees that are easy to break silently:

| File | Pins |
|---|---|
| `test_config.py` | split fractions sum to 1; callback `mode` may not be omitted; paths resolve; class order is fake-then-real |
| `test_data.py` | splits are disjoint, stratified and reproducible; val/test are not shuffled; the pipeline emits un-normalised pixels; class weights only when imbalanced |
| `test_model.py` | preprocessing maps [0,255] to [-1,1]; the output is an unsquashed logit; inference is deterministic; unfreezing keeps BatchNorm frozen |
| `test_predict.py` | high probability means real, low means fake, class and label never disagree, thresholds and the abstain band are honoured |
| `test_evaluate.py` | the majority baseline is always reported; intervals bracket the estimate; calibration is sane; **every reported metric is a measured classification statistic** — an AST check bans `np.random.uniform` and `r2_score` from all modules |
